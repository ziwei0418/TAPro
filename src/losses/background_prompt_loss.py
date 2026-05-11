import torch
import torch.nn as nn
import torch.nn.functional as F


class BackgroundPromptSuppressionLoss(nn.Module):
    """
    Class-associated background prompt suppression.

    Objectives:
    - align background image features with the corresponding parent-class background prompt
    - repel foreground image features from that background prompt
    - keep learned background prompts close to their text-encoded initialization for stability
    """

    def __init__(self, temperature=0.07, margin=0.10, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.eps = eps
        print(f"Use BackgroundPromptSuppressionLoss with temperature: {temperature}, margin: {margin}")

    def forward(
        self,
        fg_img_feature,
        bg_img_feature,
        fg_proto_feature,
        bg_prompt_feature,
        parent_indices,
        all_bg_prompt_bank,
        bg_prompt_anchor=None,
    ):
        fg_img_feature = F.normalize(fg_img_feature, dim=-1)
        bg_img_feature = F.normalize(bg_img_feature, dim=-1)
        fg_proto_feature = F.normalize(fg_proto_feature, dim=-1)
        bg_prompt_feature = F.normalize(bg_prompt_feature, dim=-1)
        all_bg_prompt_bank = F.normalize(all_bg_prompt_bank, dim=-1)

        sample_weights = fg_img_feature.new_ones(fg_img_feature.shape[0])
        sample_weights = sample_weights / (sample_weights.mean() + self.eps)

        # 1. Background alignment: background feature should match its parent background prompt
        bg_pos_sim = (bg_img_feature * bg_prompt_feature).sum(dim=-1)  # [N]
        bg_fg_neg_sim = (bg_img_feature * fg_proto_feature).sum(dim=-1)  # [N]
        bg_all_sim = bg_img_feature @ all_bg_prompt_bank.t()  # [N, P]

        same_parent_mask = F.one_hot(parent_indices, num_classes=all_bg_prompt_bank.shape[0]).bool()
        bg_other_prompt_sims = bg_all_sim.masked_fill(same_parent_mask, float("-inf"))

        pos = torch.exp(bg_pos_sim / self.temperature)
        neg_fg = torch.exp(bg_fg_neg_sim / self.temperature)
        neg_bg = torch.exp(bg_other_prompt_sims / self.temperature)
        neg_bg = torch.where(torch.isfinite(neg_bg), neg_bg, torch.zeros_like(neg_bg)).sum(dim=-1)
        align_loss_per = -torch.log(pos / (pos + neg_fg + neg_bg + self.eps) + self.eps)
        align_loss = (align_loss_per * sample_weights).mean()

        # 2. Foreground repulsion: foreground should not align with its background prompt
        fg_bg_sim = (fg_img_feature * bg_prompt_feature).sum(dim=-1)
        repel_loss_per = F.relu(fg_bg_sim - self.margin)
        repel_loss = (repel_loss_per * sample_weights).mean()

        # 3. Prompt stability: keep learned prompts anchored to text-initialized background semantics
        if bg_prompt_anchor is not None:
            bg_prompt_anchor = F.normalize(bg_prompt_anchor, dim=-1)
            stability_loss = (1.0 - F.cosine_similarity(bg_prompt_feature, bg_prompt_anchor, dim=-1)).mean()
        else:
            stability_loss = fg_img_feature.new_tensor(0.0)

        losses = {
            "align": align_loss,
            "repel": repel_loss,
            "stability": stability_loss,
        }
        stats = {
            "bg_align": align_loss.detach(),
            "bg_repel": repel_loss.detach(),
            "bg_stability": stability_loss.detach(),
            "bg_pos_sim": bg_pos_sim.mean().detach(),
            "fg_bg_sim": fg_bg_sim.mean().detach(),
        }
        return losses, stats
