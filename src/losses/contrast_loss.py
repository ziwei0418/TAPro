"""
Contrastive loss functions for WSSS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELossFG(nn.Module):
    """InfoNCE loss for foreground features"""
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        print(f'Use InfoNCELossFG with temperature: {temperature}')

    def forward(self, fg_img_feature, fg_pro_feature, bg_pro_feature):
        positive_sims = torch.zeros(1, device=fg_img_feature.device)
        negative_sims = torch.zeros(1, device=fg_img_feature.device)

        fg_img_feature = fg_img_feature / fg_img_feature.norm(dim=-1, keepdim=True)  # [N, D]
        
        batch_size = fg_img_feature.shape[0]
        for i in range(batch_size):
            curr_fg_img = fg_img_feature[i:i+1]  # [1, D]
            curr_fg_pro = fg_pro_feature[i:i+1]  # [1, D]
            curr_bg_pro = bg_pro_feature[i]  # [L, D]
            
            fg_img_fg_pro_logits = curr_fg_img @ curr_fg_pro.t()  # [1, 1]
            fg_img_bg_pro_logits = curr_fg_img @ curr_bg_pro.t()  # [1, L]
            
            positive_sims = positive_sims + torch.exp(fg_img_fg_pro_logits / self.temperature).sum()
            
            negative_sims = negative_sims + \
                           torch.exp(fg_img_fg_pro_logits / self.temperature).sum() + \
                           torch.exp(fg_img_bg_pro_logits / self.temperature).sum()
        loss = -torch.log(positive_sims / negative_sims)

        return loss


class InfoNCELossBG(nn.Module):
    """InfoNCE loss for background features"""
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        print(f'Use InfoNCELossBG with temperature: {temperature}')

    def forward(self, bg_img_feature, fg_pro_feature, bg_pro_feature):
        positive_sims = torch.zeros(1, device=bg_img_feature.device)
        negative_sims = torch.zeros(1, device=bg_img_feature.device)

        bg_img_feature = bg_img_feature / bg_img_feature.norm(dim=-1, keepdim=True)  # [N, D]
        
        batch_size = bg_img_feature.shape[0]
        for i in range(batch_size):
            curr_bg_img = bg_img_feature[i:i+1]  # [1, D]
            curr_fg_pro = fg_pro_feature[i:i+1]  # [1, D]
            curr_bg_pro = bg_pro_feature[i]  # [L, D]

            bg_img_bg_pro_logits = curr_bg_img @ curr_bg_pro.t()  # [1, L]
            bg_img_fg_pro_logits = curr_bg_img @ curr_fg_pro.t()  # [1, 1]

            positive_sims = positive_sims + torch.exp(bg_img_bg_pro_logits / self.temperature).mean()

            negative_sims = negative_sims + \
                           torch.exp(bg_img_bg_pro_logits / self.temperature).mean() + \
                           torch.exp(bg_img_fg_pro_logits / self.temperature).sum()

        loss = -torch.log(positive_sims / negative_sims)

        return loss


class MultiResConsistencyLoss(nn.Module):
    """Cross-scale CAM consistency via soft KL divergence"""
    def __init__(self, tau=4.0):
        super().__init__()
        self.tau = tau

    def forward(self, cam_fine, cam_coarse):
        # Downsample fine to match coarse spatial size
        cam_fine_down = F.adaptive_avg_pool2d(cam_fine, cam_coarse.shape[2:])
        # Treat CAMs as spatial distributions per class (soft KL)
        p = F.softmax(cam_fine_down.flatten(2) / self.tau, dim=-1).clamp(min=1e-8)
        q = F.softmax(cam_coarse.flatten(2) / self.tau, dim=-1).clamp(min=1e-8)
        # KL(p || q) — p=fine guides q=coarse
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        return kl_loss
