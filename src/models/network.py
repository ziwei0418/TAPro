"""
Main network architecture: ClsNetwork
"""
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import trunc_normal_

from .encoder import UNIEncoder
from .fpn import SimpleFeaturePyramid
from .text_encoder import (
    get_text_prompts,
    get_background_prompts,
    encode_text_prompts,
)
from src.utils.hierarchical import merge_to_parent_predictions


class AdaptiveLayer(nn.Module):
    def __init__(self, in_dim, n_ratio, out_dim):
        super().__init__()
        hidden_dim = int(in_dim * n_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ClsNetwork(nn.Module):
    def __init__(
        self,
        hf_token=None,
        cls_num_classes=4,
        fpn_out_dims=(64, 128, 320, 512),
        n_ratio=0.5,
        l_fea_path=None,
        use_lora=True,
        use_text_proto=False,
        dataset_name="bcss",
        text_prompt_mode="pathology",
        text_encoder_mode="auto",
        use_hybrid_proto_blend=False,
        hybrid_alpha_mode="learnable",
        hybrid_alpha_fixed=0.5,
        hybrid_alpha_parent_values=None,
        use_bg_prompt_suppression=False,
        bg_prompt_mode="pathology",
    ):
        super().__init__()
        self.cls_num_classes = cls_num_classes
        self.in_channels = list(fpn_out_dims)

        self.encoder = UNIEncoder(hf_token=hf_token, img_size=224, use_lora=use_lora)
        self.sfp = SimpleFeaturePyramid(in_dim=1024, out_dims=fpn_out_dims)
        self.pooling = F.adaptive_avg_pool2d

        self.l_fc1 = AdaptiveLayer(512, n_ratio, self.in_channels[0])
        self.l_fc2 = AdaptiveLayer(512, n_ratio, self.in_channels[1])
        self.l_fc3 = AdaptiveLayer(512, n_ratio, self.in_channels[2])
        self.l_fc4 = AdaptiveLayer(512, n_ratio, self.in_channels[3])

        with open(l_fea_path, "rb") as lf:
            info = pkl.load(lf)
            self.register_buffer("visual_proto", info["features"].cpu())
            self.k_list = info["k_list"]
            self.cumsum_k = info["cumsum_k"]

        self.total_classes = sum(self.k_list)

        self.use_text_proto = use_text_proto
        self.use_hybrid_proto_blend = use_hybrid_proto_blend
        self.hybrid_alpha_mode = hybrid_alpha_mode.lower()
        self.hybrid_alpha_fixed = float(hybrid_alpha_fixed)
        self.use_bg_prompt_suppression = use_bg_prompt_suppression

        subclass_to_parent = []
        for parent_idx, k in enumerate(self.k_list):
            subclass_to_parent.extend([parent_idx] * k)
        self.register_buffer("subclass_to_parent", torch.tensor(subclass_to_parent, dtype=torch.long))

        if self.use_text_proto:
            text_prompts = get_text_prompts(dataset_name, prompt_mode=text_prompt_mode)
            text_emb = encode_text_prompts(text_prompts, encoder_mode=text_encoder_mode)
            self.register_buffer("text_proto", text_emb)
            num_parents = len(self.k_list)
            if self.hybrid_alpha_mode == "learnable":
                self.proto_alpha_raw = nn.Parameter(torch.zeros(num_parents))
                alpha_desc = "per-parent learnable alpha, init=0.5"
            elif self.hybrid_alpha_mode == "fixed":
                fixed_alpha = torch.full((num_parents,), self.hybrid_alpha_fixed, dtype=torch.float32)
                if hybrid_alpha_parent_values is not None:
                    fixed_alpha = torch.tensor(list(hybrid_alpha_parent_values), dtype=torch.float32)
                    if fixed_alpha.numel() != num_parents:
                        raise ValueError(
                            f"hybrid_alpha_parent_values must have {num_parents} values, "
                            f"got {fixed_alpha.numel()}."
                        )
                fixed_alpha = fixed_alpha.clamp(0.0, 1.0)
                self.register_buffer("proto_alpha_fixed", fixed_alpha)
                alpha_desc = f"fixed alpha={fixed_alpha.tolist()}"
            else:
                raise ValueError(f"Unsupported hybrid_alpha_mode={hybrid_alpha_mode}")
            print(f"Hybrid text-visual prototypes enabled ({alpha_desc})")
            print(f"  Text prompts: {len(text_prompts)}, Visual protos: {self.visual_proto.shape[0]}")
            print(f"  Alpha shape: [{num_parents}] (one per parent class)")
            print(f"  Alpha mode: {self.hybrid_alpha_mode}")
            print(f"  Prompt mode: {text_prompt_mode} | Text encoder mode: {text_encoder_mode}")
            print(f"  Use hybrid prototype blending in forward: {self.use_hybrid_proto_blend}")
            with torch.no_grad():
                text_emb_cpu = text_emb.cpu().float()
                vp_cpu = self.visual_proto.cpu().float()
                vp_norm = vp_cpu / (vp_cpu.norm(dim=-1, keepdim=True) + 1e-8)
                cross_sim = text_emb_cpu @ vp_norm.t()
                diag = cross_sim.diag()
                class_names = ["TUM"] * 3 + ["STR"] * 3 + ["LYM"] * 3 + ["NEC"] * 3 if dataset_name == "bcss" else None
                print("  -- Cross-modal alignment (text<->visual cosine sim) --")
                for i, sim_val in enumerate(diag):
                    flag = "OK" if sim_val > 0.3 else "WARN" if sim_val > 0.1 else "BAD"
                    prefix = class_names[i] if class_names is not None else f"C{i}"
                    print(f"    [{flag}] {prefix}[{i%3}]: {sim_val:.4f}")
                print(f"  Mean same-subclass sim: {diag.mean():.4f}")
                text_sim = text_emb_cpu @ text_emb_cpu.t()
                off_diag = text_sim[~torch.eye(text_sim.shape[0], dtype=bool)]
                print(f"  Text-text off-diag sim: mean={off_diag.mean():.4f}, max={off_diag.max():.4f}")
                if off_diag.max() > 0.85:
                    print(f"  WARNING: Some text prompts are near-duplicates (max={off_diag.max():.4f})")

        if self.use_bg_prompt_suppression:
            bg_prompts = get_background_prompts(dataset_name, prompt_mode=bg_prompt_mode)
            if len(bg_prompts) != len(self.k_list):
                raise ValueError(
                    f"Background prompts must match number of parent classes. "
                    f"Got {len(bg_prompts)} prompts for {len(self.k_list)} parent classes."
                )
            bg_emb = encode_text_prompts(bg_prompts, encoder_mode=text_encoder_mode)
            self.register_buffer("bg_prompt_init", bg_emb)
            self.bg_visual_proto = nn.Parameter(bg_emb.clone())
            print("Class-associated background prompt suppression enabled")
            print(f"  Background prompts: {len(bg_prompts)} (one per parent class)")
            print(f"  Background prompt mode: {bg_prompt_mode} | Text encoder mode: {text_encoder_mode}")
            print("  Background branch uses text-guided visual prototypes (text anchor + learnable visual proto)")
            with torch.no_grad():
                bg_norm = bg_emb.float() / (bg_emb.float().norm(dim=-1, keepdim=True) + 1e-8)
                if getattr(self, "use_text_proto", False):
                    fg_parent = []
                    start = 0
                    for k in self.k_list:
                        fg_parent.append(self.text_proto[start:start + k].float().mean(dim=0))
                        start += k
                    fg_parent = torch.stack(fg_parent, dim=0)
                    fg_parent = fg_parent / (fg_parent.norm(dim=-1, keepdim=True) + 1e-8)
                    sim = bg_norm @ fg_parent.t()
                    print("  -- Background prompt vs parent foreground text similarity --")
                    for idx in range(sim.shape[0]):
                        row = " ".join([f"{v:.3f}" for v in sim[idx]])
                        print(f"    BG[{idx}] -> {row}")

        self.l_fea = self.visual_proto

        self.logit_scale1 = nn.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale2 = nn.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale3 = nn.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale4 = nn.Parameter(torch.ones([1]) * 1 / 0.07)

    def get_param_groups(self):
        regularized = []
        not_regularized = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]

    def aggregate_parent_predictions(self, predictions, method="max"):
        return merge_to_parent_predictions(predictions, self.k_list, method=method)

    def _get_parent_alpha(self, device):
        if self.hybrid_alpha_mode == "learnable":
            return torch.sigmoid(self.proto_alpha_raw).to(device)
        return self.proto_alpha_fixed.to(device)

    def _build_hybrid_prototypes(self, device):
        visual = F.normalize(self.visual_proto.to(device), dim=-1)
        text = F.normalize(self.text_proto.to(device), dim=-1)

        parent_alpha = self._get_parent_alpha(device)
        subclass_alpha = []
        for parent_idx, k in enumerate(self.k_list):
            subclass_alpha.extend([parent_alpha[parent_idx]] * k)
        subclass_alpha = torch.stack(subclass_alpha).unsqueeze(-1)

        hybrid = subclass_alpha * text + (1.0 - subclass_alpha) * visual
        return F.normalize(hybrid, dim=-1)

    def get_hybrid_alpha_analysis(self):
        if not getattr(self, "use_text_proto", False):
            return None
        with torch.no_grad():
            alpha = self._get_parent_alpha(device=self.visual_proto.device).detach().float().cpu()
            if len(self.k_list) == 4:
                names = ["TUM", "STR", "LYM", "NEC"]
            else:
                names = [f"C{i}" for i in range(len(self.k_list))]
            return {
                "mode": self.hybrid_alpha_mode,
                "values": {name: alpha[i].item() for i, name in enumerate(names)},
                "mean": alpha.mean().item(),
                "min": alpha.min().item(),
                "max": alpha.max().item(),
            }

    def get_background_prompt_bank(self, device=None):
        if not getattr(self, "use_bg_prompt_suppression", False):
            raise RuntimeError("Background prompt bank requested, but use_bg_prompt_suppression is disabled.")
        proto = self.bg_visual_proto if device is None else self.bg_visual_proto.to(device)
        return F.normalize(proto, dim=-1)

    def get_background_prompt_analysis(self):
        if not getattr(self, "use_bg_prompt_suppression", False):
            return None
        with torch.no_grad():
            init = F.normalize(self.bg_prompt_init.detach().float().cpu(), dim=-1)
            cur = F.normalize(self.bg_visual_proto.detach().float().cpu(), dim=-1)
            drift = 1.0 - F.cosine_similarity(cur, init, dim=-1)
            inter = cur @ cur.t()
            off_diag = inter[~torch.eye(inter.shape[0], dtype=torch.bool)]
            return {
                "drift_mean": drift.mean().item(),
                "drift_max": drift.max().item(),
                "inter_mean": off_diag.mean().item() if off_diag.numel() > 0 else 0.0,
                "inter_max": off_diag.max().item() if off_diag.numel() > 0 else 0.0,
            }

    def _compute_scale_logits(self, feats, proto, logit_scale):
        """Compute subclass logits from feature maps and prototypes."""
        bsz, ch, h, w = feats.shape
        tokens = feats.permute(0, 2, 3, 1).reshape(bsz, h * w, ch)
        tokens = F.normalize(tokens, dim=-1)
        logits = logit_scale * (tokens @ proto.t().float())
        return logits.view(bsz, h, w, -1).permute(0, 3, 1, 2)

    def forward(self, x):
        uni_feats = self.encoder(x)
        _x = self.sfp(uni_feats)

        logit_scale1 = self.logit_scale1
        logit_scale2 = self.logit_scale2
        logit_scale3 = self.logit_scale3
        logit_scale4 = self.logit_scale4

        if getattr(self, "use_text_proto", False) and getattr(self, "use_hybrid_proto_blend", False):
            base_l_fea = self._build_hybrid_prototypes(x.device)
        else:
            base_l_fea = F.normalize(self.visual_proto.to(x.device), dim=-1)
        l_fea = base_l_fea
        aux_losses = {}

        l_fea1 = self.l_fc1(l_fea)
        l_fea2 = self.l_fc2(l_fea)
        l_fea3 = self.l_fc3(l_fea)
        l_fea4 = self.l_fc4(l_fea)

        out1 = self._compute_scale_logits(_x[0], l_fea1, logit_scale1)
        cam1 = out1.clone().detach()
        cls1 = self.pooling(out1, (1, 1)).view(-1, self.total_classes)

        out2 = self._compute_scale_logits(_x[1], l_fea2, logit_scale2)
        cam2 = out2.clone().detach()
        cls2 = self.pooling(out2, (1, 1)).view(-1, self.total_classes)

        out3 = self._compute_scale_logits(_x[2], l_fea3, logit_scale3)
        cam3 = out3.clone().detach()
        cls3 = self.pooling(out3, (1, 1)).view(-1, self.total_classes)

        out4 = self._compute_scale_logits(_x[3], l_fea4, logit_scale4)
        cam4 = out4.clone()
        cls4 = self.pooling(out4, (1, 1)).view(-1, self.total_classes)

        return cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, self.k_list, aux_losses
