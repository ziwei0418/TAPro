"""
Main network architecture: ClsNetwork
"""
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import UNIEncoder
from .fpn import SimpleFeaturePyramid
from .vq_modules import VQCodebook, VQCAMHead
from .text_encoder import get_text_prompts, encode_text_prompts
from src.losses.contrast_loss import MultiResConsistencyLoss


class AdaptiveLayer(nn.Module):
    """Projects l_fea (512-dim) to match each FPN scale's channel dim."""
    def __init__(self, in_dim, n_ratio, target_dim):
        super().__init__()
        hidden_dim = int(target_dim * n_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, target_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ClsNetwork(nn.Module):
    """Multi-scale WSSS network:
    - Frozen UNI encoder with LoRA adapters
    - Trainable FPN for multi-scale features
    - Hybrid text-visual prototypes (per-class learnable blending)
    - Optional VQ codebook for discrete semantic prototypes
    - Optional multi-resolution consistency loss
    """
    def __init__(
        self,
        hf_token,
        fpn_out_dims=(128, 320, 512),
        l_fea_path=None,
        n_ratio=0.5,
        use_lora=False,
        use_vq=False,
        vq_num_codes=128,
        vq_code_dim=256,
        vq_commitment_weight=0.25,
        vq_cam_weight=0.3,
        use_multires_consistency=False,
        use_text_proto=False,
        dataset_name="bcss",
    ):
        super().__init__()
        self.in_channels = fpn_out_dims  # [128, 320, 512]

        # Frozen UNI encoder with optional LoRA
        self.encoder = UNIEncoder(hf_token=hf_token, img_size=224, use_lora=use_lora)
        # Trainable Simple Feature Pyramid
        self.sfp = SimpleFeaturePyramid(in_dim=1024, out_dims=fpn_out_dims)

        self.pooling = F.adaptive_avg_pool2d

        ## AdaptiveLayer projects l_fea (512-dim) to match each scale
        self.l_fc2 = AdaptiveLayer(512, n_ratio, self.in_channels[0])  # 512 -> 128
        self.l_fc3 = AdaptiveLayer(512, n_ratio, self.in_channels[1])  # 512 -> 320
        self.l_fc4 = AdaptiveLayer(512, n_ratio, self.in_channels[2])  # 512 -> 512

        # Load visual prototypes from pickle file
        with open(l_fea_path, "rb") as lf:
            info = pkl.load(lf)
            self.register_buffer('visual_proto', info['features'].cpu())  # [12, 512]
            self.k_list = info['k_list']
            self.cumsum_k = info['cumsum_k']

        self.total_classes = sum(self.k_list)

        # --- Hybrid Text-Visual Prototypes ---
        self.use_text_proto = use_text_proto
        if self.use_text_proto:
            prompts = get_text_prompts(dataset_name)
            text_emb = encode_text_prompts(prompts)
            self.register_buffer('text_proto', text_emb)  # [12, 512]
            # Per-parent-class learnable blending: each of 4 classes gets its own α
            num_parents = len(self.k_list)  # 4 (TUM, STR, LYM, NEC)
            self.proto_alpha_raw = nn.Parameter(torch.zeros(num_parents))  # [4]
            print(f"Hybrid text-visual prototypes enabled (per-class learnable alpha, init=0.5) [{dataset_name.upper()}]")
            print(f"  Text prompts: {len(prompts)}, Visual protos: {self.visual_proto.shape[0]}")
            print(f"  Alpha shape: [{num_parents}] (one per parent class)")
            # --- Diagnostic: cross-modal alignment quality ---
            with torch.no_grad():
                text_emb_cpu = text_emb.cpu().float()
                vp_cpu = self.visual_proto.cpu().float()
                vp_norm = vp_cpu / (vp_cpu.norm(dim=-1, keepdim=True) + 1e-8)
                cross_sim = text_emb_cpu @ vp_norm.t()  # [12, 12]
                diag = cross_sim.diag()  # same-subclass similarity
                if dataset_name == "luad":
                    class_names = ["TE"]*3 + ["NEC"]*3 + ["LYM"]*3 + ["TAS"]*3
                else:
                    class_names = ["TUM"]*3 + ["STR"]*3 + ["LYM"]*3 + ["NEC"]*3
                print(f"  -- Cross-modal alignment (text<->visual cosine sim) --")
                for i, (name, sim_val) in enumerate(zip(class_names, diag)):
                    flag = "OK" if sim_val > 0.3 else "WARN" if sim_val > 0.1 else "BAD"
                    print(f"    [{flag}] {name}[{i%3}]: {sim_val:.4f}")
                print(f"  Mean same-subclass sim: {diag.mean():.4f}")
                text_sim = text_emb_cpu @ text_emb_cpu.t()  # [12, 12]
                off_diag = text_sim[~torch.eye(12, dtype=bool)]
                print(f"  Text-text off-diag sim: mean={off_diag.mean():.4f}, max={off_diag.max():.4f}")
                if off_diag.max() > 0.85:
                    print(f"  WARNING: Some text prompts are near-duplicates (max={off_diag.max():.4f})")

        # Backward compat: l_fea will be computed in forward()
        self.l_fea = self.visual_proto  # fallback if text_proto not used

        self.logit_scale2 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale3 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale4 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)

        # --- SISH-inspired: VQ Codebook Prototypes ---
        self.use_vq = use_vq
        self.vq_cam_weight = vq_cam_weight
        self.vq_warmup_epochs = 0  # text-guided init handles stability
        self._current_epoch = 0
        _proto_dim = 512  # CONCH/UNI prototype space dimensionality
        if self.use_vq:
            self.vq_head2 = VQCAMHead(self.in_channels[0], self.total_classes,
                                     num_codes=vq_num_codes, code_dim=vq_code_dim,
                                     commitment_weight=vq_commitment_weight,
                                     proto_dim=_proto_dim)
            self.vq_head3 = VQCAMHead(self.in_channels[1], self.total_classes,
                                     num_codes=vq_num_codes, code_dim=vq_code_dim,
                                     commitment_weight=vq_commitment_weight,
                                     proto_dim=_proto_dim)
            self.vq_head4 = VQCAMHead(self.in_channels[2], self.total_classes,
                                     num_codes=vq_num_codes, code_dim=vq_code_dim,
                                     commitment_weight=vq_commitment_weight,
                                     proto_dim=_proto_dim)
            print(f"VQ Codebook enabled: {vq_num_codes} codes, dim={vq_code_dim}, cam_weight={vq_cam_weight}")
            print(f"  Semantic regularization: VQ codes → prototype space ({_proto_dim}d)")

            # --- Text-Guided VQ Initialization ---
            if self.use_text_proto:
                print(f"  Text-guided VQ init: seeding codebooks from {self.text_proto.shape[0]} text prototypes")
                self.vq_head2.init_codebook_from_prototypes(self.text_proto)
                self.vq_head3.init_codebook_from_prototypes(self.text_proto)
                self.vq_head4.init_codebook_from_prototypes(self.text_proto)
            print(f"  VQ warmup: weight=0 for first {self.vq_warmup_epochs} epoch(s)")

        # --- SISH-inspired: Multi-resolution Consistency ---
        self.use_multires_consistency = use_multires_consistency
        if self.use_multires_consistency:
            self.multires_loss_fn = MultiResConsistencyLoss()
            print("Multi-resolution consistency loss enabled")

    def get_param_groups(self):
        """Separate params into regularized (weights) and not-regularized (biases/norms)."""
        regularized = []
        not_regularized = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

    def forward(self, x):
        # --- UNI encoder (frozen weights + trainable LoRA) ---
        uni_feats = self.encoder(x)  # 3 x [B, 1024, 14, 14]

        # --- Trainable SFP ---
        _x = self.sfp(uni_feats)  # P2(28x28), P3(14x14), P4(7x7)

        logit_scale2 = self.logit_scale2
        logit_scale3 = self.logit_scale3
        logit_scale4 = self.logit_scale4

        imshape = [f.shape for f in _x]
        image_features = [f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]) for f in _x]
        _x2, _x3, _x4 = image_features

        # --- Hybrid text-visual prototype blending ---
        if self.use_text_proto:
            # Per-parent-class alpha: [4] → expand to [12] via k_list
            parent_alpha = torch.sigmoid(self.proto_alpha_raw)  # [4], each ∈ (0, 1)
            # Expand: e.g. k_list=[3,3,3,3] → [α0,α0,α0, α1,α1,α1, α2,α2,α2, α3,α3,α3]
            alpha = torch.cat([a.expand(k) for a, k in zip(parent_alpha, self.k_list)])  # [12]
            alpha = alpha.unsqueeze(1)  # [12, 1] for broadcasting with [12, 512]
            l_fea = alpha * self.text_proto + (1 - alpha) * self.visual_proto
        else:
            l_fea = self.visual_proto
        l_fea = l_fea.to(x.device)
        l_fea2 = self.l_fc2(l_fea)
        l_fea3 = self.l_fc3(l_fea)
        l_fea4 = self.l_fc4(l_fea)

        total_vq_loss = torch.tensor(0.0, device=x.device)

        # VQ warmup: ramp up weight over warmup epochs
        if self.use_vq and self._current_epoch < self.vq_warmup_epochs:
            effective_vq_w = 0.0  # VQ off during warmup
        else:
            effective_vq_w = self.vq_cam_weight

        # --- Scale 2 (28x28) ---
        _x2 = _x2 / _x2.norm(dim=-1, keepdim=True)
        logits_per_image2 = logit_scale2 * _x2 @ l_fea2.t().float()
        out2 = logits_per_image2.view(imshape[0][0], imshape[0][2], imshape[0][3], -1).permute(0, 3, 1, 2)
        if self.use_vq:
            vq_cam2, vq_loss2 = self.vq_head2(_x2, imshape[0][2], imshape[0][3])
            total_vq_loss = total_vq_loss + vq_loss2
            out2 = (1 - effective_vq_w) * out2 + effective_vq_w * vq_cam2
        cam2 = out2.clone().detach()
        cls2 = self.pooling(out2, (1, 1)).view(-1, self.total_classes)

        # --- Scale 3 (14x14) ---
        _x3 = _x3 / _x3.norm(dim=-1, keepdim=True)
        logits_per_image3 = logit_scale3 * _x3 @ l_fea3.t().float()
        out3 = logits_per_image3.view(imshape[1][0], imshape[1][2], imshape[1][3], -1).permute(0, 3, 1, 2)
        if self.use_vq:
            vq_cam3, vq_loss3 = self.vq_head3(_x3, imshape[1][2], imshape[1][3])
            total_vq_loss = total_vq_loss + vq_loss3
            out3 = (1 - effective_vq_w) * out3 + effective_vq_w * vq_cam3
        cam3 = out3.clone().detach()
        cls3 = self.pooling(out3, (1, 1)).view(-1, self.total_classes)

        # --- Scale 4 (7x7) ---
        _x4 = _x4 / _x4.norm(dim=-1, keepdim=True)
        logits_per_image4 = logit_scale4 * _x4 @ l_fea4.t().float()
        out4 = logits_per_image4.view(imshape[2][0], imshape[2][2], imshape[2][3], -1).permute(0, 3, 1, 2)
        if self.use_vq:
            vq_cam4, vq_loss4 = self.vq_head4(_x4, imshape[2][2], imshape[2][3])
            total_vq_loss = total_vq_loss + vq_loss4
            out4 = (1 - effective_vq_w) * out4 + effective_vq_w * vq_cam4
        cam4 = out4.clone()
        cls4 = self.pooling(out4, (1, 1)).view(-1, self.total_classes)

        # --- Multi-resolution consistency loss (needs gradients from out, not cam) ---
        multires_loss = torch.tensor(0.0, device=x.device)
        if self.use_multires_consistency:
            multires_loss = (self.multires_loss_fn(out2, out3)
                           + self.multires_loss_fn(out3, out4))

        # --- Semantic regularization loss: link VQ codes to prototypes ---
        sem_reg_loss = torch.tensor(0.0, device=x.device)
        if self.use_vq and self.use_text_proto and self.training:
            sem_reg_loss = (self.vq_head2.compute_sem_reg_loss(l_fea)
                          + self.vq_head3.compute_sem_reg_loss(l_fea)
                          + self.vq_head4.compute_sem_reg_loss(l_fea)) / 3.0

        aux_losses = {
            'vq_loss': total_vq_loss,
            'multires_loss': multires_loss,
            'sem_reg_loss': sem_reg_loss
        }
        return cls2, cam2, cls3, cam3, cls4, cam4, l_fea, self.k_list, aux_losses
