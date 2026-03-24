# ============================================================
# Multi-backbone Segmentation Model
# Supports: SegFormer (mit-b3), Phikon (ViT-B), CTransPath (Swin-T)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig

from .config import CLASS_NAMES
from .lora import inject_lora_segformer, inject_lora_vit


# ── Lightweight UPerNet-style decoder for ViT/Swin backbones ──

class SimpleSegDecoder(nn.Module):
    """Simple FPN + fuse decoder for single-scale or multi-scale features."""

    def __init__(self, in_dims, hidden_dim=256, num_classes=4):
        super().__init__()
        # Lateral convs (project each scale to hidden_dim)
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ) for d in in_dims
        ])
        # Fuse + predict
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * len(in_dims), hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )

    def forward(self, features, target_size):
        """
        features: list of tensors [(B, C_i, H_i, W_i), ...]
        target_size: (H, W) of original input
        """
        # Upsample to intermediate size (H/4, W/4) to save VRAM
        inter_size = (target_size[0] // 4, target_size[1] // 4)
        laterals = []
        for feat, lat_conv in zip(features, self.laterals):
            x = lat_conv(feat)
            x = F.interpolate(x, size=inter_size, mode='bilinear', align_corners=False)
            laterals.append(x)
        fused = torch.cat(laterals, dim=1)
        out = self.fuse(fused)
        return F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)


# ── SegFormer ─────────────────────────────────────────────────

class SegFormerSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        num_classes = cfg.model.num_classes
        backbone_id = cfg.model.backbone

        seg_cfg = SegformerConfig.from_pretrained(backbone_id)
        seg_cfg.num_labels = num_classes
        seg_cfg.id2label = {i: n for i, n in enumerate(CLASS_NAMES)}
        seg_cfg.label2id = {n: i for i, n in enumerate(CLASS_NAMES)}
        seg_cfg.decoder_hidden_size = 512

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            backbone_id, config=seg_cfg, ignore_mismatched_sizes=True,
        )
        for p in self.model.segformer.parameters():
            p.requires_grad_(False)

        self.deep_sup = cfg.model.deep_supervision
        if cfg.model.use_lora:
            inject_lora_segformer(
                self.model.segformer,
                rank=cfg.model.lora_rank, alpha=cfg.model.lora_alpha,
            )

        if self.deep_sup:
            self.aux_head1 = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Dropout2d(0.1), nn.Conv2d(256, num_classes, 1),
            )
            self.aux_head2 = nn.Sequential(
                nn.Conv2d(320, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Dropout2d(0.1), nn.Conv2d(256, num_classes, 1),
            )
        self._print_params()

    def forward(self, x):
        H, W = x.shape[2:]
        enc = self.model.segformer(x, output_hidden_states=True, return_dict=True)
        hs = enc.hidden_states
        logits = self.model.decode_head(hs)
        logits = F.interpolate(logits, (H, W), mode='bilinear', align_corners=False)
        if not self.deep_sup or not self.training:
            return logits, None, None
        aux1 = F.interpolate(self.aux_head1(hs[1]), (H, W), mode='bilinear', align_corners=False)
        aux2 = F.interpolate(self.aux_head2(hs[2]), (H, W), mode='bilinear', align_corners=False)
        return logits, aux1, aux2

    def get_param_groups(self, lr, backbone_scale=0.1):
        lora_p, dec_p, aux_p = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'model.segformer' in name:
                lora_p.append(p)
            elif 'aux_head' in name:
                aux_p.append(p)
            else:
                dec_p.append(p)
        groups = [
            {'params': lora_p, 'lr': lr * backbone_scale, 'name': 'lora'},
            {'params': dec_p, 'lr': lr, 'name': 'decoder'},
        ]
        if aux_p:
            groups.append({'params': aux_p, 'lr': lr, 'name': 'aux'})
        return groups

    def _print_params(self):
        t = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total: {t/1e6:.2f}M | Trainable: {tr/1e6:.2f}M")


# ── Phikon (ViT-B pathology backbone) ────────────────────────

class PhikonSeg(nn.Module):
    """Phikon ViT-B backbone + FPN decoder for segmentation."""

    # ViT-B: 12 blocks, hidden_dim=768, patch_size=16
    EXTRACT_LAYERS = [2, 5, 8, 11]  # multi-scale feature extraction

    def __init__(self, cfg):
        super().__init__()
        from transformers import ViTModel
        num_classes = cfg.model.num_classes
        backbone_id = cfg.model.get('backbone', 'owkin/phikon')

        self.backbone = ViTModel.from_pretrained(backbone_id)
        hidden_dim = self.backbone.config.hidden_size  # 768

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        if cfg.model.use_lora:
            inject_lora_vit(
                self.backbone,
                rank=cfg.model.lora_rank, alpha=cfg.model.lora_alpha,
            )

        # Feature projection: split ViT single-scale into pseudo multi-scale
        self.feat_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.EXTRACT_LAYERS])
        self.feat_projs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim // (2 ** i), 1)
            for i, _ in enumerate(self.EXTRACT_LAYERS)
        ])

        in_dims = [hidden_dim // (2 ** i) for i in range(len(self.EXTRACT_LAYERS))]
        self.decoder = SimpleSegDecoder(in_dims, hidden_dim=256, num_classes=num_classes)

        self.deep_sup = cfg.model.deep_supervision
        if self.deep_sup:
            self.aux_head = nn.Sequential(
                nn.Conv2d(in_dims[1], 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Dropout2d(0.1), nn.Conv2d(128, num_classes, 1),
            )
        self._print_params()

    def _reshape_tokens(self, tokens, H_patches, W_patches):
        """(B, N, C) -> (B, C, H, W) — drop CLS token."""
        B, N, C = tokens.shape
        tokens = tokens[:, 1:, :]  # remove CLS
        return tokens.transpose(1, 2).reshape(B, C, H_patches, W_patches)

    def forward(self, x):
        H, W = x.shape[2:]
        out = self.backbone(x, output_hidden_states=True, return_dict=True,
                             interpolate_pos_encoding=True)
        hidden_states = out.hidden_states  # (n_layers+1,) each (B, N, C)

        patch_size = self.backbone.config.patch_size
        H_p, W_p = H // patch_size, W // patch_size

        features = []
        for i, layer_idx in enumerate(self.EXTRACT_LAYERS):
            tokens = self.feat_norms[i](hidden_states[layer_idx + 1])
            feat = self._reshape_tokens(tokens, H_p, W_p)
            feat = self.feat_projs[i](feat)
            features.append(feat)

        logits = self.decoder(features, (H, W))
        if not self.deep_sup or not self.training:
            return logits, None, None

        aux = F.interpolate(
            self.aux_head(features[1]), (H, W), mode='bilinear', align_corners=False,
        )
        return logits, aux, None

    def get_param_groups(self, lr, backbone_scale=0.1):
        bb_p, dec_p = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'backbone' in name:
                bb_p.append(p)
            else:
                dec_p.append(p)
        return [
            {'params': bb_p, 'lr': lr * backbone_scale, 'name': 'backbone'},
            {'params': dec_p, 'lr': lr, 'name': 'decoder'},
        ]

    def _print_params(self):
        t = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Phikon Total: {t/1e6:.2f}M | Trainable: {tr/1e6:.2f}M")


# ── CTransPath (Swin-T pathology backbone) ───────────────────

class CTransPathSeg(nn.Module):
    """CTransPath Swin-T backbone + FPN decoder for segmentation."""

    def __init__(self, cfg):
        super().__init__()
        from transformers import SwinModel
        num_classes = cfg.model.num_classes
        backbone_id = cfg.model.get('backbone', 'wisdomik/QuiltNet-B-16')

        self.backbone = SwinModel.from_pretrained(backbone_id)

        # Swin-T hidden dims per stage
        hidden_sizes = list(self.backbone.config.hidden_sizes)  # e.g., [96, 192, 384, 768]

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        if cfg.model.use_lora:
            # LoRA for Swin attention Q/V
            count = 0
            from .lora import LoRALinear
            for name, module in self.backbone.named_modules():
                if hasattr(module, 'query') and isinstance(module.query, nn.Linear):
                    module.query = LoRALinear(module.query, cfg.model.lora_rank, cfg.model.lora_alpha)
                    count += 1
                if hasattr(module, 'value') and isinstance(module.value, nn.Linear):
                    module.value = LoRALinear(module.value, cfg.model.lora_rank, cfg.model.lora_alpha)
                    count += 1
            tr = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            print(f"  LoRA (Swin) injected into {count} layers | Trainable: {tr/1e6:.2f}M")

        self.decoder = SimpleSegDecoder(hidden_sizes, hidden_dim=256, num_classes=num_classes)

        self.deep_sup = cfg.model.deep_supervision
        if self.deep_sup:
            self.aux_head = nn.Sequential(
                nn.Conv2d(hidden_sizes[1], 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Dropout2d(0.1), nn.Conv2d(128, num_classes, 1),
            )
        self._print_params()

    def forward(self, x):
        H, W = x.shape[2:]
        out = self.backbone(x, output_hidden_states=True, return_dict=True)

        # Swin hidden states: list of (B, H*W, C) for each stage
        # reshaped_hidden_states: list of (B, C, H, W)
        features = list(out.reshaped_hidden_states)

        logits = self.decoder(features, (H, W))
        if not self.deep_sup or not self.training:
            return logits, None, None

        aux = F.interpolate(
            self.aux_head(features[1]), (H, W), mode='bilinear', align_corners=False,
        )
        return logits, aux, None

    def get_param_groups(self, lr, backbone_scale=0.1):
        bb_p, dec_p = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'backbone' in name:
                bb_p.append(p)
            else:
                dec_p.append(p)
        return [
            {'params': bb_p, 'lr': lr * backbone_scale, 'name': 'backbone'},
            {'params': dec_p, 'lr': lr, 'name': 'decoder'},
        ]

    def _print_params(self):
        t = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  CTransPath Total: {t/1e6:.2f}M | Trainable: {tr/1e6:.2f}M")


# ── Factory ───────────────────────────────────────────────────

BACKBONE_REGISTRY = {
    'segformer': SegFormerSeg,
    'phikon': PhikonSeg,
    'ctranspath': CTransPathSeg,
}


def build_seg_model(cfg):
    """Factory function: build segmentation model based on cfg.model.type."""
    model_type = cfg.model.get('type', 'segformer')
    if model_type not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. Options: {list(BACKBONE_REGISTRY.keys())}")
    print(f"\nBuilding {model_type} model...")
    return BACKBONE_REGISTRY[model_type](cfg)
