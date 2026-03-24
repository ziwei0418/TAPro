# ============================================================
# LoRA (Low-Rank Adaptation) for SegFormer
# ============================================================

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper around a frozen linear layer."""

    def __init__(self, frozen_linear: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        self.frozen = frozen_linear
        in_f = frozen_linear.in_features
        out_f = frozen_linear.out_features
        device = frozen_linear.weight.device
        dtype = frozen_linear.weight.dtype

        self.lora_A = nn.Parameter(torch.empty(rank, in_f, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        with torch.no_grad():
            h = self.frozen(x)
        return h + self.scale * (x @ self.lora_A.T @ self.lora_B.T)


def inject_lora_segformer(model: nn.Module, rank: int = 8, alpha: float = 8.0):
    """Inject LoRA adapters into SegFormer's self-attention Q/V projections."""
    count = 0
    for name, module in model.named_modules():
        if type(module).__name__ == 'SegformerEfficientSelfAttention':
            if isinstance(module.query, nn.Linear):
                module.query = LoRALinear(module.query, rank, alpha)
                count += 1
            if isinstance(module.value, nn.Linear):
                module.value = LoRALinear(module.value, rank, alpha)
                count += 1
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA injected into {count} layers | Trainable params: {lora_params/1e6:.2f}M")
    return model


def inject_lora_vit(model: nn.Module, rank: int = 8, alpha: float = 8.0):
    """Inject LoRA adapters into ViT self-attention Q/V projections (for Phikon, DINOv2, etc.)."""
    count = 0
    for name, module in model.named_modules():
        # Standard ViT attention: module.attention.attention has query/key/value
        if hasattr(module, 'attention') and hasattr(module.attention, 'query'):
            attn = module.attention
            if isinstance(attn.query, nn.Linear):
                attn.query = LoRALinear(attn.query, rank, alpha)
                count += 1
            if isinstance(attn.value, nn.Linear):
                attn.value = LoRALinear(attn.value, rank, alpha)
                count += 1
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA (ViT) injected into {count} layers | Trainable params: {lora_params/1e6:.2f}M")
    return model
