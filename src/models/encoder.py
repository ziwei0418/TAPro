"""
UNI Encoder with LoRA adapters
"""
import math
import torch
import torch.nn as nn
import timm
from huggingface_hub import login as hf_login

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for a frozen nn.Linear.

    output = frozen_linear(x) + scale * (x @ A.T @ B.T)
    Only A and B are trainable.  scale = alpha / rank.
    """

    def __init__(self, frozen_linear: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        self.frozen = frozen_linear  # kept frozen
        in_f, out_f = frozen_linear.in_features, frozen_linear.out_features
        # Match device and dtype of frozen layer to avoid CPU/GPU mismatch
        device = frozen_linear.weight.device
        dtype = frozen_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, in_f, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        # Frozen path (no grad)
        with torch.no_grad():
            h = self.frozen(x)
        # LoRA path (trainable)
        h = h + self.scale * (x @ self.lora_A.T @ self.lora_B.T)
        return h


class UNIEncoder(nn.Module):
    """Frozen UNI ViT-L/16 feature extractor with LoRA adapters.
    Extracts intermediate features at 4 block positions.
    LoRA is injected into attn.qkv of the target blocks.
    All outputs share the same spatial size: [B, 1024, grid, grid].
    """
    BLOCK_IDXS = (11, 17, 23)   # ViT-L has 24 blocks (Block 5 removed — too shallow)
    EMBED_DIM  = 1024

    def __init__(self, hf_token, img_size=224, lora_rank=8, lora_alpha=8.0, use_lora=True):
        super().__init__()
        hf_login(token=hf_token, add_to_git_credential=False)
        self.model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
            img_size=img_size,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.grid_size = img_size // self.model.patch_embed.patch_size[0]  # 14

        # --- Inject LoRA into target blocks if enabled ---
        if use_lora:
            self._inject_lora(lora_rank, lora_alpha)

    def _inject_lora(self, rank, alpha):
        """Replace attn.qkv in target blocks with LoRA-wrapped version."""
        for idx in self.BLOCK_IDXS:
            block = self.model.blocks[idx]
            original_qkv = block.attn.qkv  # frozen nn.Linear(1024, 3072)
            block.attn.qkv = LoRALinear(original_qkv, rank=rank, alpha=alpha)

    def forward(self, x):
        """x: [B, 3, H, W] -> list of 4 tensors [B, 1024, G, G]"""
        B = x.shape[0]
        G = self.grid_size
        D = self.EMBED_DIM

        # Use forward hooks on specific blocks — reliable across all timm versions
        captured = {}

        def make_hook(block_idx):
            def hook(module, inp, out):
                # out may be Tensor [B, N, D] or tuple; extract tensor
                t = out[0] if isinstance(out, tuple) else out
                # Strip CLS prefix token if N = G*G+1
                if t.ndim == 3 and t.shape[1] == G * G + 1:
                    t = t[:, 1:, :]
                captured[block_idx] = t.reshape(B, G, G, D).permute(0, 3, 1, 2).contiguous()
            return hook

        hooks = [
            self.model.blocks[idx].register_forward_hook(make_hook(idx))
            for idx in self.BLOCK_IDXS
        ]
        try:
            self.model(x)
        finally:
            for h in hooks:
                h.remove()

        return [captured[idx] for idx in self.BLOCK_IDXS]  # 4 x [B, 1024, 14, 14]
