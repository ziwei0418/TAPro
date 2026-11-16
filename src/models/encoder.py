"""
带 LoRA 适配器的 UNI 编码器

把冻结的 UNI ViT-L/16 作为特征提取器，并在指定 block 的 attn.qkv 上注入 LoRA 低秩适配器，
从而以极小的可训练参数量微调编码器。前向返回 4 个 block 位置的中间特征。
"""
import os
import math
import torch
import torch.nn as nn
import timm
from huggingface_hub import login as hf_login

class LoRALinear(nn.Module):
    """对冻结的 nn.Linear 做低秩适配（LoRA）的包装层。

    output = frozen_linear(x) + scale * (x @ A.T @ B.T)
    只有 A 和 B 可训练，scale = alpha / rank。
    """

    def __init__(self, frozen_linear: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        self.frozen = frozen_linear  # 冻结的原始线性层
        in_f, out_f = frozen_linear.in_features, frozen_linear.out_features
        # 使 LoRA 参数与冻结层保持相同的设备与 dtype，避免 CPU/GPU 不匹配
        device = frozen_linear.weight.device
        dtype = frozen_linear.weight.dtype
        # A 用 kaiming 均匀初始化，B 初始化为 0，保证初始时 LoRA 分支输出为 0
        self.lora_A = nn.Parameter(torch.empty(rank, in_f, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        # 冻结分支（不参与梯度计算）
        with torch.no_grad():
            h = self.frozen(x)
        # LoRA 分支（可训练）：x -> A^T -> B^T
        h = h + self.scale * (x @ self.lora_A.T @ self.lora_B.T)
        return h


class UNIEncoder(nn.Module):
    """冻结的 UNI ViT-L/16 特征提取器，注入 LoRA 适配器。

    在 4 个 block 位置提取中间特征；LoRA 只注入到目标 block 的 attn.qkv 上。
    所有输出共享相同的空间尺寸：[B, 1024, grid, grid]。
    """
    BLOCK_IDXS = (5, 11, 17, 23)   # ViT-L 共有 24 个 block，取其中 4 个位置
    EMBED_DIM  = 1024

    def __init__(self, hf_token=None, img_size=224, lora_rank=8, lora_alpha=8.0, use_lora=True):
        super().__init__()
        # 若未显式传入 token，则从环境变量 HF_TOKEN 读取（避免硬编码密钥进仓库）
        if not hf_token:
            hf_token = os.environ.get("HF_TOKEN", "")
        # 登录 HuggingFace，加载受权限保护的 UNI 权重
        # 离线模式下登录会触发 whoami 网络请求而失败；token 与权重已缓存时可安全跳过
        try:
            hf_login(token=hf_token, add_to_git_credential=False)
        except Exception as _e:
            print(f"[UNIEncoder] HF login skipped ({type(_e).__name__}); relying on cached token/weights")
        self.model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
            img_size=img_size,
        )
        # 冻结整个编码器
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.grid_size = img_size // self.model.patch_embed.patch_size[0]  # 224/16=14

        # 若启用 LoRA，则向目标 block 注入 LoRA 适配器
        if use_lora:
            self._inject_lora(lora_rank, lora_alpha)

    def _inject_lora(self, rank, alpha):
        """将目标 block 的 attn.qkv 替换为 LoRA 包装版本。"""
        for idx in self.BLOCK_IDXS:
            block = self.model.blocks[idx]
            original_qkv = block.attn.qkv  # 冻结的 nn.Linear(1024, 3072)
            block.attn.qkv = LoRALinear(original_qkv, rank=rank, alpha=alpha)

    def forward(self, x):
        """x: [B, 3, H, W] -> 4 个特征张量的列表 [B, 1024, G, G]。"""
        B = x.shape[0]
        G = self.grid_size
        D = self.EMBED_DIM

        # 在指定 block 上注册 forward hook，捕获中间输出（兼容所有 timm 版本）
        captured = {}

        def make_hook(block_idx):
            def hook(module, inp, out):
                # out 可能是 Tensor [B, N, D] 或 tuple，这里统一取出张量
                t = out[0] if isinstance(out, tuple) else out
                # 若 N = G*G+1，则去掉开头的 CLS token
                if t.ndim == 3 and t.shape[1] == G * G + 1:
                    t = t[:, 1:, :]
                # 还原为 (B, D, G, G) 的特征图格式
                captured[block_idx] = t.reshape(B, G, G, D).permute(0, 3, 1, 2).contiguous()
            return hook

        hooks = [
            self.model.blocks[idx].register_forward_hook(make_hook(idx))
            for idx in self.BLOCK_IDXS
        ]
        try:
            self.model(x)
        finally:
            # 无论是否异常都移除 hook，避免影响后续前向
            for h in hooks:
                h.remove()

        return [captured[idx] for idx in self.BLOCK_IDXS]  # 4 x [B, 1024, 14, 14]

    def forward_with_attention_distance(self, x, attn_block_idx=23, target_size=7):
        """返回编码器特征，以及由注意力分布计算的池化距离矩阵。

        距离矩阵由所选 block 的自注意力分布计算得到，并将 UNI 的 14x14 token 网格
        池化到 FPN p4 层的 7x7 网格。
        """
        B = x.shape[0]
        G = self.grid_size
        D = self.EMBED_DIM

        captured = {}
        attn_holder = {}

        def make_block_hook(block_idx):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if t.ndim == 3 and t.shape[1] == G * G + 1:
                    t = t[:, 1:, :]
                captured[block_idx] = t.reshape(B, G, G, D).permute(0, 3, 1, 2).contiguous()
            return hook

        def attn_pre_hook(module, inp):
            # 手动复现注意力计算，提取注意力分布
            tokens = inp[0]
            qkv = module.qkv(tokens)
            qkv = qkv.reshape(B, tokens.shape[1], 3, module.num_heads, module.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            q = module.q_norm(q)
            k = module.k_norm(k)
            attn = (q * module.scale) @ k.transpose(-2, -1)
            # 对多头注意力取平均，得到单张注意力图
            attn = attn.softmax(dim=-1).mean(dim=1)
            # 去掉 CLS token 对应的行列
            if attn.shape[1] == G * G + 1:
                attn = attn[:, 1:, 1:]
            attn_holder["attn"] = attn.detach()

        hooks = [
            self.model.blocks[idx].register_forward_hook(make_block_hook(idx))
            for idx in self.BLOCK_IDXS
        ]
        attn_hook = self.model.blocks[attn_block_idx].attn.register_forward_pre_hook(attn_pre_hook)
        try:
            self.model(x)
        finally:
            for h in hooks:
                h.remove()
            attn_hook.remove()

        # 注意力图 -> 每个 token 的注意力分布向量
        attn = attn_holder["attn"]
        profiles = attn.reshape(B, G, G, G * G).permute(0, 3, 1, 2)
        # 池化到 target_size x target_size，并对每个 token 的注意力分布做归一化
        profiles = torch.nn.functional.adaptive_avg_pool2d(profiles, (target_size, target_size))
        profiles = profiles.flatten(2).transpose(1, 2).contiguous()
        profiles = profiles / profiles.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        # 用 L1 距离的一半作为注意力分布之间的距离矩阵
        attention_distance = 0.5 * torch.cdist(profiles.float(), profiles.float(), p=1)
        return [captured[idx] for idx in self.BLOCK_IDXS], attention_distance
