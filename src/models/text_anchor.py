"""文本锚定的语义原型（Text-Anchored Prototype）。

参照 ReSet 的「text-anchored semantic prototypes」思想，但把 ReSet 里
「离线 JSON 通道排序 + 可学习软阈值」升级成「真文本编码器驱动的可微通道选择」：

  saliency = (W · text) ⊙ visual          # 文本→视觉映射 + 视觉联合门控
  mask     = sigmoid((saliency - θ) / τ)   # 逐子类可学习阈值软掩码
  proto    = normalize(visual ⊙ mask)      # 语义掩码作为类别原型

注意：不用 argsort 做 rank（不可微），而是对连续显著性打分直接软阈值，
这样映射层 W 才能通过掩码拿到梯度。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextAnchoredPrototype(nn.Module):
    """用文本原型锚定视觉原型的通道选择，输出掩蔽后的类别原型。

    Args:
        dim: 原型维度（TAPro 中文本/视觉原型均为 512）。
        num_subclasses: 子类别总数（如 BCSS 4 父类 × 3 子类 = 12）。
        tau: 软阈值温度，越小掩码越锐利。
        map_bias: 文本→视觉映射层是否带 bias。
    """

    def __init__(self, dim=512, num_subclasses=12, tau=0.1, map_bias=False):
        super().__init__()
        self.dim = dim
        self.tau = tau
        # 文本→视觉通道映射（跨模态投影头）
        self.map = nn.Linear(dim, dim, bias=map_bias)
        # 每个子类一个直接可学习阈值（初值 0 -> mask≈0.5）
        self.threshold_raw = nn.Parameter(torch.zeros(num_subclasses))

    def forward(self, text_proto, visual_proto):
        """text_proto / visual_proto: [S, dim]，S = num_subclasses。"""
        t = F.normalize(text_proto, dim=-1)  # 12 * 512
        v = F.normalize(visual_proto, dim=-1)  # 12 * 512
        # 文本→视觉映射后与视觉原型逐通道联合门控，得到通道显著性
        saliency = self.map(t) * v # 12 * 512
        # 可学习阈值软掩码（连续打分，全程可微）
        mask = torch.sigmoid((saliency - self.threshold_raw.unsqueeze(1)) / self.tau)  # 12 * 512
        # 掩蔽后的视觉原型作为类别原型，L2 归一化
        proto = F.normalize(visual_proto * mask, dim=-1)  # 12 * 512
        return proto, mask
