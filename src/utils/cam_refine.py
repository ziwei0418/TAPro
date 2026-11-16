"""特征区域（feature-region）CAM 精修工具。

实现论文中的「边界感知图扩散（BAGD, Boundary-Aware Graph Diffusion）」：
1. apply_cam_power_calibration：对融合后的 CAM 做逐类别幂校准（均衡）；
2. feature_region_refine_scores：在 F4 特征 token 图上做一步残差扩散，
   并用注意力上下文距离抑制跨边界的分数泄漏。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_blend_tensor(
    low_scores: torch.Tensor,
    channels: int,
    cfg,
) -> torch.Tensor:
    """构造逐类别的扩散强度 λ_c（即论文公式 (12) 里的 lambda_c）。

    优先用 feature_region_class_blends（逐类别列表）；否则用统一的 feature_region_blend。
    """
    class_blends = getattr(cfg.train, "feature_region_class_blends", None)
    if class_blends is not None:
        class_blends = list(class_blends)
        # 若列表长度为 channels-1，则视为「只给了前景类别」，前面补一个 0 给背景
        if len(class_blends) == channels - 1:
            class_blends = [0.0] + class_blends
        blend_tensor = low_scores.new_tensor(class_blends).view(1, -1, 1)
        if blend_tensor.shape[1] != channels:
            raise ValueError(
                "feature_region_class_blends must match fused CAM channels: "
                f"got {blend_tensor.shape[1]}, expected {channels} "
                f"(or {channels - 1} foreground classes)"
            )
        return blend_tensor

    blend = float(getattr(cfg.train, "feature_region_blend", 0.35))
    return low_scores.new_full((1, channels, 1), blend)


def apply_cam_power_calibration(
    fuse_scores: torch.Tensor,
    cfg,
) -> torch.Tensor:
    """在特征区域精修之前，对融合 CAM 做逐类别的幂校准（论文中的「均衡」步骤）。"""
    if not getattr(cfg.train, "use_cam_power_calibration", False):
        return fuse_scores

    bsz, channels, height, width = fuse_scores.shape
    power_values = list(getattr(cfg.train, "cam_power_values", []))
    if len(power_values) == channels + 1:
        # 某些配置会把背景 BG 作为最后一通道；这里融合后的 CAM 只有前景，故截掉
        power_values = power_values[:channels]
    if len(power_values) != channels:
        raise ValueError(
            "cam_power_values must match fused CAM channels: "
            f"got {len(power_values)}, expected {channels} "
            f"(or {channels + 1} including BG)"
        )

    blend = float(getattr(cfg.train, "cam_power_blend", 0.30))
    blend = max(0.0, min(1.0, blend))
    eps = float(getattr(cfg.train, "cam_power_eps", 1e-6))

    # 逐类别做幂运算，再按类别归一化，重塑类别间的竞争关系
    powers = fuse_scores.new_tensor(power_values).view(1, channels, 1, 1)
    scores = fuse_scores.clamp_min(eps)
    calibrated = scores.pow(powers)
    calibrated = calibrated / calibrated.sum(dim=1, keepdim=True).clamp_min(eps)

    # 保留原始每个像素的前景总质量，使校准只改变类间相对强弱，
    # 而不改变前景 vs 背景的整体强度。
    original_mass = fuse_scores.sum(dim=1, keepdim=True)
    calibrated = calibrated * original_mass
    return (1.0 - blend) * fuse_scores + blend * calibrated.view(bsz, channels, height, width)


def feature_region_refine_scores(
    fuse_scores: torch.Tensor,
    feature_map: torch.Tensor,
    cfg,
    attention_distance: torch.Tensor | None = None,
) -> torch.Tensor:
    """在基础模型特征 token 分辨率上，对 CAM 分数做一步残差图扩散。

    对应论文公式 (9)(11)(12)：
      K_ij = <f_i, f_j>，N(i) = {j: K_ij ≥ δ} ∪ {i}
      L_ij = K_ij/τ - β·D_ij，W_ij = softmax_{j∈N(i)}(L_ij)
      z_c = (1-λ_c)·s_c + λ_c·W·s_c

    Args:
        fuse_scores: 融合后的 CAM 分数，形状 [B,C,H,W]。
        feature_map: 特征 token，形状 [B,D,h,w]，通常是 FPN 第 4 尺度（F4）。
        attention_distance: 注意力上下文距离矩阵 D_ij，形状 [*, n, n]；为 None 时跳过边界抑制。
    """
    if not getattr(cfg.train, "use_feature_region_refine", False):
        return fuse_scores

    bsz, channels, height, width = fuse_scores.shape
    _, _, fh, fw = feature_map.shape
    num_tokens = fh * fw

    # 把 CAM 分数下采样到图分辨率（7x7）并展平为 token 序列
    low_scores = F.interpolate(fuse_scores, size=(fh, fw), mode="bilinear", align_corners=True)
    low_scores = low_scores.flatten(2)

    # 归一化特征 token，计算特征亲和度 K_ij = <f_i, f_j>
    # 先对每个 token 的特征向量做 L2 归一化，再做内积，所以 sim[i,j] = <f_i, f_j> 就是余弦相似度。
    feats = F.normalize(feature_map.flatten(2).float(), dim=1)
    sim = torch.einsum("bdn,bdm->bnm", feats, feats)

    # 邻域构造：K_ij ≥ δ（sim_threshold），并强制包含自环 {i}
    # 只保留「特征足够相似」的邻居边，外加自己到自己。
    sim_threshold = float(getattr(cfg.train, "feature_region_sim_threshold", 0.55))
    temp = float(getattr(cfg.train, "feature_region_temp", 0.10))
    mask = sim >= sim_threshold
    eye = torch.eye(num_tokens, device=sim.device, dtype=torch.bool).unsqueeze(0)
    mask = mask | eye

    # 扩散强度 λ_c（逐类别） 
    # 每个类别一个 λ（[0.35, 0.35, 0.20, 0.45]），控制该类被平滑的强度。
    blend_tensor = _as_blend_tensor(low_scores, channels, cfg)

    # 扩散 logits：K_ij/τ - β·D_ij
    # K_ij/τ 是局部外观亲和度，相似的特征互相扩散；
    # β·D_ij 是注意力上下文距离，当两个 token 参与不同的全局注意力上下文时 D_ij 很大，把它减掉就能压低跨越这些「上下文相异边」的扩散概率——即使它们特征长得像（比如交界两侧的组织）。
    logits = sim / max(temp, 1e-6)
    if attention_distance is not None and getattr(cfg.train, "use_attention_boundary_refine", False):
        beta = float(getattr(cfg.train, "attention_boundary_beta", 2.0))
        if attention_distance.shape[-2:] != logits.shape[-2:]:
            raise ValueError(
                "attention_distance must match TAGD token graph shape: "
                f"got {tuple(attention_distance.shape)}, expected (*, {num_tokens}, {num_tokens})"
            )
        logits = logits - beta * attention_distance.to(device=logits.device, dtype=logits.dtype)

    # 邻域外的边置为极小值，再做 softmax 得到扩散权重 W_ij
    weights = logits.masked_fill(~mask, -1e4)
    weights = torch.softmax(weights, dim=-1)
    # W·s_c：按注意力权重聚合邻域分数
    smoothed = torch.einsum("bcn,btn->bct", low_scores, weights)

    # 一步残差扩散：z_c = (1-λ_c)·s_c + λ_c·W·s_c
    refined_low = (1.0 - blend_tensor) * low_scores + blend_tensor * smoothed

    # 上采样回原始 CAM 分辨率
    refined_low = refined_low.view(bsz, channels, fh, fw)
    return F.interpolate(refined_low, size=(height, width), mode="bilinear", align_corners=True)
