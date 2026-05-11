"""Feature-region CAM refinement utilities."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_blend_tensor(
    low_scores: torch.Tensor,
    channels: int,
    cfg,
) -> torch.Tensor:
    class_blends = getattr(cfg.train, "feature_region_class_blends", None)
    if class_blends is not None:
        class_blends = list(class_blends)
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
    """Apply class-wise CAM power calibration before feature-region refinement."""
    if not getattr(cfg.train, "use_cam_power_calibration", False):
        return fuse_scores

    bsz, channels, height, width = fuse_scores.shape
    power_values = list(getattr(cfg.train, "cam_power_values", []))
    if len(power_values) == channels + 1:
        # Some notes include BG as the last channel; fused CAM here has foreground only.
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

    powers = fuse_scores.new_tensor(power_values).view(1, channels, 1, 1)
    scores = fuse_scores.clamp_min(eps)
    calibrated = scores.pow(powers)
    calibrated = calibrated / calibrated.sum(dim=1, keepdim=True).clamp_min(eps)

    # Preserve original per-pixel foreground mass so this reshapes class competition
    # without globally changing foreground-vs-background strength.
    original_mass = fuse_scores.sum(dim=1, keepdim=True)
    calibrated = calibrated * original_mass
    return (1.0 - blend) * fuse_scores + blend * calibrated.view(bsz, channels, height, width)


def feature_region_refine_scores(
    fuse_scores: torch.Tensor,
    feature_map: torch.Tensor,
    cfg,
    attention_distance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Smooth CAM scores over foundation-feature regions at token resolution.

    Args:
        fuse_scores: fused CAM scores shaped [B,C,H,W].
        feature_map: feature tokens shaped [B,D,h,w], typically FPN scale-4.
    """
    if not getattr(cfg.train, "use_feature_region_refine", False):
        return fuse_scores

    bsz, channels, height, width = fuse_scores.shape
    _, _, fh, fw = feature_map.shape
    num_tokens = fh * fw

    low_scores = F.interpolate(fuse_scores, size=(fh, fw), mode="bilinear", align_corners=True)
    low_scores = low_scores.flatten(2)

    feats = F.normalize(feature_map.flatten(2).float(), dim=1)
    sim = torch.einsum("bdn,bdm->bnm", feats, feats)

    sim_threshold = float(getattr(cfg.train, "feature_region_sim_threshold", 0.55))
    temp = float(getattr(cfg.train, "feature_region_temp", 0.10))
    mask = sim >= sim_threshold
    eye = torch.eye(num_tokens, device=sim.device, dtype=torch.bool).unsqueeze(0)
    mask = mask | eye

    blend_tensor = _as_blend_tensor(low_scores, channels, cfg)
    logits = sim / max(temp, 1e-6)
    if attention_distance is not None and getattr(cfg.train, "use_attention_boundary_refine", False):
        beta = float(getattr(cfg.train, "attention_boundary_beta", 2.0))
        if attention_distance.shape[-2:] != logits.shape[-2:]:
            raise ValueError(
                "attention_distance must match TAGD token graph shape: "
                f"got {tuple(attention_distance.shape)}, expected (*, {num_tokens}, {num_tokens})"
            )
        logits = logits - beta * attention_distance.to(device=logits.device, dtype=logits.dtype)

    weights = logits.masked_fill(~mask, -1e4)
    weights = torch.softmax(weights, dim=-1)
    smoothed = torch.einsum("bcn,btn->bct", low_scores, weights)

    refined_low = (1.0 - blend_tensor) * low_scores + blend_tensor * smoothed

    refined_low = refined_low.view(bsz, channels, fh, fw)
    return F.interpolate(refined_low, size=(height, width), mode="bilinear", align_corners=True)
