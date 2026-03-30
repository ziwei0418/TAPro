# ============================================================
# Loss Functions for Segmentation Training
# Includes: Dice, BootstrappedCE, OHEM, Lovász-Softmax
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, smooth=1.0):
        super().__init__()
        self.C = num_classes
        self.ig = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        valid = (targets != self.ig)
        t = targets.clone()
        t[~valid] = 0
        oh = F.one_hot(t, self.C).permute(0, 3, 1, 2).float()
        v = valid.unsqueeze(1).float()
        inter = (probs * oh * v).sum((2, 3))
        union = ((probs + oh) * v).sum((2, 3))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BootstrappedCE(nn.Module):
    """Noise-robust cross-entropy with bootstrapping."""

    def __init__(self, beta=0.3, ignore_index=255):
        super().__init__()
        self.beta = beta
        self.ig = ignore_index

    def forward(self, logits, targets):
        valid = (targets != self.ig)
        ce_gt = F.cross_entropy(logits, targets, ignore_index=self.ig, reduction='none')
        with torch.no_grad():
            pseudo = logits.detach().argmax(dim=1)
        ce_model = F.cross_entropy(logits, pseudo, ignore_index=self.ig, reduction='none')
        loss = (1.0 - self.beta) * ce_gt + self.beta * ce_model
        return loss[valid].mean()


# ── OHEM (Online Hard Example Mining) ─────────────────────────

class OHEMCrossEntropy(nn.Module):
    """Only backprop on top-K% hardest pixels."""

    def __init__(self, ignore_index=255, top_k_ratio=0.3, min_kept=1000):
        super().__init__()
        self.ig = ignore_index
        self.top_k_ratio = top_k_ratio
        self.min_kept = min_kept

    def forward(self, logits, targets):
        loss_map = F.cross_entropy(logits, targets, ignore_index=self.ig, reduction='none')
        valid = (targets != self.ig)
        loss_valid = loss_map[valid]
        k = max(self.min_kept, int(self.top_k_ratio * loss_valid.numel()))
        k = min(k, loss_valid.numel())
        topk_loss, _ = loss_valid.topk(k)
        return topk_loss.mean()


# ── Lovász-Softmax ────────────────────────────────────────────

def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovász extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_softmax_flat(probs, labels, classes='present'):
    """
    Multi-class Lovász-Softmax loss.
    probs: (N, C) float softmax probabilities
    labels: (N,) long class indices
    """
    if probs.numel() == 0:
        return probs * 0.0
    C = probs.shape[1]
    losses = []
    for c in range(C):
        fg = (labels == c).float()  # (N,)
        if classes == 'present' and fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()  # (N,)
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = _lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if not losses:
        return torch.tensor(0.0, device=probs.device, requires_grad=True)
    return torch.stack(losses).mean()


class LovaszSoftmax(nn.Module):
    """Lovász-Softmax loss — directly optimizes IoU. Per-image to save VRAM."""

    def __init__(self, ignore_index=255):
        super().__init__()
        self.ig = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        B, C, H, W = probs.shape
        losses = []
        for b in range(B):
            valid = (targets[b] != self.ig)
            if valid.sum() == 0:
                continue
            p = probs[b].permute(1, 2, 0)[valid]  # (N_valid, C)
            t = targets[b][valid]                   # (N_valid,)
            losses.append(_lovasz_softmax_flat(p, t))
        if not losses:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return torch.stack(losses).mean()


# ── Combined Loss ─────────────────────────────────────────────

class SegLoss(nn.Module):
    """Combined loss: BootstrappedCE + Lovász/Dice + OHEM + deep supervision."""

    def __init__(self, cfg, class_weights=None):
        super().__init__()
        self.ig = cfg.dataset.ignore_index
        self.ce_w = cfg.train.ce_weight
        self.dice_w = cfg.train.dice_weight
        self.deep_sup = cfg.model.deep_supervision
        nc = cfg.dataset.num_classes

        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=cfg.train.label_smoothing,
            ignore_index=self.ig,
        )
        self.dice = DiceLoss(nc, self.ig)
        self.bootstrap = BootstrappedCE(cfg.train.noise_beta, self.ig) \
            if cfg.train.use_noise_robust_loss else None

        # Lovász-Softmax (directly optimize IoU)
        self.use_lovasz = cfg.train.get('use_lovasz', True)
        self.lovasz_w = cfg.train.get('lovasz_weight', 0.5)
        self.lovasz = LovaszSoftmax(self.ig) if self.use_lovasz else None

        # OHEM for auxiliary heads
        self.use_ohem = cfg.train.get('use_ohem', True)
        self.ohem = OHEMCrossEntropy(self.ig, top_k_ratio=0.3) if self.use_ohem else None

    def forward(self, main, aux1, aux2, targets):
        # Main loss
        if self.bootstrap is not None:
            loss = self.ce_w * self.bootstrap(main, targets)
        else:
            loss = self.ce_w * self.ce(main, targets)

        # Dice or Lovász
        loss = loss + self.dice_w * self.dice(main, targets)
        if self.lovasz is not None:
            loss = loss + self.lovasz_w * self.lovasz(main, targets)

        # Deep supervision with OHEM
        if self.deep_sup:
            aux_loss_fn = self.ohem if self.ohem is not None else self.ce
            if aux1 is not None:
                loss = loss + 0.3 * aux_loss_fn(aux1, targets)
            if aux2 is not None:
                loss = loss + 0.3 * aux_loss_fn(aux2, targets)
        return loss
