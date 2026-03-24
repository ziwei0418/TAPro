# ============================================================
# Loss Functions for Segmentation Training
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

    def __init__(self, beta=0.1, ignore_index=255):
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


class SegLoss(nn.Module):
    """Combined CE + Dice + optional bootstrapping + deep supervision."""

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

    def forward(self, main, aux1, aux2, targets):
        if self.bootstrap is not None:
            loss = self.ce_w * self.bootstrap(main, targets)
        else:
            loss = self.ce_w * self.ce(main, targets)
        loss = loss + self.dice_w * self.dice(main, targets)
        if self.deep_sup and aux1 is not None:
            loss = loss + 0.3 * self.ce(aux1, targets)
        if self.deep_sup and aux2 is not None:
            loss = loss + 0.3 * self.ce(aux2, targets)
        return loss
