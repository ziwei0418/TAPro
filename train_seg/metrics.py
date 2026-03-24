# ============================================================
# Segmentation Metrics (IoU, Dice, fwIoU)
# ============================================================

import numpy as np


class SegMetrics:
    """Confusion-matrix based segmentation metrics."""

    def __init__(self, num_classes, ignore_index=255):
        self.C = num_classes
        self.ig = ignore_index
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.C, self.C), dtype=np.int64)

    def update(self, pred, gt):
        valid = gt != self.ig
        p, g = pred[valid], gt[valid]
        self.cm += np.bincount(g * self.C + p, minlength=self.C**2).reshape(self.C, self.C)

    def compute(self):
        h = self.cm.astype(np.float64)
        diag = np.diag(h)
        iou = diag / (h.sum(1) + h.sum(0) - diag + 1e-8)
        dice = 2 * diag / (h.sum(1) + h.sum(0) + 1e-8)
        freq = h.sum(1) / (h.sum() + 1e-8)
        return {
            'iou': iou,
            'miou': iou.mean(),
            'dice': dice.mean(),
            'fwiou': (freq * iou).sum(),
        }
