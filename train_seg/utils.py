# ============================================================
# Utility functions: class weights, validation, file fixing
# ============================================================

import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .config import CLASS_NAMES
from .metrics import SegMetrics
from .crf import CRF_AVAILABLE, apply_crf


def fix_double_ext(directory):
    """Rename .png.png → .png files in the given directory."""
    files = glob.glob(os.path.join(directory, '*.png.png'))
    if files:
        print(f"  Fixing {len(files)} double-extension files in {directory}")
        for f in files:
            new_name = f.replace('.png.png', '.png')
            os.rename(f, new_name)


def compute_class_weights(pseudo_dir, num_classes=5, max_samples=1000):
    """Compute median-frequency balanced class weights from pseudo labels."""
    counts = np.zeros(num_classes, dtype=np.float64)
    paths = glob.glob(os.path.join(pseudo_dir, '*.png'))[:max_samples]
    for p in paths:
        m = np.array(Image.open(p))
        if m.ndim == 3:
            m = m[:, :, 0]
        m = np.clip(m, 0, num_classes - 1)
        for c in range(num_classes):
            counts[c] += (m == c).sum()
    freq = counts / (counts.sum() + 1e-8)
    median = np.median(freq[freq > 1e-6])
    weights = np.where(freq > 1e-6, median / (freq + 1e-8), 1.0)
    weights = np.clip(weights, 0.5, 4.0)
    print(f"  Class weights: { {n: f'{w:.2f}' for n, w in zip(CLASS_NAMES, weights)} }")
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def validate(model, loader, device, cfg, use_crf=False):
    """Run validation and compute segmentation metrics."""
    model.eval()
    metrics = SegMetrics(cfg.dataset.num_classes, cfg.dataset.ignore_index)
    mean = np.array(cfg.dataset.mean)
    std = np.array(cfg.dataset.std)

    for imgs, masks in tqdm(loader, ncols=80, desc='Val', leave=False):
        imgs = imgs.to(device).float()
        out, _, _ = model(imgs)
        probs = F.softmax(out, dim=1)

        for i in range(imgs.shape[0]):
            if use_crf and CRF_AVAILABLE:
                img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
                img_rgb = ((img_np * std + mean) * 255).clip(0, 255).astype(np.uint8)
                pred = apply_crf(img_rgb, probs[i].cpu().numpy())
            else:
                pred = out[i].argmax(0).cpu().numpy()
            metrics.update(pred, masks[i].numpy())

    model.train()
    return metrics.compute()
