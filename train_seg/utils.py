# ============================================================
# Utility functions: class weights, validation, TTA, self-training
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


def compute_class_weights(pseudo_dir, num_classes=5, max_samples=1000,
                          ignore_index=255, class_names=None):
    """Compute median-frequency balanced class weights from valid pseudo labels."""
    counts = np.zeros(num_classes, dtype=np.float64)
    paths = glob.glob(os.path.join(pseudo_dir, '*.png'))[:max_samples]
    for p in paths:
        m = np.array(Image.open(p))
        if m.ndim == 3:
            m = m[:, :, 0]
        valid = (m != ignore_index) & (m >= 0) & (m < num_classes)
        for c in range(num_classes):
            counts[c] += ((m == c) & valid).sum()
    if counts.sum() <= 0:
        print("  Class weights: no valid pseudo labels found, using uniform weights")
        return torch.ones(num_classes, dtype=torch.float32)
    freq = counts / (counts.sum() + 1e-8)
    median = np.median(freq[freq > 1e-6])
    weights = np.where(freq > 1e-6, median / (freq + 1e-8), 1.0)
    weights = np.clip(weights, 0.5, 4.0)
    names = list(class_names) if class_names is not None else CLASS_NAMES[:num_classes]
    print(f"  Class weights: { {n: f'{w:.2f}' for n, w in zip(names, weights)} }")
    return torch.tensor(weights, dtype=torch.float32)


# ── Multi-scale TTA (Test-Time Augmentation) ──────────────────

@torch.no_grad()
def tta_predict(model, img, scales=(0.75, 1.0, 1.25), flip=True):
    """
    Multi-scale + flip TTA prediction.
    img: (1, C, H, W) tensor on device
    Returns: (1, num_classes, H, W) averaged probabilities
    """
    B, C, H, W = img.shape
    total_probs = None

    for scale in scales:
        sH, sW = int(H * scale), int(W * scale)
        scaled = F.interpolate(img, (sH, sW), mode='bilinear', align_corners=False)

        out, _, _ = model(scaled)
        probs = F.softmax(out, dim=1)
        probs = F.interpolate(probs, (H, W), mode='bilinear', align_corners=False)

        if total_probs is None:
            total_probs = probs
        else:
            total_probs += probs

        if flip:
            # Horizontal flip
            out_flip, _, _ = model(scaled.flip(3))
            probs_flip = F.softmax(out_flip, dim=1).flip(3)
            probs_flip = F.interpolate(probs_flip, (H, W), mode='bilinear', align_corners=False)
            total_probs += probs_flip

    n = len(scales) * (2 if flip else 1)
    return total_probs / n


# ── Validate ──────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device, cfg, use_crf=False, use_tta=False):
    """Run validation with optional TTA and CRF."""
    model.eval()
    metrics = SegMetrics(cfg.dataset.num_classes, cfg.dataset.ignore_index)
    mean = np.array(cfg.dataset.mean)
    std = np.array(cfg.dataset.std)
    tta_scales = cfg.train.get('tta_scales', [0.75, 1.0, 1.25])

    for imgs, masks in tqdm(loader, ncols=80, desc='Val', leave=False):
        imgs = imgs.to(device).float()

        if use_tta:
            # TTA: process one by one
            for i in range(imgs.shape[0]):
                probs = tta_predict(model, imgs[i:i+1], scales=tta_scales)
                pred = probs[0].argmax(0).cpu().numpy()
                metrics.update(pred, masks[i].numpy())
        else:
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


# ── Self-Training: Re-generate pseudo labels ──────────────────

@torch.no_grad()
def regenerate_pseudo_labels(model, train_dataset, device, cfg, output_dir):
    """
    Self-training: use current model to re-generate better pseudo labels.
    Overwrites pseudo labels in output_dir.
    """
    from torch.utils.data import DataLoader
    from .dataset import get_val_aug, BCSSTrainSeg, LUADTrainSeg

    model.eval()
    mean = list(cfg.dataset.mean)
    std = list(cfg.dataset.std)

    # Create a temporary dataset with val augmentations (no random crop)
    ds_name = getattr(cfg.dataset, 'name', 'bcss').lower()
    TrainCls = LUADTrainSeg if ds_name == 'luad' else BCSSTrainSeg
    temp_ds = TrainCls(
        img_root=cfg.dataset.train_root,
        pseudo_dir=cfg.dataset.pseudo_dir,
        transform=get_val_aug(mean, std),
        ignore_index=cfg.dataset.ignore_index,
    )
    temp_loader = DataLoader(temp_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for imgs, _ in tqdm(temp_loader, ncols=80, desc='Re-gen pseudo labels', leave=False):
        imgs = imgs.to(device).float()
        out, _, _ = model(imgs)
        preds = out.argmax(1).cpu().numpy()
        for i in range(len(preds)):
            if count < len(temp_ds.img_paths):
                fname = os.path.basename(temp_ds.img_paths[count])
                mask = Image.fromarray(preds[i].astype(np.uint8))
                mask.save(os.path.join(output_dir, fname))
                count += 1

    model.train()
    print(f"  ✓ Re-generated {count} pseudo labels → {output_dir}")
    return count
