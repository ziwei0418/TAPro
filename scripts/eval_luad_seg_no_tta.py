#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_seg.config import load_config
from train_seg.dataset import LUADValSeg, get_val_aug
from train_seg.metrics import SegMetrics
from train_seg.model import build_seg_model


CLASS_NAMES = ["TE", "NEC", "LYM", "TAS"]
PALETTE = np.array(
    [
        [205, 51, 51],    # TE
        [0, 255, 0],      # NEC
        [65, 105, 225],   # LYM
        [255, 165, 0],    # TAS
        [255, 255, 255],  # ignored/background display
    ],
    dtype=np.uint8,
)


def colorize(mask: np.ndarray) -> np.ndarray:
    vis = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    for c in range(4):
        vis[mask == c] = PALETTE[c]
    return vis


def per_image_iou(pred: np.ndarray, gt: np.ndarray, classes: list[int]) -> tuple[float, list[float]]:
    vals = []
    for c in classes:
        p = pred == c
        g = gt == c
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        vals.append(float(inter / union) if union > 0 else 0.0)
    return float(np.mean(vals)), vals


def make_panel(image_path: str, gt: np.ndarray, pred: np.ndarray, out_path: Path) -> None:
    img = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    gt_vis = colorize(gt)
    pred_vis = colorize(pred)
    panel = np.concatenate([img, gt_vis, pred_vis], axis=1)
    cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/seg_luad/best_ema.pth")
    parser.add_argument("--out-dir", default="predictions_luad_seg_test_no_tta")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "pred_masks"
    top_dir = out_dir / "top10_exact3_panels"
    pred_dir.mkdir(parents=True, exist_ok=True)
    top_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", load_config("luad"))
    cfg.dataset.test_root = "data/LUAD-HistoSeg"
    cfg.dataset.num_classes = 4
    cfg.dataset.ignore_index = 255
    cfg.train.use_tta_val = False

    test_ds = LUADValSeg(
        cfg.dataset.test_root,
        "test",
        transform=get_val_aug(list(cfg.dataset.mean), list(cfg.dataset.std)),
    )
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_seg_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    metrics = SegMetrics(cfg.dataset.num_classes, cfg.dataset.ignore_index)
    rows = []
    sample_idx = 0

    with torch.no_grad():
        for imgs, masks in tqdm(loader, ncols=80, desc="LUAD test no-TTA"):
            imgs = imgs.to(device).float()
            logits, _, _ = model(imgs)
            preds = logits.argmax(1).cpu().numpy().astype(np.uint8)
            masks_np = masks.numpy()

            for b in range(preds.shape[0]):
                pred = preds[b]
                gt = masks_np[b]
                img_path = test_ds.img_paths[sample_idx]
                name = os.path.basename(img_path)

                valid_gt = gt[gt != cfg.dataset.ignore_index]
                present = sorted(int(c) for c in np.unique(valid_gt) if 0 <= c < 4)
                metrics.update(pred, gt)

                Image.fromarray(pred).save(pred_dir / name)

                if len(present) == 3:
                    miou, ious = per_image_iou(pred, gt, present)
                    row = {
                        "filename": name,
                        "image_path": img_path,
                        "miou": miou,
                        "classes": ",".join(CLASS_NAMES[c] for c in present),
                    }
                    for c, v in zip(present, ious):
                        row[f"iou_{CLASS_NAMES[c]}"] = v
                    rows.append(row)

                sample_idx += 1

    overall = metrics.compute()
    rows.sort(key=lambda r: r["miou"], reverse=True)
    top10 = rows[:10]

    for rank, row in enumerate(top10, 1):
        name = row["filename"]
        gt = np.array(Image.open(Path(cfg.dataset.test_root) / "test" / "mask" / name))
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        gt[gt == 4] = cfg.dataset.ignore_index
        pred = np.array(Image.open(pred_dir / name))
        make_panel(row["image_path"], gt, pred, top_dir / f"{rank:02d}_{name}")

    with open(out_dir / "top10_exact3.csv", "w", newline="") as f:
        fieldnames = ["rank", "filename", "miou", "classes"] + [f"iou_{n}" for n in CLASS_NAMES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(top10, 1):
            out = {k: row.get(k, "") for k in fieldnames}
            out["rank"] = rank
            writer.writerow(out)

    with open(out_dir / "metrics.txt", "w") as f:
        f.write("LUAD segmentation test evaluation, no TTA\n")
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"samples: {len(test_ds)}\n")
        f.write(f"exact_3_class_candidates: {len(rows)}\n")
        f.write(f"mIoU: {overall['miou'] * 100:.4f}\n")
        f.write(f"mDice: {overall['dice'] * 100:.4f}\n")
        f.write(f"FwIoU: {overall['fwiou'] * 100:.4f}\n")
        for name, val in zip(CLASS_NAMES, overall["iou"]):
            f.write(f"IoU_{name}: {val * 100:.4f}\n")

    print("\nLUAD segmentation test evaluation, no TTA")
    print(f"checkpoint: {args.checkpoint}")
    print(f"samples: {len(test_ds)}")
    print(f"exact 3-class candidates: {len(rows)}")
    print(
        f"mIoU={overall['miou'] * 100:.2f}% | "
        f"mDice={overall['dice'] * 100:.2f}% | "
        f"FwIoU={overall['fwiou'] * 100:.2f}%"
    )
    print("per-class IoU:", " | ".join(f"{n}:{v * 100:.2f}%" for n, v in zip(CLASS_NAMES, overall["iou"])))
    print(f"saved masks: {pred_dir}")
    print(f"saved top10 panels: {top_dir}")
    print(f"saved top10 csv: {out_dir / 'top10_exact3.csv'}")
    for rank, row in enumerate(top10, 1):
        print(f"{rank:02d}. {row['filename']} | mIoU={row['miou'] * 100:.2f}% | {row['classes']}")


if __name__ == "__main__":
    main()
