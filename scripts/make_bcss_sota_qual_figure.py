#!/usr/bin/env python3
import argparse
import csv
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


CLASS_NAMES = ["TUM", "STR", "LYM", "NEC"]
PALETTE = np.array(
    [
        [220, 20, 60],
        [34, 139, 34],
        [30, 144, 255],
        [153, 0, 255],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)
METHOD_ZIPS = {
    "CAM": "CAM.zip",
    "Grad-CAM": "GRAD_CAM.zip",
    "AFA": "AFA.zip",
    "TPRO": "TPRO.zip",
    "PBIP": "PBIP.zip",
}


def normalized_key(path: Path) -> str:
    name = path.name
    if name.endswith(".png.png"):
        name = name[:-4]
    return name


def extract_if_needed(zip_path: Path, out_dir: Path) -> None:
    marker = out_dir / ".extracted"
    if marker.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    marker.write_text(zip_path.name + "\n")


def build_png_index(root: Path) -> dict[str, Path]:
    index = {}
    for path in root.rglob("*.png"):
        index.setdefault(normalized_key(path), path)
    return index


def load_image(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return np.array(img)


def rgb_to_label(arr: np.ndarray) -> np.ndarray:
    out = np.full(arr.shape[:2], 255, dtype=np.uint8)
    for label, color in enumerate(PALETTE[:4]):
        out[np.all(arr[:, :, :3] == color, axis=-1)] = label
    out[np.all(arr[:, :, :3] == PALETTE[4], axis=-1)] = 255
    return out


def load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path)
    if size is not None and img.size != size:
        img = img.resize(size, Image.NEAREST)
    arr = np.array(img)

    if arr.ndim == 3:
        mapped = rgb_to_label(arr)
        if np.any(mapped != 255):
            arr = mapped
        else:
            arr = arr[:, :, 0]

    arr = arr.astype(np.int32)
    arr[arr == 4] = 255
    arr[~np.isin(arr, [0, 1, 2, 3, 255])] = 255
    return arr.astype(np.uint8)


def colorize(mask: np.ndarray) -> np.ndarray:
    vis = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    for c in range(4):
        vis[mask == c] = PALETTE[c]
    return vis


def present_gt_classes(gt: np.ndarray) -> list[int]:
    labels = np.unique(gt[gt != 255])
    return [int(x) for x in labels if 0 <= int(x) < 4]


def per_image_miou(pred: np.ndarray, gt: np.ndarray, classes: list[int]) -> float:
    valid = gt != 255
    vals = []
    for c in classes:
        p = (pred == c) & valid
        g = (gt == c) & valid
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        vals.append(float(inter / union) if union > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def write_csv(path: Path, rows: list[dict], methods: list[str]) -> None:
    fields = ["rank", "filename", "classes", "ours_miou", "best_sota", "best_sota_miou", "margin"]
    fields += [f"{m}_miou" for m in methods]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            out = {k: row.get(k, "") for k in fields}
            out["rank"] = rank
            writer.writerow(out)


def draw_grid(rows: list[dict], methods: list[str], out_png: Path, out_pdf: Path, dpi: int) -> None:
    columns = ["Image", "Mask"] + methods + ["Ours"]
    nrows, ncols = len(rows), len(columns)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.35 * ncols, 1.35 * nrows), squeeze=False)

    for col, title in enumerate(columns):
        axes[0, col].set_title(title, fontsize=9)

    for r, row in enumerate(rows):
        panels = [row["image"], colorize(row["gt"])]
        panels.extend(colorize(row["preds"][m]) for m in methods)
        panels.append(colorize(row["ours"]))
        for c, panel in enumerate(panels):
            axes[r, c].imshow(panel)
            axes[r, c].axis("off")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.04, hspace=0.08)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select exact-3-class BCSS examples where Ours beats SOTA masks and draw a paper-style figure."
    )
    parser.add_argument("--data-root", default="data/BCSS-WSSS")
    parser.add_argument("--sota-dir", default="data/sota_masks/bcss")
    parser.add_argument("--ours-dir", default="predictions_bcss_seg_test_no_tta/pred_masks")
    parser.add_argument("--work-dir", default="qualitative_bcss_sota")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    image_dir = data_root / "test" / "img"
    gt_dir = data_root / "test" / "mask"
    sota_dir = Path(args.sota_dir)
    ours_dir = Path(args.ours_dir)
    work_dir = Path(args.work_dir)
    extract_dir = work_dir / "extracted"
    row_dir = work_dir / "rows"
    work_dir.mkdir(parents=True, exist_ok=True)
    row_dir.mkdir(parents=True, exist_ok=True)

    methods = list(METHOD_ZIPS)
    method_indices = {}
    for method, zip_name in METHOD_ZIPS.items():
        zip_path = sota_dir / zip_name
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing {method} zip: {zip_path}")
        method_root = extract_dir / method
        extract_if_needed(zip_path, method_root)
        method_indices[method] = build_png_index(method_root)

    ours_index = build_png_index(ours_dir)
    gt_paths = sorted(gt_dir.glob("*.png"))
    candidates = []

    for gt_path in tqdm(gt_paths, ncols=80, desc="Scanning BCSS candidates"):
        key = gt_path.name
        image_path = image_dir / key
        if not image_path.exists() or key not in ours_index:
            continue
        if any(key not in method_indices[m] for m in methods):
            continue

        gt = load_mask(gt_path)
        classes = present_gt_classes(gt)
        if len(classes) != 3:
            continue

        size = (gt.shape[1], gt.shape[0])
        ours = load_mask(ours_index[key], size=size)
        preds = {m: load_mask(method_indices[m][key], size=size) for m in methods}
        ours_miou = per_image_miou(ours, gt, classes)
        sota_scores = {m: per_image_miou(preds[m], gt, classes) for m in methods}
        best_sota = max(sota_scores, key=sota_scores.get)
        margin = ours_miou - sota_scores[best_sota]
        if margin < args.min_margin or margin <= 0:
            continue

        candidates.append(
            {
                "filename": key,
                "classes": ",".join(CLASS_NAMES[c] for c in classes),
                "ours_miou": f"{ours_miou * 100:.4f}",
                "best_sota": best_sota,
                "best_sota_miou": f"{sota_scores[best_sota] * 100:.4f}",
                "margin": f"{margin * 100:.4f}",
                "image_path": image_path,
                "gt": gt,
                "ours": ours,
                "preds": preds,
                "image": load_image(image_path, size=size),
                **{f"{m}_miou": f"{sota_scores[m] * 100:.4f}" for m in methods},
            }
        )

    candidates.sort(key=lambda r: (float(r["margin"]), float(r["ours_miou"])), reverse=True)
    top = candidates[: args.top_k]
    write_csv(work_dir / "candidates_exact3_ours_best_all.csv", candidates, methods)
    write_csv(work_dir / "candidates_exact3_ours_best_top.csv", top, methods)

    for rank, row in enumerate(top, 1):
        draw_grid([row], methods, row_dir / f"{rank:02d}_{Path(row['filename']).stem}.png", row_dir / f"{rank:02d}_{Path(row['filename']).stem}.pdf", args.dpi)

    figure_rows = top[: args.rows]
    if figure_rows:
        draw_grid(
            figure_rows,
            methods,
            work_dir / "bcss_qualitative_exact3_ours_best.png",
            work_dir / "bcss_qualitative_exact3_ours_best.pdf",
            args.dpi,
        )

    print(f"Scanned masks: {len(gt_paths)}")
    print(f"Candidates where GT has exactly 3 classes and Ours is best: {len(candidates)}")
    print(f"Saved all-candidate CSV: {work_dir / 'candidates_exact3_ours_best_all.csv'}")
    print(f"Saved top CSV: {work_dir / 'candidates_exact3_ours_best_top.csv'}")
    if figure_rows:
        print(f"Saved figure: {work_dir / 'bcss_qualitative_exact3_ours_best.png'}")
    for rank, row in enumerate(top, 1):
        print(
            f"{rank:02d}. {row['filename']} | classes={row['classes']} | "
            f"Ours={row['ours_miou']} | best_sota={row['best_sota']}:{row['best_sota_miou']} | "
            f"margin={row['margin']}"
        )


if __name__ == "__main__":
    main()
