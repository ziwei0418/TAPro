# Usage

This document summarizes common commands for the anonymous WSSS histopathology segmentation codebase.

## Install

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train WSSS CAM model

```bash
venv/bin/python scripts/train.py \
  --config configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml \
  --gpu 0 \
  --seed 42 \
  --epoch 8
```

## Run ablation tables

```bash
venv/bin/python scripts/run_wsss_paper_ablation.py \
  --gpu 0 \
  --seeds 42 3407 2026 \
  --epoch 8 \
  --tables 1 2 3
```

## Inference / visualization

```bash
venv/bin/python scripts/inference.py \
  --config configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml \
  --checkpoint <path/to/best_cam.pth> \
  --split test \
  --gpu 0

venv/bin/python scripts/visualize.py \
  --config configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml \
  --checkpoint <path/to/best_cam.pth> \
  --output_dir framework_figs \
  --gpu 0
```

## Downstream segmentation evaluation

```bash
venv/bin/python scripts/eval_bcss_seg_no_tta.py \
  --checkpoint checkpoints/seg_bcss/best_ema.pth \
  --out-dir predictions_bcss_seg_test_no_tta

venv/bin/python scripts/eval_luad_seg_no_tta.py \
  --checkpoint checkpoints/seg_luad/best_ema.pth \
  --out-dir predictions_luad_seg_test_no_tta
```

## Qualitative comparison figure

```bash
venv/bin/python scripts/make_bcss_sota_qual_figure.py \
  --rows 2 \
  --top-k 30 \
  --min-margin 0.0
```

The script expects SOTA BCSS mask archives under `data/sota_masks/bcss/` and writes outputs to `qualitative_bcss_sota/`.

## Notes

- Do not commit datasets, checkpoints, private tokens, predictions, or generated figures.
- Keep public documentation free of personal names, local machine paths, and private links.
- Raw prediction masks are class-index PNG files; colorize them before visual inspection.
