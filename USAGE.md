# Usage

Common commands for the core WSSS codebase.

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

## Generate pseudo masks

```bash
venv/bin/python scripts/inference.py \
  --config configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml \
  --checkpoint <path/to/best_cam.pth> \
  --split test \
  --gpu 0
```

## Visualization

```bash
venv/bin/python scripts/visualize.py \
  --config configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml \
  --checkpoint <path/to/best_cam.pth> \
  --output_dir framework_figs \
  --gpu 0
```

## Notes

Do not commit paper drafts, ablation notes, datasets, checkpoints, private tokens, predictions, or generated figures.
