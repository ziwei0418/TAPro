# AB-TAGD for Weakly Supervised Histopathology Segmentation

This repository contains the WSSS code used for the BCSS paper experiments. The main method is **AB-TAGD**: Activation-Balanced Attention-Boundary Tissue-Adaptive Graph Diffusion for pseudo-mask generation from image-level labels.

The paper contribution is the WSSS pseudo-mask generation pipeline, not the optional downstream segmentation trainer.

## Method Summary

Given an H&E patch with only image-level tissue labels, the pipeline:

1. Extracts frozen dense features using a UNI pathology encoder and a Simple Feature Pyramid.
2. Builds subclass-level visual/text/hybrid tissue prototypes.
3. Generates multi-scale CAMs from projected prototypes and feature maps.
4. Merges subclass CAMs into parent tissue CAMs.
5. Fuses CAM scales 2, 3, and 4.
6. Applies class-wise CAM activation balance.
7. Refines CAMs using one-step class-adaptive TAGD with an attention-boundary penalty.
8. Converts refined CAMs into pseudo masks by foreground argmax.

Core implementation:

- `src/models/network.py`: UNI encoder, FPN, visual/text/hybrid prototypes, multi-scale CAM generation.
- `src/utils/cam_refine.py`: activation balance and AB-TAGD refinement.
- `src/engine/wsss.py`: pseudo-mask generation.
- `src/utils/evaluate.py`: validation/test evaluation with the same CAM pipeline.
- `configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml`: full method config.

## Repository Layout

```text
WSSS_2026_RESEARCH/
├── configs/
│   ├── default.yaml
│   ├── luad.yaml
│   └── experiments/
│       └── bcss_fixed_0p6_feature_region_abdiv_beta2.yaml
├── data/
│   ├── BCSS-WSSS/
│   ├── LUAD-HistoSeg/
│   └── prototypes/
├── paper/
│   ├── method_abtagd_miccai.tex
│   └── method_abtagd_miccai.md
├── scripts/
│   ├── train.py
│   ├── inference.py
│   ├── visualize.py
│   └── run_wsss_paper_ablation.py
├── src/
│   ├── datasets/
│   ├── engine/
│   ├── losses/
│   ├── models/
│   └── utils/
├── train_log/
│   └── wsss_paper_ablation_20260508_065932/
└── train_seg/
```

## Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The main WSSS code expects:

- BCSS data under `data/BCSS-WSSS/`.
- Visual prototype bank under `data/prototypes/bcss_label_fea_pro_3333.pkl`.
- A valid Hugging Face token in the config for loading the UNI encoder if required by the local environment.

## Run Full AB-TAGD on BCSS

```bash
venv/bin/python scripts/train.py \
  --config configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml \
  --gpu 0 \
  --seed 42 \
  --epoch 8
```

The script trains, evaluates on validation/test, reloads the best checkpoint, and generates training pseudo masks.

Expected outputs are created under the paths specified in the config:

```text
train_log/bcss_fixed_0p6_feature_region_abdiv_beta2/
├── checkpoints/<timestamp>_seed42/best_cam.pth
└── predictions/*.png
```

## Run Paper Ablations

```bash
venv/bin/python scripts/run_wsss_paper_ablation.py \
  --gpu 0 \
  --seeds 42 3407 2026 \
  --epoch 8 \
  --tables 1 2 3
```

The ablation runner is continue-on-error: unsupported rows are logged as skipped instead of stopping the full run.

Current paper ablation results are stored in:

```text
train_log/wsss_paper_ablation_20260508_065932/
├── configs/
├── logs/
├── summary_raw.tsv
├── summary_mean_std.txt
└── work/*/checkpoints/*/best_cam.pth
```

Pseudo-mask prediction folders inside the ablation work directory were removed to save disk space; they can be regenerated from the saved checkpoints if needed.

## Main Ablation Tables

The current ablation runner creates three groups:

- Table 1: visual baseline, text prototype, hybrid prototype, and full AB-TAGD.
- Table 2: prototype/source sensitivity for the full pipeline.
- Table 3: AB-TAGD hyperparameter and design sensitivity.

The primary paper number should be taken from `t1_full_abtagd` in `summary_mean_std.txt`. Table 3 is a robustness/sensitivity table; small differences across beta or activation-balance variants are within run-to-run variance and should not be framed as separate improvements.

## Optional LUAD Config

`configs/luad.yaml` is kept for LUAD-style experiments, but the cleaned paper state is centered on BCSS WSSS. If LUAD is used, verify output paths first because old root `checkpoints/` and `predictions_luad/` folders were removed during cleanup.

## Optional Downstream Segmentation

`train_seg/` contains downstream pseudo-label segmentation experiments. This stage is not the main paper contribution and is not required to reproduce the WSSS ablation tables.

## Notes

- The main WSSS method uses no CRF, no TTA, and no SAM.
- The full method uses frozen UNI dense features and CONCH/text-guided hybrid prototypes.
- AB-TAGD is implemented as a one-step graph refinement, not iterative graph smoothing.
