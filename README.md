# Anonymous WSSS Histopathology Segmentation Code

This repository contains code for weakly supervised semantic segmentation (WSSS) experiments on histopathology images. The pipeline learns class activation maps (CAMs) from image-level tissue labels and generates pseudo masks without dense supervision during CAM training.

The method combines frozen pathology foundation features, hybrid text-visual tissue prototypes, activation-balanced CAM fusion, and Boundary-Aware Graph Diffusion (BAGD) for pseudo-mask refinement.

## Method Overview

Given an H&E patch with image-level tissue labels, the pipeline:

1. Extracts frozen dense visual features from a pathology foundation encoder.
2. Builds visual prototype banks and pathology text prototypes.
3. Blends text and visual prototypes to produce subclass-aware CAMs.
4. Merges subclass CAMs into parent tissue CAMs.
5. Fuses CAMs across scales and applies activation balancing.
6. Refines balanced CAMs with BAGD using feature affinity and attention-boundary cues.
7. Converts refined CAMs into pseudo masks by foreground argmax.

The pseudo-mask generation pipeline does not use dense CRF, test-time augmentation, SAM, or an external segmentation model.

## Repository Layout

```text
configs/                 Experiment configs
scripts/                 Training, evaluation, inference, and visualization scripts
src/                     WSSS model, losses, datasets, and CAM utilities
train_seg/               Optional downstream segmentation model code
paper/                   Local manuscript notes and method drafts
requirements.txt         Python dependencies
install.sh               Environment installation helper
```

Important implementation files:

```text
src/models/network.py        Foundation encoder, feature pyramid, prototypes, CAM heads
src/utils/cam_refine.py      Activation balance and BAGD refinement
src/engine/wsss.py           Pseudo-mask generation pipeline
src/utils/evaluate.py        CAM-mask evaluation utilities
scripts/train.py             WSSS training entry point
scripts/run_wsss_paper_ablation.py  Ablation runner
scripts/eval_bcss_seg_no_tta.py     BCSS downstream segmentation evaluation
scripts/eval_luad_seg_no_tta.py     LUAD downstream segmentation evaluation
scripts/make_bcss_sota_qual_figure.py  Qualitative SOTA comparison figure
```

## Environment

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CONCH is recommended for pathology text prototypes. If it is not installed by the environment setup, install it separately:

```bash
python -m pip install "git+https://github.com/Mahmoodlab/CONCH.git"
```

## Data Layout

Place datasets and prototype files under `data/`:

```text
data/BCSS-WSSS/{training,val,test}
data/LUAD-HistoSeg/{train,val,test}
data/prototypes/bcss_label_fea_pro_3333.pkl
data/prototypes/luad_label_fea_proto_3333.pkl
```

Large datasets, checkpoints, predictions, and qualitative figures are intentionally ignored by git.

## WSSS Training

BCSS full configuration:

```bash
venv/bin/python scripts/train.py \
  --config configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml \
  --gpu 0 \
  --seed 42 \
  --epoch 8
```

Run the paper ablation groups:

```bash
venv/bin/python scripts/run_wsss_paper_ablation.py \
  --gpu 0 \
  --seeds 42 3407 2026 \
  --epoch 8 \
  --tables 1 2 3
```

## Downstream Segmentation Evaluation

The downstream segmentation stage is optional and is used to evaluate pseudo masks with a segmentation model checkpoint.

BCSS no-TTA evaluation:

```bash
venv/bin/python scripts/eval_bcss_seg_no_tta.py \
  --checkpoint checkpoints/seg_bcss/best_ema.pth \
  --out-dir predictions_bcss_seg_test_no_tta
```

LUAD no-TTA evaluation:

```bash
venv/bin/python scripts/eval_luad_seg_no_tta.py \
  --checkpoint checkpoints/seg_luad/best_ema.pth \
  --out-dir predictions_luad_seg_test_no_tta
```

Raw prediction masks are saved as class-index PNG files. They may look black in an image viewer because pixel values are class IDs. Use the provided visualization scripts or colorized output folders for inspection.

## Qualitative SOTA Figure

To build a BCSS qualitative comparison figure, place SOTA mask archives in:

```text
data/sota_masks/bcss/AFA.zip
data/sota_masks/bcss/CAM.zip
data/sota_masks/bcss/GRAD_CAM.zip
data/sota_masks/bcss/PBIP.zip
data/sota_masks/bcss/TPRO.zip
```

Then run:

```bash
venv/bin/python scripts/make_bcss_sota_qual_figure.py \
  --rows 2 \
  --top-k 30 \
  --min-margin 0.0
```

The script selects test images whose ground-truth mask contains exactly three foreground classes and where the proposed method has higher per-image mIoU than every listed SOTA mask. Outputs are written to `qualitative_bcss_sota/`.

## Anonymous Release Notes

This repository is prepared for anonymous paper review. Public documentation should avoid personal names, institutional identifiers, local machine paths, private tokens, and private dataset links. Keep generated outputs and downloaded data outside git.
