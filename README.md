# Weakly-Supervised Semantic Segmentation for Histopathology

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

WSSS framework for histopathology segmentation — achieves semantic segmentation using **only image-level labels**. Supports two datasets: **BCSS** (breast cancer) and **LUAD-HistoSeg** (lung adenocarcinoma).

The pipeline has 3 stages:

1. **Stage 1 — WSSS Training**: Train CAM generation model (UNI encoder + VQ codebooks + text-visual prototypes)
2. **Stage 2 — Pseudo Label Generation**: Generate pseudo segmentation masks via multi-scale CAM fusion
3. **Stage 3 — Segmentation Training**: Train segmentation model (Phikon/SegFormer/CTransPath) on pseudo labels

## Supported Datasets

| Dataset | Classes | Description |
|---------|---------|-------------|
| **BCSS** | TUM, STR, LYM, NEC | Breast Cancer Semantic Segmentation |
| **LUAD-HistoSeg** | TE, NEC, LYM, TAS | Lung Adenocarcinoma Histopathological Segmentation |

## Project Structure

```
WSSS_2026_RESEARCH/
├── configs/
│   ├── default.yaml              # Stage 1 config (BCSS)
│   └── luad.yaml                 # Stage 1 config (LUAD)
├── src/                           # Stage 1 source code
│   ├── models/                    # UNI encoder, FPN, VQ, ClsNetwork
│   ├── datasets/                  # Dataset loaders
│   ├── losses/                    # InfoNCE contrastive losses
│   └── utils/                     # Evaluation, FG/BG features, etc.
├── scripts/
│   ├── train.py                   # Stage 1: WSSS training
│   ├── inference.py               # Stage 2: CAM → pseudo labels
│   └── visualize.py               # Framework visualization
├── train_seg/                     # Stage 3: Segmentation training
│   ├── config.py                  # Seg training config (BCSS + LUAD)
│   ├── model.py                   # SegFormer / Phikon / CTransPath
│   ├── dataset.py                 # Train/Val datasets
│   ├── trainer.py                 # Training loop + self-training
│   └── train_seg.py               # Entry point
├── data/
│   ├── BCSS-WSSS/                 # BCSS dataset
│   │   ├── training/              # Training images (*.png)
│   │   ├── val/{img,mask}/        # Validation
│   │   └── test/{img,mask}/       # Test
│   ├── LUAD-HistoSeg/             # LUAD dataset
│   │   ├── training/              # Training images (*.png)
│   │   ├── val/{img,mask}/        # Validation
│   │   └── test/{img,mask}/       # Test
│   └── prototypes/                # Pre-extracted visual prototypes
│       ├── bcss_label_fea_pro_3333.pkl
│       └── luad_label_fea_proto_3333.pkl
├── predictions/                   # BCSS pseudo labels (Stage 2 output)
├── predictions_luad/              # LUAD pseudo labels (Stage 2 output)
├── checkpoints/                   # Saved models
│   ├── <timestamp>/best_cam.pth   # BCSS Stage 1
│   ├── luad/<timestamp>/best_cam.pth  # LUAD Stage 1
│   ├── seg_bcss/                  # BCSS Stage 3
│   └── seg_luad/                  # LUAD Stage 3
└── venv/                          # Python virtual environment
```

## Installation

```bash
git clone https://github.com/wterrr/WSSS.git
cd WSSS
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> See [QUICKSTART.md](QUICKSTART.md) for detailed installation including troubleshooting.

---

## Full Pipeline — BCSS (Breast Cancer)

### Stage 1: WSSS Training

```bash
python scripts/train.py --config configs/default.yaml --gpu 0
```

**Key config** (`configs/default.yaml`):
```yaml
model:
  fpn_out_dims: [128, 320, 512]  # 3-scale FPN (P2, P3, P4)
  use_lora: false

train:
  epoch: 8
  samples_per_gpu: 4
  accum_steps: 4
  l2: 0.1    # Scale 2 (28x28) loss weight
  l3: 1.0    # Scale 3 (14x14) loss weight
  l4: 1.0    # Scale 4 (7x7)  loss weight
  use_vq: true
  use_text_proto: true
```

**Output:** `checkpoints/<timestamp>/best_cam.pth`

### Stage 2: Pseudo Label Generation

```bash
# Generate pseudo labels for training set (used as supervision for Stage 3)
python scripts/inference.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/<timestamp>/best_cam.pth \
    --split train --gpu 0

# Generate for test set (evaluation)
python scripts/inference.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/<timestamp>/best_cam.pth \
    --split test --gpu 0
```

**Output:** `predictions/*.png`

### Stage 3: Segmentation Training

```bash
python train_seg/train_seg.py --dataset bcss
```

**Key config** (embedded in `train_seg/config.py`):
```yaml
model:
  type: phikon              # or: segformer, ctranspath
  backbone: owkin/phikon
  use_lora: true
  deep_supervision: true

dataset:
  train_root: data/BCSS-WSSS/training
  pseudo_dir: predictions   # pseudo labels from Stage 2
  class_names: [TUM, STR, LYM, NEC]

train:
  epochs: 80
  batch_size: 16
  crop_size: 224
  lr: 0.00006
  ce_weight: 1.0
  dice_weight: 0.5
  use_noise_robust_loss: true
```

**Output:** `checkpoints/seg_bcss/best_seg.pth`

---

## Full Pipeline — LUAD-HistoSeg (Lung Adenocarcinoma)

### Stage 1: WSSS Training

```bash
python scripts/train.py --config configs/luad.yaml --gpu 0
```

**Key config** (`configs/luad.yaml`):
```yaml
model:
  fpn_out_dims: [128, 320, 512]
  label_feature_path: data/prototypes/luad_label_fea_proto_3333.pkl

dataset:
  name: luad
  train_root: data/LUAD-HistoSeg/training
  cls_num_classes: 4  # TE, NEC, LYM, TAS

train:
  epoch: 15
  use_vq: true
  use_text_proto: true
```

**Output:** `checkpoints/luad/<timestamp>/best_cam.pth`

### Stage 2: Pseudo Label Generation

```bash
# Generate pseudo labels for training set
python scripts/inference.py \
    --config configs/luad.yaml \
    --checkpoint checkpoints/luad/<timestamp>/best_cam.pth \
    --split train --gpu 0

# Generate for test set
python scripts/inference.py \
    --config configs/luad.yaml \
    --checkpoint checkpoints/luad/<timestamp>/best_cam.pth \
    --split test --gpu 0
```

**Output:** `predictions_luad/*.png`

### Stage 3: Segmentation Training

```bash
python train_seg/train_seg.py --dataset luad
```

**Key config** (embedded in `train_seg/config.py`):
```yaml
dataset:
  train_root: data/LUAD-HistoSeg/training
  pseudo_dir: predictions_luad
  class_names: [TE, NEC, LYM, TAS]

train:
  epochs: 80
  dice_weight: 1.0
  patience: 7
```

**Output:** `checkpoints/seg_luad/best_seg.pth`

---

## Quick Reference

| Action | BCSS | LUAD |
|--------|------|------|
| **Stage 1** | `python scripts/train.py --config configs/default.yaml` | `python scripts/train.py --config configs/luad.yaml` |
| **Stage 2** | `python scripts/inference.py --config configs/default.yaml --checkpoint <path> --split train` | `python scripts/inference.py --config configs/luad.yaml --checkpoint <path> --split train` |
| **Stage 3** | `python train_seg/train_seg.py --dataset bcss` | `python train_seg/train_seg.py --dataset luad` |

## Architecture

```
Stage 1: WSSS CAM Model (3-scale)
===================================
Input Image (224×224)
    ↓
UNI ViT-L/16 (frozen + LoRA)
  → Block 11, Block 17, Block 23
    ↓
Simple Feature Pyramid (SFP)
  → P2(28×28, 128ch) → P3(14×14, 320ch) → P4(7×7, 512ch)
    ↓
VQ Codebook Quantization (P2-P4)
    ↓
Hybrid Text-Visual Prototypes
    ↓
Multi-Scale CAMs → Fuse: 0.3×CAM2 + 0.3×CAM3 + 0.4×CAM4
    ↓
Pseudo Segmentation Mask

Stage 3: Segmentation Model
============================
Input Image (224×224)
    ↓
Phikon ViT-B / SegFormer / CTransPath (frozen + LoRA)
    ↓
FPN Decoder (multi-scale feature fusion)
    ↓
Segmentation Output (4 classes)
```

## Backbone Options (Stage 3)

Edit `train_seg/config.py` or pass `--dataset` flag:

| Backbone | Type | Description | VRAM |
|----------|------|-------------|------|
| **Phikon** (default) | `phikon` | Pathology-pretrained ViT-B (Owkin) | ~8GB |
| **SegFormer** | `segformer` | MIT-B3 with built-in decoder | ~10GB |
| **CTransPath** | `ctranspath` | Pathology-pretrained Swin-T | ~6GB |

## GPU Requirements

| Stage | Model | Min VRAM | Recommended |
|-------|-------|----------|-------------|
| Stage 1 | UNI ViT-L + VQ | 10GB | 12GB+ |
| Stage 3 (Phikon) | ViT-B + decoder | 8GB | 12GB |
| Stage 3 (SegFormer) | MIT-B3 | 10GB | 12GB |
| Stage 3 (CTransPath) | Swin-T | 6GB | 8GB+ |

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- **UNI** — Universal pathology foundation model (MahmoodLab)
- **Phikon** — Pathology ViT-B (Owkin)
- **CONCH** — Histopathology contrastive learning
- **BCSS** — Breast Cancer Semantic Segmentation dataset
- **LUAD-HistoSeg** — Lung Adenocarcinoma Histopathological Segmentation dataset
