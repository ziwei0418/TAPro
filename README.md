# ProBAG: Prototype-Guided Boundary-Aware Graph Diffusion for Weakly Supervised Histopathology Segmentation

This repository contains the core code for weakly supervised semantic segmentation (WSSS) on histopathology image patches. It supports CAM training from image-level labels and pseudo-mask generation for BCSS-WSSS and LUAD-HistoSeg style datasets.

## Project Structure

```text
configs/          Experiment configuration files
scripts/          Training, inference, and visualization entry points
src/              WSSS datasets, models, losses, and CAM refinement utilities
train_seg/        Optional downstream segmentation code
requirements.txt  Python dependencies
install.sh        Optional installation helper
```

Core entry points:

```text
scripts/train.py          Train the WSSS CAM model
scripts/inference.py      Generate pseudo masks from a CAM checkpoint
scripts/visualize.py      Export CAM/framework visualizations
```

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional CONCH installation for pathology text prototypes:

```bash
python -m pip install "git+https://github.com/Mahmoodlab/CONCH.git"
```

Quick import check:

```bash
venv/bin/python -c "from src.models.network import ClsNetwork; print('ok')"
```

## Data Layout

Place datasets and prototype files under `data/`:

```text
data/BCSS-WSSS/{training,val,test}
data/LUAD-HistoSeg/{train,val,test}
data/prototypes/bcss_label_fea_pro_3333.pkl
data/prototypes/luad_label_fea_proto_3333.pkl
```

Datasets, checkpoints, prediction folders, generated figures, and local paper notes are ignored by git.

## Train WSSS CAM Model

BCSS example:

```bash
venv/bin/python scripts/train.py \
  --config configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml \
  --gpu 0 \
  --seed 42 \
  --epoch 8
```

## Generate Pseudo Masks

```bash
venv/bin/python scripts/inference.py \
  --config configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml \
  --checkpoint <path/to/best_cam.pth> \
  --split test \
  --gpu 0
```

The output directory is controlled by the config field `work_dir.pred_dir`.

## Visualization

```bash
venv/bin/python scripts/visualize.py \
  --config configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml \
  --checkpoint <path/to/best_cam.pth> \
  --output_dir framework_figs \
  --gpu 0
```

## Acknowledgment

This repository is built upon and adapted from the excellent [PBIP](https://github.com/QingchenTang/PBIP) implementation.

We sincerely thank the authors for making their code publicly available and for their valuable contribution to weakly supervised semantic segmentation research.
