# ============================================================
# WSSS Stage 2: Segmentation Training — Configuration
# Supports: BCSS and LUAD-HistoSeg datasets
# ============================================================

import argparse
from omegaconf import OmegaConf

BCSS_SEG_CONFIG = """
model:
  type: phikon
  backbone: owkin/phikon
  num_classes: 4
  use_lora: true
  lora_rank: 8
  lora_alpha: 8.0
  deep_supervision: true

dataset:
  name: bcss
  train_root: data/BCSS-WSSS/training
  pseudo_dir: predictions
  val_root: data/BCSS-WSSS
  test_root: data/BCSS-WSSS
  num_classes: 4
  ignore_index: 255
  mean: [0.7024, 0.5462, 0.6965]
  std:  [0.2389, 0.2821, 0.2163]
  class_names: [TUM, STR, LYM, NEC]

train:
  epochs: 80
  batch_size: 16
  accum_steps: 1
  crop_size: 224
  num_workers: 8
  lr: 0.00006
  backbone_lr_scale: 0.1
  weight_decay: 0.05
  warmup_epochs: 3
  label_smoothing: 0.0
  ce_weight: 1.0
  dice_weight: 0.5
  use_crf_val: false
  use_noise_robust_loss: true
  noise_beta: 0.3
  use_lovasz: false
  lovasz_weight: 0.5
  use_ohem: false
  use_tta_val: true
  tta_scales: [0.75, 1.0, 1.25]
  self_training_rounds: 1
  patience: 10

work_dir:
  ckpt_dir: checkpoints/seg_bcss
  log_dir:  train_log/seg_bcss
"""

LUAD_SEG_CONFIG = """
model:
  type: phikon
  backbone: owkin/phikon
  num_classes: 4
  use_lora: true
  lora_rank: 8
  lora_alpha: 8.0
  deep_supervision: true

dataset:
  name: luad
  train_root: data/LUAD-HistoSeg/training
  pseudo_dir: predictions_luad
  val_root: data/LUAD-HistoSeg
  test_root: data/LUAD-HistoSeg
  num_classes: 4
  ignore_index: 255
  mean: [0.7024, 0.5462, 0.6965]
  std:  [0.2389, 0.2821, 0.2163]
  class_names: [TE, NEC, LYM, TAS]

train:
  epochs: 80
  batch_size: 16
  accum_steps: 1
  crop_size: 224
  num_workers: 8
  lr: 0.00006
  backbone_lr_scale: 0.1
  weight_decay: 0.05
  warmup_epochs: 3
  label_smoothing: 0.0
  ce_weight: 1.0
  dice_weight: 1.0
  use_crf_val: false
  use_noise_robust_loss: true
  noise_beta: 0.3
  use_lovasz: false
  lovasz_weight: 0.5
  use_ohem: false
  use_tta_val: true
  tta_scales: [0.75, 1.0, 1.25]
  self_training_rounds: 1
  patience: 7

work_dir:
  ckpt_dir: checkpoints/seg_luad
  log_dir:  train_log/seg_luad
"""

# For backward compat
CLASS_NAMES = ['TUM', 'STR', 'LYM', 'NEC']

CONFIGS = {
    'bcss': BCSS_SEG_CONFIG,
    'luad': LUAD_SEG_CONFIG,
}


def load_config(dataset_name=None):
    """Load segmentation config. Accepts CLI --dataset arg or explicit name."""
    if dataset_name is None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--dataset', type=str, default='bcss', choices=list(CONFIGS.keys()))
        args, _ = parser.parse_known_args()
        dataset_name = args.dataset

    cfg_str = CONFIGS.get(dataset_name, BCSS_SEG_CONFIG)
    return OmegaConf.create(cfg_str)
