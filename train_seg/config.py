# ============================================================
# WSSS Stage 2: Segmentation Training — Configuration
# ============================================================

from omegaconf import OmegaConf

SEG_CONFIG = """
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

train:
  epochs: 30
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
  noise_beta: 0.1

work_dir:
  ckpt_dir: checkpoints/seg
  log_dir:  train_log/seg
"""

CLASS_NAMES = ['TUM', 'STR', 'LYM', 'NEC']


def load_config():
    """Load the default segmentation config."""
    return OmegaConf.create(SEG_CONFIG)
