# ============================================================
# train_seg package — Segmentation training for WSSS Stage 2
# Supports: BCSS-WSSS and LUAD-HistoSeg datasets
# ============================================================

from .config import load_config, CLASS_NAMES
from .lora import LoRALinear, inject_lora_segformer, inject_lora_vit
from .model import SegFormerSeg, PhikonSeg, CTransPathSeg, build_seg_model
from .dataset import (
    BCSSTrainSeg, BCSSValSeg,
    LUADTrainSeg, LUADValSeg,
    get_train_aug, get_val_aug, get_datasets,
)
from .losses import DiceLoss, BootstrappedCE, SegLoss
from .metrics import SegMetrics
from .crf import apply_crf
from .scheduler import build_scheduler
from .utils import compute_class_weights, validate, fix_double_ext, regenerate_pseudo_labels, tta_predict
from .trainer import train_seg, train_seg_with_self_training

__all__ = [
    'load_config', 'CLASS_NAMES',
    'LoRALinear', 'inject_lora_segformer', 'inject_lora_vit',
    'SegFormerSeg', 'PhikonSeg', 'CTransPathSeg', 'build_seg_model',
    'BCSSTrainSeg', 'BCSSValSeg', 'LUADTrainSeg', 'LUADValSeg',
    'get_train_aug', 'get_val_aug', 'get_datasets',
    'DiceLoss', 'BootstrappedCE', 'SegLoss',
    'SegMetrics',
    'apply_crf',
    'build_scheduler',
    'compute_class_weights', 'validate', 'fix_double_ext',
    'regenerate_pseudo_labels', 'tta_predict',
    'train_seg', 'train_seg_with_self_training',
]
