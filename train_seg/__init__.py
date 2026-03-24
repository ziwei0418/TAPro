# ============================================================
# train_seg package — Segmentation training for WSSS Stage 2
# ============================================================

from .config import load_config, CLASS_NAMES
from .lora import LoRALinear, inject_lora_segformer, inject_lora_vit
from .model import SegFormerSeg, PhikonSeg, CTransPathSeg, build_seg_model
from .dataset import BCSSTrainSeg, BCSSValSeg, get_train_aug, get_val_aug
from .losses import DiceLoss, BootstrappedCE, SegLoss
from .metrics import SegMetrics
from .crf import apply_crf
from .scheduler import build_scheduler
from .utils import compute_class_weights, validate, fix_double_ext
from .trainer import train_seg

__all__ = [
    'load_config', 'CLASS_NAMES',
    'LoRALinear', 'inject_lora_segformer', 'inject_lora_vit',
    'SegFormerSeg', 'PhikonSeg', 'CTransPathSeg', 'build_seg_model',
    'BCSSTrainSeg', 'BCSSValSeg', 'get_train_aug', 'get_val_aug',
    'DiceLoss', 'BootstrappedCE', 'SegLoss',
    'SegMetrics',
    'apply_crf',
    'build_scheduler',
    'compute_class_weights', 'validate', 'fix_double_ext',
    'train_seg',
]
