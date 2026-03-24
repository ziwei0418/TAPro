# ============================================================
# All imports (consolidated)
# ============================================================
import argparse
import datetime
import os
import os.path
import glob
import re
import random
import logging
import math
import pickle as pkl
from functools import partial
import io

import numpy as np
import cv2
import cv2 as cv
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import torchvision.transforms.functional as TF

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
try:
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from tqdm import tqdm
import ttach as tta
from omegaconf import OmegaConf
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
from icecream import ic
ic.configureOutput(includeContext=True)

try:
    import pydensecrf.densecrf as dcrf
    import pydensecrf.utils as crf_utils
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    print("Warning: pydensecrf not installed. CRF post-processing will be disabled.")

# CONCH text encoder (histopathology-specialized, Nature Medicine 2024)
try:
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
    CONCH_AVAILABLE = True
except ImportError:
    CONCH_AVAILABLE = False
    print("Warning: conch not installed. Text prototypes will fall back to MedCLIP.")


class BCSSTrainingDataset(Dataset):
    CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]
    def __init__(self, img_root="../data/BCSS-WSSS/train", transform=None):
        super(BCSSTrainingDataset, self).__init__()
        self.get_images_and_labels(img_root)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        cls_label = self.cls_labels[index]
        assert os.path.exists(img_path), "img_path: {} does not exists".format(img_path)

        img = cv.imread(img_path, cv.IMREAD_UNCHANGED)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return os.path.basename(img_path), img, cls_label, 0

    def get_images_and_labels(self, img_root=None):
        self.img_paths = []
        self.cls_labels = []

        self.img_paths = glob.glob(os.path.join(img_root, "*.png"))
        for img_path in self.img_paths:
            term_split = re.split(r"\[|\]", img_path)
            cls_label = np.array([int(x) for x in term_split[1]])
            self.cls_labels.append(cls_label)


class BCSSTestDataset(Dataset):
    CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]
    def __init__(self, img_root="../data/BCSS-WSSS/", split="test", transform=None):
        assert split in ["test", "valid", "val"], "split must be one of [test, valid, val]"
        super(BCSSTestDataset, self).__init__()
        self.get_images_and_labels(img_root, split)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        assert os.path.exists(img_path), "img_path: {} does not exists".format(img_path)
        assert os.path.exists(mask_path), "mask_path: {} does not exists".format(mask_path)

        img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        mask = np.array(Image.open(mask_path))
        cls_label = np.array([0, 0, 0, 0])
        x = np.unique(mask) if np.unique(mask)[-1] != 4 else np.unique(mask)[:-1]
        cls_label[x] = 1

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        return os.path.basename(img_path), img, cls_label, mask

    def get_images_and_labels(self, img_root=None, split=None):
        self.img_paths = []
        self.mask_paths = []

        self.mask_paths = glob.glob(os.path.join(img_root, split, "mask", "*.png"))

        for mask_path in self.mask_paths:
            img_name = os.path.basename(mask_path)
            self.img_paths.append(os.path.join(img_root, split, "img", img_name))


class BCSSWSSSDataset(Dataset):
    CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]
    def __init__(self, img_root="../data/BCSS-WSSS/train", mask_name="pseudo_label", transform=None):
        super(BCSSWSSSDataset, self).__init__()
        self.get_images_and_labels(img_root, mask_name)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        cls_label = self.cls_labels[index]
        assert os.path.exists(img_path), "image path: {}, does not exist".format(img_path)
        assert os.path.exists(mask_path), "mask path: {}, does not exist".format(mask_path)

        img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        mask = np.array(Image.open(mask_path))
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        return os.path.basename(img_path), img, cls_label, mask

    def get_images_and_labels(self, img_root, mask_name):
        self.img_paths = []
        self.mask_paths = []
        self.cls_labels = []

        self.img_paths = glob.glob(os.path.join(img_root, "img", "*.png"))

        for img_path in self.img_paths:
            img_name = os.path.basename(img_path)
            self.mask_paths.append(os.path.join(img_root, mask_name, img_name))
            term_split = re.split(r"\[|\]", img_path)
            cls_label = np.array([int(x) for x in term_split[1]])
            self.cls_labels.append(cls_label)


# ---- utils/pyutils.py ----


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(filename='test.log'):
    ## setup logger
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v


# ---- utils/optimizer.py ----
class PolyWarmupAdamW(torch.optim.AdamW):

    def __init__(self, params, lr, weight_decay, betas, warmup_iter=None, max_iter=None, warmup_ratio=None, power=None):
        super().__init__(params, lr=lr, betas=betas,weight_decay=weight_decay, eps=1e-8)

        self.global_step = 0
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.max_iter = max_iter
        self.power = power

        self.__init_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        ## adjust lr
        if self.global_step < self.warmup_iter:

            lr_mult = 1 - (1 - self.global_step / self.warmup_iter) * (1 - self.warmup_ratio)
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        elif self.global_step < self.max_iter: 

            lr_mult = (1 - (self.global_step - self.warmup_iter) / self.max_iter) ** self.power
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        # step
        super().step(closure)

        self.global_step += 1


# ---- utils/evaluate.py ----


def compute_boundary_iou(gt, pred, num_classes, dilation=3):
    """Compute per-class Boundary IoU.
    gt, pred: [H, W] integer class labels.
    Returns: per-class boundary IoU [num_classes] and mean.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation+1, 2*dilation+1))
    per_class_intersection = np.zeros(num_classes, dtype=np.float64)
    per_class_union = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        gt_c = (gt == c).astype(np.uint8)
        pred_c = (pred == c).astype(np.uint8)
        # boundary = mask - erode(mask)
        gt_boundary = gt_c - cv2.erode(gt_c, kernel)
        pred_boundary = pred_c - cv2.erode(pred_c, kernel)
        per_class_intersection[c] = (gt_boundary & pred_boundary).sum()
        per_class_union[c] = (gt_boundary | pred_boundary).sum()

    valid = per_class_union > 0
    per_class_biou = np.zeros(num_classes)
    per_class_biou[valid] = per_class_intersection[valid] / per_class_union[valid]
    return per_class_biou


class ConfusionMatrixAllClass(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat1 = None
        self.mat2 = None

    def update(self, a, b):
        """
        :param a: ground truth
        :param b: pred
        :return:
        """
        n = self.num_classes
        if self.mat1 is None:
            self.mat1 = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        if self.mat2 is None:
            self.mat2 = torch.zeros((2, 2), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k].to(torch.int64)
            self.mat1 += torch.bincount(inds, minlength=n**2).reshape(n, n)
            a[a != 0] = 1
            b[b != 0] = 1
            k = (a >= 0) & (a < 2)
            inds = 2 * a[k].to(torch.int64) + b[k].to(torch.int64)
            self.mat2 += torch.bincount(inds, minlength=2**2).reshape(2, 2)
            del a
            del b

    def reset(self):
        if self.mat1 is not None:
            self.mat1.zero_()
        if self.mat2 is not None:
            self.mat2.zero_()

    def compute(self):
        h = self.mat1.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        dice = 2 * torch.diag(h) / (h.sum(1) + h.sum(0))
        # Frequency-weighted IoU
        freq = h.sum(1) / h.sum()
        fwiu = (freq * iu).sum()
        # Boundary IoU (binary: foreground vs background)
        h_bg_fg = self.mat2.float()
        biu = torch.diag(h_bg_fg) / (h_bg_fg.sum(1) + h_bg_fg.sum(0) - torch.diag(h_bg_fg))
        dice_bg_fg = 2 * torch.diag(h_bg_fg) / (h_bg_fg.sum(1) + h_bg_fg.sum(0))
        return acc_global, acc, iu, dice, dice_bg_fg, fwiu, biu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat1)
        torch.distributed.all_reduce(self.mat2)

    def __str__(self):
        acc_global, acc, iu, dice, fg_bg_dice, fwiu, biu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}\n'
            'dice: {}\n'
            'mean dice: {}\n'
            'fg_bg_dice: {}\n'
            'mean_fg_bg: {}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu[:-1].mean().item() * 100,
                ['{:.1f}'.format(i) for i in (dice * 100).tolist()],
                dice[:-1].mean().item() * 100,
                ['{:.1f}'.format(i) for i in (fg_bg_dice * 100).tolist()],
                fg_bg_dice.mean().item() * 100
            )
    


# ---- utils/contrast_loss.py ----

class InfoNCELossFG(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        print(f'Use InfoNCELossFG with temperature: {temperature}')

    def forward(self, fg_img_feature, fg_pro_feature, bg_pro_feature):
        positive_sims = torch.zeros(1, device=fg_img_feature.device)
        negative_sims = torch.zeros(1, device=fg_img_feature.device)

        fg_img_feature = fg_img_feature / fg_img_feature.norm(dim=-1, keepdim=True)  # [N, D]
        
        batch_size = fg_img_feature.shape[0]
        for i in range(batch_size):
            curr_fg_img = fg_img_feature[i:i+1]  # [1, D]
            curr_fg_pro = fg_pro_feature[i:i+1]  # [1, D]
            curr_bg_pro = bg_pro_feature[i]  # [L, D]
            
            fg_img_fg_pro_logits = curr_fg_img @ curr_fg_pro.t()  # [1, 1]
            fg_img_bg_pro_logits = curr_fg_img @ curr_bg_pro.t()  # [1, L]
            
            positive_sims = positive_sims + torch.exp(fg_img_fg_pro_logits / self.temperature).sum()
            
            negative_sims = negative_sims + \
                           torch.exp(fg_img_fg_pro_logits / self.temperature).sum() + \
                           torch.exp(fg_img_bg_pro_logits / self.temperature).sum()
        loss = -torch.log(positive_sims / negative_sims)

        return loss


class InfoNCELossBG(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        print(f'Use InfoNCELossBG with temperature: {temperature}')

    def forward(self, bg_img_feature, fg_pro_feature, bg_pro_feature):
        positive_sims = torch.zeros(1, device=bg_img_feature.device)
        negative_sims = torch.zeros(1, device=bg_img_feature.device)

        bg_img_feature = bg_img_feature / bg_img_feature.norm(dim=-1, keepdim=True)  # [N, D]
        
        batch_size = bg_img_feature.shape[0]
        for i in range(batch_size):
            curr_bg_img = bg_img_feature[i:i+1]  # [1, D]
            curr_fg_pro = fg_pro_feature[i:i+1]  # [1, D]
            curr_bg_pro = bg_pro_feature[i]  # [L, D]

            bg_img_bg_pro_logits = curr_bg_img @ curr_bg_pro.t()  # [1, L]
            bg_img_fg_pro_logits = curr_bg_img @ curr_fg_pro.t()  # [1, 1]

            positive_sims = positive_sims + torch.exp(bg_img_bg_pro_logits / self.temperature).mean()

            negative_sims = negative_sims + \
                           torch.exp(bg_img_bg_pro_logits / self.temperature).mean() + \
                           torch.exp(bg_img_fg_pro_logits / self.temperature).sum()

        loss = -torch.log(positive_sims / negative_sims)

        return loss

# ---- utils/hierarchical_utils.py ----

def pair_features(fg_features, bg_features, text_features, labels):
    batch_indices, class_indices = torch.where(labels == 1)
    
    paired_fg_features = [] 
    paired_bg_features = [] 
    paired_fg_text = [] 
    paired_bg_text = [] 
 
    for i in range(len(batch_indices)):
        curr_class = class_indices[i]
        
        curr_fg = fg_features[i]  # [D]
        curr_bg = bg_features[i]  # [D]
        
        curr_fg_text = text_features[curr_class]  # [D]
        
        bg_text_indices = [j for j in range(text_features.shape[0]) if j != curr_class]
        curr_bg_text = text_features[bg_text_indices]  # [3, D]
        
        paired_fg_features.append(curr_fg)
        paired_bg_features.append(curr_bg)
        paired_fg_text.append(curr_fg_text)
        paired_bg_text.append(curr_bg_text)
    
    paired_fg_features = torch.stack(paired_fg_features)  # [N, D]
    paired_bg_features = torch.stack(paired_bg_features)  # [N, D]
    paired_fg_text = torch.stack(paired_fg_text)         # [N, D]
    paired_bg_text = torch.stack(paired_bg_text)         # [N, 3, D]
    
    return {
        'fg_features': paired_fg_features,  
        'bg_features': paired_bg_features,  
        'fg_text': paired_fg_text,         
        'bg_text': paired_bg_text,
        'labels': class_indices              # subclass indices for each paired sample
    }


def merge_to_parent_predictions(predictions, k_list, method='max'):
    parent_preds = []
    start_idx = 0
    
    for k in k_list:
        if k > 1:
            class_preds = predictions[:, start_idx:start_idx + k]
            
            if method == 'max':
                class_probs = torch.softmax(class_preds, dim=1)
                parent_pred = (class_probs * class_preds).sum(dim=1)
            else:  # method == 'mean'
                parent_pred = torch.mean(class_preds, dim=1)
            
            parent_preds.append(parent_pred)
        else:
            parent_preds.append(predictions[:, start_idx])
        
        start_idx += k
    parent_preds = torch.stack(parent_preds, dim=1)
    
    return parent_preds


def merge_subclass_cams_to_parent(cams, k_list, method='max'):
    batch_size, _, H, W = cams.shape
    num_parent_classes = len(k_list)

    parent_cams = torch.zeros(batch_size, num_parent_classes, H, W, 
                            device=cams.device, dtype=cams.dtype)
    
    start_idx = 0
    for parent_idx, k in enumerate(k_list):
        if k > 1: 
            subclass_cams = cams[:, start_idx:start_idx + k, :, :]
            
            if method == 'max':
                B, k, H, W = subclass_cams.shape
                cams_flat = subclass_cams.view(B, k, H*W)  # [B, k, H*W]
                cams_probs = torch.softmax(cams_flat, dim=1)  # [B, k, H*W]
                parent_cam_flat = (cams_probs * cams_flat).sum(dim=1)  # [B, H*W]
                parent_cam = parent_cam_flat.view(B, H, W)  # [B, H, W]
            else:  # method == 'mean'
                parent_cam = torch.mean(subclass_cams, dim=1)    # [B, H, W]
            
            parent_cams[:, parent_idx, :, :] = parent_cam
        else:
            parent_cams[:, parent_idx, :, :] = cams[:, start_idx, :, :]
        
        start_idx += k
    
    return parent_cams


def expand_parent_to_subclass_labels(parent_labels, k_list):
    batch_size = parent_labels.size(0)
    total_subclasses = sum(k_list)
    
    subclass_labels = torch.zeros(batch_size, total_subclasses, 
                                device=parent_labels.device, 
                                dtype=parent_labels.dtype)
    start_idx = 0
    for parent_idx, k in enumerate(k_list):
        parent_label = parent_labels[:, parent_idx:parent_idx+1]  # [batch_size, 1]
        subclass_labels[:, start_idx:start_idx+k] = parent_label.repeat(1, k)
        start_idx += k
    
    return subclass_labels 

# ---- utils/fgbg_feature.py ----

class MaskAdapter_DynamicThreshold(nn.Module):
    def __init__(self, alpha, mask_cam=False):
        super(MaskAdapter_DynamicThreshold, self).__init__()

        self.alpha = alpha
        self.mask_cam = mask_cam

        print(f"MaskAdapter_DynamicThreshold:")
        print(f"  alpha: {alpha}")
        print(f"  mask_cam: {mask_cam}")

    def forward(self, x):
        binary_mask = []
        for i in range(x.shape[0]):
            th = torch.max(x[i]) * self.alpha
            binary_mask.append(
                torch.where(x[i] >= th, torch.ones_like(x[0]), torch.zeros_like(x[0]))
            )
        binary_mask = torch.stack(binary_mask, dim=0)

        if self.mask_cam:
            return x * binary_mask
        else:
            return binary_mask

class FeatureExtractor:
    def __init__(self, mask_adapter, clip_size=224):
        self.mask_adapter = mask_adapter
        self.clip_size = clip_size
        
    def prepare_cam_mask(self, cam, N):
        cam_224 = F.interpolate(cam, (self.clip_size, self.clip_size), 
                              mode="bilinear", align_corners=True)
        cam_224 = cam_224.reshape(N * cam.size(1), 1, self.clip_size, self.clip_size)

        cam_224_mask = self.mask_adapter(cam_224)
        
        return cam_224, cam_224_mask
        
    def prepare_image(self, img):
        return F.interpolate(img, (self.clip_size, self.clip_size), 
                           mode="bilinear", align_corners=True)
        
    @torch.amp.autocast('cuda')
    def extract_features(self, img_224, cam_224, cam_224_mask, label):
        batch_indices, class_indices = torch.where(label == 1)
        
        img_selected = img_224[batch_indices]  # [N, C, H, W]
        cam_selected = cam_224[batch_indices, class_indices]  # [N, H, W]
        mask_selected = cam_224_mask[batch_indices, class_indices]  # [N, H, W]
        
        cam_expanded = cam_selected.unsqueeze(1)  # [N, 1, H, W]
        mask_expanded = mask_selected.unsqueeze(1)  # [N, 1, H, W]
        
        fg_features = cam_expanded * img_selected  # [N, C, H, W]
        bg_features = (1 - cam_expanded) * img_selected  # [N, C, H, W]
        
        fg_masks = mask_expanded  # [N, 1, H, W]
        bg_masks = 1 - mask_expanded  # [N, 1, H, W]
        
        return fg_features, bg_features, fg_masks, bg_masks
        
    @torch.amp.autocast('cuda')
    def get_masked_features(self, fg_features, bg_features, fg_masks, bg_masks, clip_model):
        
        fg_min = fg_features.amin(dim=(2, 3), keepdim=True)
        fg_max = fg_features.amax(dim=(2, 3), keepdim=True)
        normalized_fg_features = (fg_features - fg_min) / (fg_max - fg_min + 1e-8)
        
        bg_min = bg_features.amin(dim=(2, 3), keepdim=True)
        bg_max = bg_features.amax(dim=(2, 3), keepdim=True)
        normalized_bg_features = (bg_features - bg_min) / (bg_max - bg_min + 1e-8)
        
        # CLIP is frozen — no_grad saves ~30-40% compute and memory
        with torch.no_grad():
            fg_img_features = clip_model.vision_model(normalized_fg_features*fg_masks)
            bg_img_features = clip_model.vision_model(normalized_bg_features*bg_masks)
            
        return fg_img_features, bg_img_features

        
    @torch.amp.autocast('cuda')
    def process_batch(self, inputs, cam, label, clip_model):
        if not torch.any(label == 1):
            return None
            
        cam_224 = F.interpolate(cam, (self.clip_size, self.clip_size), 
                               mode="bilinear", align_corners=True)
       
        cam_224_mask = self.mask_adapter(cam_224)
        
        fg_features, bg_features, fg_masks, bg_masks = self.extract_features(
            inputs, cam_224, cam_224_mask, label
        )
        
        fg_features, bg_features = self.get_masked_features(
            fg_features, bg_features, fg_masks, bg_masks, clip_model
        )
        
        return {
            'fg_features': fg_features,
            'bg_features': bg_features,
            'fg_masks': fg_masks,
            'bg_masks': bg_masks,
            'cam_224': cam_224,
            'cam_224_mask': cam_224_mask
        }

    def print_debug_info(self, batch_info):
        if batch_info is None:
            print("No foreground samples in batch")
            return
                
        print("\nFeature extraction debug info:")
        print(f"Number of foreground samples: {batch_info['fg_masks'].shape[0]}")
        print(f"Foreground features shape: {batch_info['fg_features'].shape}")
        print(f"Background features shape: {batch_info['bg_features'].shape}")
        print(f"CAM shape: {batch_info['cam_224'].shape}")
        print(f"Mask shape: {batch_info['cam_224_mask'].shape}")
        print(f"Foreground mask mean: {batch_info['fg_masks'].mean().item():.4f}")
        print(f"Background mask mean: {batch_info['bg_masks'].mean().item():.4f}")

# ---- SISH-Inspired Modules ----

class VQCodebook(nn.Module):
    """Semantic Vector Quantization codebook with EMA updates.
    Uses cosine similarity for codebook matching.
    Supports text-guided initialization: codebook entries are seeded from
    text prototype projections (macro→micro knowledge transfer).
    """
    def __init__(self, num_codes=64, code_dim=256, commitment_weight=0.25,
                 ema_decay=0.99, dead_code_threshold=2):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold

        # Codebook embeddings (NOT trained by gradient — updated via EMA)
        codebook = torch.randn(num_codes, code_dim)
        codebook = F.normalize(codebook, dim=-1)  # init normalized
        self.register_buffer('codebook', codebook)
        self.register_buffer('ema_count', torch.zeros(num_codes))
        self.register_buffer('ema_weight', self.codebook.clone())
        self.register_buffer('usage_count', torch.zeros(num_codes))
        self._initialized = False

    def init_from_prototypes(self, proto_codes):
        """Text-guided init: seed codebook from projected text prototypes.
        proto_codes: [num_protos, code_dim] — text protos projected to code space.
        Each prototype spawns num_codes/num_protos entries with small noise.
        """
        num_protos = proto_codes.shape[0]
        codes_per_proto = self.num_codes // num_protos
        remainder = self.num_codes - codes_per_proto * num_protos
        all_codes = []
        for i in range(num_protos):
            n_copies = codes_per_proto + (1 if i < remainder else 0)
            base = proto_codes[i:i+1].expand(n_copies, -1)  # [n, D]
            noise = torch.randn_like(base) * 0.05  # small perturbation
            all_codes.append(base + noise)
        all_codes = F.normalize(torch.cat(all_codes, dim=0), dim=-1)  # [K, D]
        self.codebook.copy_(all_codes)
        self.ema_weight.copy_(all_codes)
        self._initialized = True
        print(f"    VQ codebook initialized from {num_protos} text prototypes "
              f"({codes_per_proto}+ codes each)")

    def _init_from_data(self, z_c):
        """Fallback: initialize from first batch of encoder outputs (if no text init)."""
        if self._initialized:
            return  # already initialized from text prototypes
        n = min(z_c.shape[0], self.num_codes)
        idx = torch.randperm(z_c.shape[0])[:n]
        init_data = F.normalize(z_c[idx].detach().float(), dim=-1)
        self.codebook[:n] = init_data
        self.ema_weight[:n] = init_data
        self._initialized = True

    def forward(self, z_c):
        # z_c: [N, D] continuous features
        if not self._initialized and self.training:
            self._init_from_data(z_c)

        # Cosine distance: normalize both, then max similarity
        z_norm = F.normalize(z_c, dim=-1)
        cb_norm = F.normalize(self.codebook, dim=-1)
        cosine_sim = z_norm @ cb_norm.t()          # [N, K]
        indices = torch.argmax(cosine_sim, dim=1)  # [N]
        z_q = self.codebook[indices]                # [N, D]

        # EMA codebook update (only during training)
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.num_codes).float()  # [N, K]
                batch_count = one_hot.sum(dim=0)                      # [K]
                batch_sum = one_hot.t() @ z_c.detach().float()        # [K, D]

                self.ema_count = self.ema_decay * self.ema_count + (1 - self.ema_decay) * batch_count
                self.ema_weight = self.ema_decay * self.ema_weight + (1 - self.ema_decay) * batch_sum

                # Laplace smoothing + normalize codebook
                n = self.ema_count.sum()
                count_smoothed = (self.ema_count + 1e-5) / (n + self.num_codes * 1e-5) * n
                self.codebook = self.ema_weight / count_smoothed.unsqueeze(1)
                self.codebook = F.normalize(self.codebook, dim=-1)  # keep normalized

                # Track usage for dead code revival
                self.usage_count += batch_count

            # Dead code revival: reset unused codes to random encoder outputs
            if self.usage_count.sum() > 0:
                dead_mask = self.usage_count < self.dead_code_threshold
                num_dead = dead_mask.sum().item()
                if num_dead > 0 and z_c.shape[0] > 0:
                    num_replace = min(num_dead, z_c.shape[0])
                    replace_idx = torch.randperm(z_c.shape[0])[:num_replace]
                    dead_indices = dead_mask.nonzero(as_tuple=True)[0][:num_replace]
                    with torch.no_grad():
                        new_codes = F.normalize(z_c[replace_idx].detach().float(), dim=-1)
                        self.codebook[dead_indices] = new_codes
                        self.ema_weight[dead_indices] = new_codes
                        self.ema_count[dead_indices] = 1.0
                        self.usage_count[dead_indices] = 0.0

        # Commitment loss on normalized space (cosine-friendly)
        commitment_loss = F.mse_loss(z_norm, F.normalize(z_q, dim=-1).detach())
        vq_loss = self.commitment_weight * commitment_loss

        # Straight-through estimator
        z_q_st = z_c + (z_q - z_c).detach()
        return z_q_st, vq_loss


class VQCAMHead(nn.Module):
    """Produces discrete CAM from VQ codebook for one scale.
    Includes semantic regularization and text-guided codebook initialization."""
    def __init__(self, in_dim, num_classes, num_codes=128, code_dim=256,
                 commitment_weight=0.25, proto_dim=512):
        super().__init__()
        self.code_dim = code_dim
        self.proto_dim = proto_dim
        self.project = nn.Sequential(
            nn.Linear(in_dim, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, code_dim),
        )
        self.vq = VQCodebook(num_codes, code_dim, commitment_weight)
        self.cam_head = nn.Conv2d(code_dim, num_classes, 1)
        # Semantic regularization: project VQ codes → prototype space
        self.sem_reg_proj = nn.Linear(code_dim, proto_dim, bias=False)
        # Reverse projection: prototype space → code space (for text-guided init)
        self.proto_to_code = nn.Linear(proto_dim, code_dim, bias=False)

    def init_codebook_from_prototypes(self, text_prototypes):
        """Initialize VQ codebook from text prototypes (macro→micro transfer).
        text_prototypes: [num_protos, proto_dim] (e.g. [12, 512])
        Projects text protos to code_dim, then seeds codebook.
        """
        with torch.no_grad():
            proto_codes = self.proto_to_code(text_prototypes.float())  # [12, 256]
            proto_codes = F.normalize(proto_codes, dim=-1)
            self.vq.init_from_prototypes(proto_codes)

    def forward(self, features, H, W):
        # features: [B*H*W, in_dim]
        B = features.shape[0] // (H * W)
        z = self.project(features)                          # [B*H*W, code_dim]
        z_q, vq_loss = self.vq(z)                           # [B*H*W, code_dim]
        z_q = z_q.view(B, H, W, self.code_dim).permute(0, 3, 1, 2)  # [B, D, H, W]
        cam_vq = self.cam_head(z_q)                         # [B, num_classes, H, W]
        return cam_vq, vq_loss

    def compute_sem_reg_loss(self, prototypes, tau=0.1):
        """Semantic regularization: pull VQ codes toward nearest prototypes.
        prototypes: [num_protos, proto_dim] (l_fea, the hybrid prototypes)
        Returns scalar loss.
        """
        # Project VQ codes to prototype space
        vq_codes = self.vq.codebook.detach()  # [128, 256] — detach to not backprop through EMA codebook
        vq_proj = self.sem_reg_proj(vq_codes)  # [128, 512]
        vq_proj = F.normalize(vq_proj, dim=-1)
        proto_norm = F.normalize(prototypes.detach(), dim=-1)  # [12, 512]

        # Soft assignment: each VQ code → which prototype?
        sim = vq_proj @ proto_norm.t()  # [128, 12]
        assign = F.softmax(sim / tau, dim=1)  # [128, 12]

        # Target: weighted combination of prototypes
        target = assign @ proto_norm  # [128, 512]
        target = F.normalize(target, dim=-1)

        # Loss: maximize cosine similarity between projected VQ codes and targets
        loss = 1.0 - (vq_proj * target).sum(dim=-1).mean()
        return loss


class MultiResConsistencyLoss(nn.Module):
    """Cross-scale CAM consistency via soft KL divergence.
    Treats CAMs as spatial probability distributions and allows scales to
    differ (P2=texture, P4=semantic) while staying related.
    Temperature tau controls softness: high tau = more permissive.
    """
    def __init__(self, tau=4.0):
        super().__init__()
        self.tau = tau

    def forward(self, cam_fine, cam_coarse):
        # Downsample fine to match coarse spatial size
        cam_fine_down = F.adaptive_avg_pool2d(cam_fine, cam_coarse.shape[2:])
        # Treat CAMs as spatial distributions per class (soft KL)
        p = F.softmax(cam_fine_down.flatten(2) / self.tau, dim=-1).clamp(min=1e-8)
        q = F.softmax(cam_coarse.flatten(2) / self.tau, dim=-1).clamp(min=1e-8)
        # KL(p || q) — p=fine guides q=coarse
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        return kl_loss


class HashProjectionHead(nn.Module):
    """Projects features to binary hash codes with straight-through estimator."""
    def __init__(self, in_dim=512, hash_dim=128):
        super().__init__()
        self.fc = nn.Linear(in_dim, hash_dim)
        self.bn = nn.BatchNorm1d(hash_dim)

    def forward(self, x):
        h = self.bn(self.fc(x))
        # sign() with straight-through estimator for gradient
        binary = h + (torch.sign(h) - h).detach()
        return binary, h


class HashMemoryBank:
    """FIFO memory bank storing patch features + hashes + labels for hard negative mining.
    Stores features from previous batches so HashContrastiveLoss can mine
    hard negatives across thousands of patches instead of just 3 text prototypes.
    """
    def __init__(self, capacity=4096, feat_dim=512, hash_dim=128):
        self.capacity = capacity
        self.feat_dim = feat_dim
        self.hash_dim = hash_dim
        self.features = None   # [capacity, feat_dim]
        self.hashes = None     # [capacity, hash_dim]
        self.labels = None     # [capacity] — class index per entry
        self.ptr = 0
        self.full = False

    def push(self, features, hashes, labels):
        """Add batch of features to memory bank (FIFO)."""
        features = features.detach()
        hashes = hashes.detach()
        labels = labels.detach()
        n = features.shape[0]
        device = features.device

        if self.features is None:
            self.features = torch.zeros(self.capacity, self.feat_dim, device=device)
            self.hashes = torch.zeros(self.capacity, self.hash_dim, device=device)
            self.labels = torch.zeros(self.capacity, dtype=torch.long, device=device)

        if n >= self.capacity:
            self.features = features[-self.capacity:].clone()
            self.hashes = hashes[-self.capacity:].clone()
            self.labels = labels[-self.capacity:].clone()
            self.ptr = 0
            self.full = True
            return

        end = self.ptr + n
        if end <= self.capacity:
            self.features[self.ptr:end] = features
            self.hashes[self.ptr:end] = hashes
            self.labels[self.ptr:end] = labels
        else:
            first = self.capacity - self.ptr
            self.features[self.ptr:] = features[:first]
            self.hashes[self.ptr:] = hashes[:first]
            self.labels[self.ptr:] = labels[:first]
            rest = n - first
            self.features[:rest] = features[first:]
            self.hashes[:rest] = hashes[first:]
            self.labels[:rest] = labels[first:]
            self.full = True
        self.ptr = end % self.capacity

    @property
    def size(self):
        return self.capacity if self.full else self.ptr

    def get_all(self):
        """Return all valid entries."""
        s = self.size
        return self.features[:s], self.hashes[:s], self.labels[:s]


class HashHardNegativeMiner(nn.Module):
    """Hash-guided hard negative mining for InfoNCE.
    Instead of replacing InfoNCE, this module ENHANCES it by mining
    hard negatives from a memory bank using binary hash codes.
    Returns hard negative features that are fed back to InfoNCE.
    """
    def __init__(self, in_dim=512, hash_dim=128, top_k=16,
                 temperature=0.07, quant_weight=0.01, bank_capacity=4096):
        super().__init__()
        self.hash_head = HashProjectionHead(in_dim, hash_dim)
        self.top_k = top_k
        self.temperature = temperature
        self.quant_weight = quant_weight
        self.hash_dim = hash_dim
        self.memory_bank = HashMemoryBank(capacity=bank_capacity,
                                          feat_dim=in_dim, hash_dim=hash_dim)
        print(f"HashHardNegativeMiner: bank_capacity={bank_capacity}, top_k={top_k}")

    def forward(self, fg_features, fg_labels=None):
        """Mine hard negatives and compute auxiliary hash loss."""
        device = fg_features.device
        batch_size = fg_features.shape[0]

        fg_hash, fg_h = self.hash_head(fg_features)
        fg_norm = F.normalize(fg_features, dim=-1)

        quant_loss = self.quant_weight * (1 - fg_h.abs()).mean()

        bank_size = self.memory_bank.size
        use_bank = bank_size >= self.top_k

        hard_neg_loss = torch.tensor(0.0, device=device)

        if use_bank and fg_labels is not None:
            bank_feats, bank_hashes, bank_labels = self.memory_bank.get_all()
            bank_feats_norm = F.normalize(bank_feats, dim=-1)

            for i in range(batch_size):
                curr_fg = fg_norm[i:i+1]
                curr_label = fg_labels[i]

                with torch.no_grad():
                    curr_hash_det = fg_hash[i:i+1].detach()
                    hamming_dist = (self.hash_dim - curr_hash_det @ bank_hashes.t()) / 2
                    hamming_dist = hamming_dist.squeeze(0)

                    same_class_mask = (bank_labels == curr_label)
                    hamming_dist[same_class_mask] = float('inf')
                    valid_k = min(self.top_k, (~same_class_mask).sum().item())

                if valid_k > 0:
                    _, top_k_idx = hamming_dist.topk(valid_k, largest=False)
                    hard_negs = bank_feats_norm[top_k_idx]

                    pos_mask = (bank_labels == curr_label)
                    if pos_mask.sum() > 0:
                        pos_feats = bank_feats_norm[pos_mask][:self.top_k]
                        pos_sim = torch.exp(curr_fg @ pos_feats.t() / self.temperature).sum()
                    else:
                        pos_sim = torch.exp(torch.tensor(1.0 / self.temperature, device=device))

                    neg_sim = torch.exp(curr_fg @ hard_negs.t() / self.temperature).sum()
                    hard_neg_loss = hard_neg_loss - torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))

            hard_neg_loss = hard_neg_loss / max(batch_size, 1)

        if self.training and fg_labels is not None:
            self.memory_bank.push(fg_features.detach(), fg_hash.detach(), fg_labels.detach())

        return hard_neg_loss + quant_loss


# ---- utils/trainutils.py ----


def get_wsss_dataset(cfg, val_root=None):
    MEAN, STD = get_mean_std(cfg.dataset.name)
    _val_root = val_root if val_root is not None else cfg.dataset.val_root

    transform = {
        "train": A.Compose([
            A.Normalize(MEAN, STD),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(),
            ToTensorV2(transpose_mask=True),
        ]),
        "val": A.Compose([
            A.Normalize(MEAN, STD),
            ToTensorV2(transpose_mask=True),
        ])
    }
    train_dataset = BCSSWSSSDataset(cfg.dataset.train_root, mask_name=cfg.dataset.mask_root,transform=transform["train"])
    val_dataset = BCSSTestDataset(_val_root, split="valid", transform=transform["val"])

    return train_dataset, val_dataset


def get_cls_dataset(cfg, split="val", val_root=None, p=0.5, enable_rotation=True):
    """Load training + val/test dataset.
    
    Args:
        val_root: Override path for val/test dataset. If None, uses cfg.dataset.val_root.
                  Pass cfg.dataset.test_root to evaluate on the test split.
    """
    MEAN, STD = get_mean_std(cfg.dataset.name)
    _val_root = val_root if val_root is not None else cfg.dataset.val_root

    train_transforms = [
        A.Normalize(MEAN, STD),
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
    ]
    if enable_rotation:
        train_transforms.append(A.RandomRotate90())
    train_transforms.append(ToTensorV2(transpose_mask=True))

    transform = {
        "train": A.Compose(train_transforms),
        "val": A.Compose([
            A.Normalize(MEAN, STD),
            ToTensorV2(transpose_mask=True),
        ]),
    }

    train_dataset = BCSSTrainingDataset(cfg.dataset.train_root, transform=transform["train"])
    val_dataset = BCSSTestDataset(_val_root, split, transform=transform["val"])

    return train_dataset, val_dataset


def get_mean_std(dataset):
    norm = [[0.66791496, 0.47791372, 0.70623304], [0.1736589,  0.22564577, 0.19820057]]
    return norm[0], norm[1]


def all_reduced(x, n_gpus):
    dist.all_reduce(x)
    x /= n_gpus


# ---- utils/validate.py ----


def get_seg_label(cams, inputs, label):
    """Generate segmentation labels from CAMs"""
    with torch.no_grad():
        b, c, h, w = inputs.shape
        label = label.view(b, -1, 1, 1).cpu().data.numpy()
        cams = cams.cpu().data.numpy()
        cams = np.maximum(cams, 0)
        
        # Normalize CAMs to [0,1]
        channel_max = np.max(cams, axis=(2, 3), keepdims=True)
        channel_min = np.min(cams, axis=(2, 3), keepdims=True)
        cams = (cams - channel_min) / (channel_max - channel_min + 1e-6)
        cams = cams * label
        cams = torch.from_numpy(cams).float()
        
        # Resize to input dimensions
        cams = F.interpolate(cams, size=(h, w), mode="bilinear", align_corners=True)
        cam_max = torch.max(cams, dim=1, keepdim=True)[0]
        bg_cam = (1 - cam_max) ** 10
        cam_all = torch.cat([cams, bg_cam], dim=1)

    return cams



def validate(model=None, data_loader=None, cfg=None, cls_loss_func=None):
    """Validation function with test-time augmentation"""
    model.eval()
    avg_meter = AverageMeter("all_cls_acc4", "avg_cls_acc4", "cls_loss")
    fuse234_matrix = ConfusionMatrixAllClass(num_classes=cfg.dataset.cls_num_classes + 1)
    boundary_iou_sum = np.zeros(5, dtype=np.float64)  # 4 classes + BG
    boundary_iou_count = 0
    
    # Test-time augmentation setup
    use_tta = getattr(cfg.train, 'use_tta', True)
    if use_tta:
        tta_transform = tta.Compose([
            tta.HorizontalFlip(),
            tta.Multiply(factors=[0.9, 1.0, 1.1])
        ])
    with torch.no_grad():
        for data in tqdm(data_loader,
                         total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data
            inputs = inputs.cuda().float()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda().float()

            cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list, _ = model(inputs, )

            cls1 = merge_to_parent_predictions(cls1, k_list, method=cfg.train.merge_test)
            cls2 = merge_to_parent_predictions(cls2, k_list, method=cfg.train.merge_test)
            cls3 = merge_to_parent_predictions(cls3, k_list, method=cfg.train.merge_test)
            cls4 = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_test)

            cls_loss1 = cls_loss_func(cls1, cls_label)
            cls_loss2 = cls_loss_func(cls2, cls_label)
            cls_loss3 = cls_loss_func(cls3, cls_label)
            cls_loss4 = cls_loss_func(cls4, cls_label)
            cls_loss = cfg.train.l1 * cls_loss1 + cfg.train.l2 * cls_loss2 + cfg.train.l3 * cls_loss3 + cfg.train.l4 * cls_loss4

            cls4 = (torch.sigmoid(cls4) > 0.5).float()
            all_cls_acc4 = (cls4 == cls_label).all(dim=1).float().sum() / cls4.shape[0] * 100
            avg_cls_acc4 = ((cls4 == cls_label).sum(dim=0) / cls4.shape[0]).mean() * 100
            avg_meter.add({"all_cls_acc4": all_cls_acc4, "avg_cls_acc4": avg_cls_acc4, "cls_loss": cls_loss})
            
            if use_tta:
                # Generate evaluation CAMs with TTA
                cams1 = []
                cams2 = []
                cams3 = []
                cams4 = []
                for tta_trans in tta_transform:
                    augmented_tensor = tta_trans.augment_image(inputs)
                    cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list, _ = model(augmented_tensor)

                    cam1 = merge_subclass_cams_to_parent(cam1, k_list, method=cfg.train.merge_test)
                    cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test)
                    cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test)
                    cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test)

                    cls1 = merge_to_parent_predictions(cls1, k_list, method=cfg.train.merge_test)
                    cls2 = merge_to_parent_predictions(cls2, k_list, method=cfg.train.merge_test)
                    cls3 = merge_to_parent_predictions(cls3, k_list, method=cfg.train.merge_test)
                    cls4 = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_test)

                    cam1 = get_seg_label(cam1, augmented_tensor, cls_label).cuda()
                    cam1 = tta_trans.deaugment_mask(cam1).unsqueeze(dim=0)
                    cams1.append(cam1)
                    cam2 = get_seg_label(cam2, augmented_tensor, cls_label).cuda()
                    cam2 = tta_trans.deaugment_mask(cam2).unsqueeze(dim=0)
                    cams2.append(cam2)
                    cam3 = get_seg_label(cam3, augmented_tensor, cls_label).cuda()
                    cam3 = tta_trans.deaugment_mask(cam3).unsqueeze(dim=0)
                    cams3.append(cam3)
                    cam4 = get_seg_label(cam4, augmented_tensor, cls_label).cuda()
                    cam4 = tta_trans.deaugment_mask(cam4).unsqueeze(dim=0)
                    cams4.append(cam4)

                cams1 = torch.cat(cams1, dim=0).mean(dim=0)
                cams2 = torch.cat(cams2, dim=0).mean(dim=0)
                cams3 = torch.cat(cams3, dim=0).mean(dim=0)
                cams4 = torch.cat(cams4, dim=0).mean(dim=0)
            else:
                # No TTA: single forward pass for CAMs
                cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list, _ = model(inputs)

                cam1 = merge_subclass_cams_to_parent(cam1, k_list, method=cfg.train.merge_test)
                cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test)
                cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test)
                cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test)

                cams1 = get_seg_label(cam1, inputs, cls_label).cuda()
                cams2 = get_seg_label(cam2, inputs, cls_label).cuda()
                cams3 = get_seg_label(cam3, inputs, cls_label).cuda()
                cams4 = get_seg_label(cam4, inputs, cls_label).cuda()

            # Fuse multi-scale predictions
            fuse234 = 0.3 * cams2 + 0.3 * cams3 + 0.4 * cams4
            fuse_label234 = torch.argmax(fuse234, dim=1).cuda()
            
            fuse234_matrix.update(labels.detach().clone(), fuse_label234.clone())

            # Accumulate Boundary IoU
            gt_np = labels.cpu().numpy()    # [B, H, W]
            pred_np = fuse_label234.cpu().numpy()
            for b in range(gt_np.shape[0]):
                b_biou = compute_boundary_iou(gt_np[b], pred_np[b], num_classes=5)  # 4 classes + BG
                boundary_iou_sum += b_biou
                boundary_iou_count += 1

    all_cls_acc4, avg_cls_acc4, cls_loss = avg_meter.pop('all_cls_acc4'), avg_meter.pop("avg_cls_acc4"), avg_meter.pop(
        "cls_loss")
    acc_global, acc, iu, dice, dice_bg_fg, fwiu, biu = fuse234_matrix.compute()
    boundary_iou_per_class = boundary_iou_sum / max(boundary_iou_count, 1)
    metrics = {
        'iou': iu,           # per-class IoU [C+1]
        'miou': iu[:-1].mean(),
        'fwiou': fwiu,
        'binary_iou': biu[1] if len(biu) > 1 else biu.mean(),  # foreground Binary IoU (fg vs bg)
        'boundary_iou': boundary_iou_per_class[:4].mean(),  # mean Boundary IoU (4 FG classes)
        'dice': dice[:-1].mean(),
    }
    model.train()
    return all_cls_acc4, avg_cls_acc4, metrics, cls_loss


class DenseCRF(object):
    """Dense CRF post-processing for refining segmentation boundaries."""
    def __init__(self, iter_max=10, pos_w=3, pos_xy_std=1,
                 bi_w=4, bi_xy_std=67, bi_rgb_std=3):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        """
        Args:
            image: [H, W, 3] uint8 RGB image
            probmap: [C, H, W] float32 probability map
        Returns:
            [C, H, W] refined probability map
        """
        C, H, W = probmap.shape
        U = crf_utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)
        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )
        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))
        return Q


def generate_cam(model=None, data_loader=None, cfg=None, cls_loss_func=None):

    model.eval()
    
    # CRF post-processing setup
    use_crf = getattr(cfg.train, 'use_crf', False) and CRF_AVAILABLE
    if use_crf:
        crf_processor = DenseCRF(
            iter_max=getattr(cfg.train, 'crf_iter_max', 10),
            pos_w=getattr(cfg.train, 'crf_pos_w', 3),
            pos_xy_std=getattr(cfg.train, 'crf_pos_xy_std', 1),
            bi_w=getattr(cfg.train, 'crf_bi_w', 4),
            bi_xy_std=getattr(cfg.train, 'crf_bi_xy_std', 67),
            bi_rgb_std=getattr(cfg.train, 'crf_bi_rgb_std', 3),
        )
        MEAN = np.array([0.66791496, 0.47791372, 0.70623304])
        STD = np.array([0.1736589, 0.22564577, 0.19820057])
        print("CRF post-processing enabled")

    PALETTE = [
        [255, 0, 0],      # TUM
        [0, 255, 0],      # STR
        [0, 0, 255],      # LYM
        [153, 0, 255],    # NEC
        [255, 255, 255],  # BACK
        [0, 0, 0],
    ]

    with torch.no_grad():
        for data in tqdm(data_loader,
                         total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data

            inputs = inputs.cuda().float()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda().float()

            cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list, _ = model(inputs, )

            cam1 = merge_subclass_cams_to_parent(cam1, k_list, method=cfg.train.merge_test)
            cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test)
            cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test)
            cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test)

            cam1 = get_seg_label(cam1, inputs, cls_label).cuda()
            cam2 = get_seg_label(cam2, inputs, cls_label).cuda()
            cam3 = get_seg_label(cam3, inputs, cls_label).cuda()
            cam4 = get_seg_label(cam4, inputs, cls_label).cuda()

            fuse234 = 0.3 * cam2 + 0.3 * cam3 + 0.4 * cam4
            if use_crf:
                # Apply CRF per sample: need original image (uint8 RGB) + softmax probs
                probs = F.softmax(fuse234, dim=1)  # [B, C, H, W]
                for i in range(len(name)):
                    # Denormalize image for CRF
                    img_np = inputs[i].cpu().numpy().transpose(1, 2, 0)  # [H,W,3]
                    img_rgb = ((img_np * STD + MEAN) * 255).clip(0, 255).astype(np.uint8)
                    # CRF refinement
                    refined = crf_processor(img_rgb, probs[i].cpu().numpy().astype(np.float32))
                    label = np.argmax(refined, axis=0).astype(np.uint8)
                    pred_mask = Image.fromarray(label).convert('P')
                    flat_palette = [val for sublist in PALETTE for val in sublist]
                    pred_mask.putpalette(flat_palette)
                    pred_mask.save(os.path.join(cfg.work_dir.pred_dir, name[i] + ".png"))
            else:
                output_fuse234 = torch.argmax(fuse234, dim=1).long()
                for i in range(len(output_fuse234)):
                    pred_mask = Image.fromarray(output_fuse234[i].cpu().clone().squeeze().numpy().astype(np.uint8)).convert('P')
                    flat_palette = [val for sublist in PALETTE for val in sublist]
                    pred_mask.putpalette(flat_palette)
                    pred_mask.save(os.path.join(cfg.work_dir.pred_dir, name[i] + ".png"))
    model.train()
    return 

# ---------------------------------------------------------------

# ---------------------------------------------------------------
# UNI ViT-L/16 Encoder + LoRA Adapters + Simple Feature Pyramid
# Replaces Segformer MixVisionTransformer backbone
# ---------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for a frozen nn.Linear.

    output = frozen_linear(x) + scale * (x @ A.T @ B.T)
    Only A and B are trainable.  scale = alpha / rank.
    """

    def __init__(self, frozen_linear: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        self.frozen = frozen_linear  # kept frozen
        in_f, out_f = frozen_linear.in_features, frozen_linear.out_features
        # Match device and dtype of frozen layer to avoid CPU/GPU mismatch
        device = frozen_linear.weight.device
        dtype = frozen_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, in_f, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        # Frozen path (no grad)
        with torch.no_grad():
            h = self.frozen(x)
        # LoRA path (trainable)
        h = h + self.scale * (x @ self.lora_A.T @ self.lora_B.T)
        return h


class UNIEncoder(nn.Module):
    """Frozen UNI ViT-L/16 feature extractor with LoRA adapters.
    Extracts intermediate features at 4 block positions.
    LoRA is injected into attn.qkv of the target blocks.
    All outputs share the same spatial size: [B, 1024, grid, grid].
    """
    BLOCK_IDXS = (5, 11, 17, 23)   # ViT-L has 24 blocks
    EMBED_DIM  = 1024

    def __init__(self, hf_token, img_size=224, lora_rank=8, lora_alpha=8.0, use_lora=True):
        super().__init__()
        from huggingface_hub import login as hf_login
        hf_login(token=hf_token, add_to_git_credential=False)
        self.model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
            img_size=img_size,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.grid_size = img_size // self.model.patch_embed.patch_size[0]  # 14

        # --- Inject LoRA into target blocks if enabled ---
        if use_lora:
            self._inject_lora(lora_rank, lora_alpha)

    def _inject_lora(self, rank, alpha):
        """Replace attn.qkv in target blocks with LoRA-wrapped version."""
        for idx in self.BLOCK_IDXS:
            block = self.model.blocks[idx]
            original_qkv = block.attn.qkv  # frozen nn.Linear(1024, 3072)
            block.attn.qkv = LoRALinear(original_qkv, rank=rank, alpha=alpha)

    def forward(self, x):
        """x: [B, 3, H, W] -> list of 4 tensors [B, 1024, G, G]"""
        B = x.shape[0]
        G = self.grid_size
        D = self.EMBED_DIM

        # Use forward hooks on specific blocks — reliable across all timm versions
        captured = {}

        def make_hook(block_idx):
            def hook(module, inp, out):
                # out may be Tensor [B, N, D] or tuple; extract tensor
                t = out[0] if isinstance(out, tuple) else out
                # Strip CLS prefix token if N = G*G+1
                if t.ndim == 3 and t.shape[1] == G * G + 1:
                    t = t[:, 1:, :]
                captured[block_idx] = t.reshape(B, G, G, D).permute(0, 3, 1, 2).contiguous()
            return hook

        hooks = [
            self.model.blocks[idx].register_forward_hook(make_hook(idx))
            for idx in self.BLOCK_IDXS
        ]
        try:
            self.model(x)
        finally:
            for h in hooks:
                h.remove()

        return [captured[idx] for idx in self.BLOCK_IDXS]  # 4 x [B, 1024, 14, 14]


class SimpleFeaturePyramid(nn.Module):
    """ViTDet-style Simple Feature Pyramid.

    Takes 4 feature maps from UNI (all [B, 1024, 14, 14]) and produces
    multi-scale outputs matching Segformer mit_b1 spatial sizes and dims:
        P1: [B, D1, 56, 56]  <- 2x deconv upsample from block 5
        P2: [B, D2, 28, 28]  <- 1x deconv upsample from block 11
        P3: [B, D3, 14, 14]  <- identity projection from block 17
        P4: [B, D4,  7,  7]  <- stride-2 pool from block 23
    """
    def __init__(self, in_dim=1024, out_dims=(64, 128, 320, 512)):
        super().__init__()
        D1, D2, D3, D4 = out_dims

        # P1: 14 -> 28 -> 56  (2x deconv upsample)
        self.up1 = nn.Sequential(
            nn.Conv2d(in_dim, D1, 1, bias=False), nn.BatchNorm2d(D1), nn.GELU(),
            nn.ConvTranspose2d(D1, D1, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(D1), nn.GELU(),
            nn.ConvTranspose2d(D1, D1, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(D1), nn.GELU(),
            nn.Conv2d(D1, D1, 3, padding=1, bias=False), nn.BatchNorm2d(D1), nn.GELU(),
        )
        # P2: 14 -> 28  (1x deconv upsample)
        self.up2 = nn.Sequential(
            nn.Conv2d(in_dim, D2, 1, bias=False), nn.BatchNorm2d(D2), nn.GELU(),
            nn.ConvTranspose2d(D2, D2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(D2), nn.GELU(),
            nn.Conv2d(D2, D2, 3, padding=1, bias=False), nn.BatchNorm2d(D2), nn.GELU(),
        )
        # P3: 14 -> 14  (identity + channel projection)
        self.proj3 = nn.Sequential(
            nn.Conv2d(in_dim, D3, 1, bias=False), nn.BatchNorm2d(D3), nn.GELU(),
            nn.Conv2d(D3, D3, 3, padding=1, bias=False), nn.BatchNorm2d(D3), nn.GELU(),
        )
        # P4: 14 -> 7  (stride-2 pool + channel projection)
        self.down4 = nn.Sequential(
            nn.Conv2d(in_dim, D4, 1, bias=False), nn.BatchNorm2d(D4), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(D4, D4, 3, padding=1, bias=False), nn.BatchNorm2d(D4), nn.GELU(),
        )

    def forward(self, feats):
        f5, f11, f17, f23 = feats   # all [B, 1024, 14, 14]
        p1 = self.up1(f5)           # [B, D1, 56, 56]
        p2 = self.up2(f11)          # [B, D2, 28, 28]
        p3 = self.proj3(f17)        # [B, D3, 14, 14]
        p4 = self.down4(f23)        # [B, D4,  7,  7]
        return p1, p2, p3, p4


# ---- model/model.py ----


class AdaptiveLayer(nn.Module):
    def __init__(self, in_dim, n_ratio, out_dim):
        super().__init__()
        hidden_dim = int(in_dim * n_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Text prompts for MedCLIP text encoder — 3 per class to match k_list=[3,3,3,3]
TISSUE_TEXT_PROMPTS = [
    # TUM (3 subclasses)
    "dense cluster of tumor epithelial cells in breast tissue",
    "scattered invasive tumor cells with high nuclear grade",
    "tumor cells forming glandular or cribriform structures",
    # STR (3 subclasses)
    "dense fibrous stromal tissue with abundant collagen bundles",
    "loose reactive stroma with fibroblasts surrounding tumor",
    "myxoid stromal tissue with sparse cellularity",
    # LYM (3 subclasses)
    "dense lymphocytic aggregate in tumor microenvironment",
    "scattered tumor-infiltrating lymphocytes between cancer cells",
    "lymphocyte-rich inflammatory region with plasma cells",
    # NEC (3 subclasses)
    "coagulative necrotic tissue with cell debris and ghost outlines",
    "early necrosis with pyknotic and karyorrhectic nuclei",
    "necrotic region with hemorrhagic extravasation",
]
# NOTE: BACK (background) is intentionally excluded from text prototypes.
# In WSSS, BG is computed implicitly as (1 - max_fg_cam)^10 in get_seg_label().
# Adding BG prototypes would require changing k_list, total_classes, and all merge functions.


class ClsNetwork(nn.Module):
    def __init__(self,
                 hf_token=None,
                 cls_num_classes=4,
                 fpn_out_dims=(64, 128, 320, 512),
                 n_ratio=0.5,
                 l_fea_path=None,
                 use_lora=True,
                 use_vq=False,
                 vq_num_codes=128,
                 vq_code_dim=256,
                 vq_commitment_weight=0.25,
                 vq_cam_weight=0.3,
                 use_multires_consistency=False,
                 use_text_proto=False):
        super().__init__()
        self.cls_num_classes = cls_num_classes
        self.in_channels = list(fpn_out_dims)  # match Segformer mit_b1 dims

        # Frozen UNI encoder (optionally with trainable LoRA)
        self.encoder = UNIEncoder(hf_token=hf_token, img_size=224, use_lora=use_lora)
        # Trainable Simple Feature Pyramid
        self.sfp = SimpleFeaturePyramid(in_dim=1024, out_dims=fpn_out_dims)

        self.pooling = F.adaptive_avg_pool2d

        ## medclip - AdaptiveLayer projects l_fea (512-dim) to match each scale
        self.l_fc1 = AdaptiveLayer(512, n_ratio, self.in_channels[0])  # 512 -> 64
        self.l_fc2 = AdaptiveLayer(512, n_ratio, self.in_channels[1])  # 512 -> 128
        self.l_fc3 = AdaptiveLayer(512, n_ratio, self.in_channels[2])  # 512 -> 320
        self.l_fc4 = AdaptiveLayer(512, n_ratio, self.in_channels[3])  # 512 -> 512

        with open(l_fea_path, "rb") as lf:
            info = pkl.load(lf)
            self.register_buffer('visual_proto', info['features'].cpu())  # [12, 512]
            self.k_list = info['k_list']
            self.cumsum_k = info['cumsum_k']

        self.total_classes = sum(self.k_list)

        # --- Hybrid Text-Visual Prototypes (Option C) ---
        self.use_text_proto = use_text_proto
        if self.use_text_proto:
            text_emb = self._encode_text_prompts(TISSUE_TEXT_PROMPTS)
            self.register_buffer('text_proto', text_emb)  # [12, 512]
            # Per-parent-class learnable blending: each of 4 classes gets its own α
            # sigmoid(raw) → α ∈ (0, 1), init raw=0.0 → sigmoid(0)=0.5
            num_parents = len(self.k_list)  # 4 (TUM, STR, LYM, NEC)
            self.proto_alpha_raw = nn.Parameter(torch.zeros(num_parents))  # [4]
            print(f"Hybrid text-visual prototypes enabled (per-class learnable alpha, init=0.5)")
            print(f"  Text prompts: {len(TISSUE_TEXT_PROMPTS)}, Visual protos: {self.visual_proto.shape[0]}")
            print(f"  Alpha shape: [{num_parents}] (one per parent class)")
            # --- Diagnostic: cross-modal alignment quality ---
            with torch.no_grad():
                text_emb_cpu = text_emb.cpu().float()
                vp_cpu = self.visual_proto.cpu().float()
                vp_norm = vp_cpu / (vp_cpu.norm(dim=-1, keepdim=True) + 1e-8)
                cross_sim = text_emb_cpu @ vp_norm.t()  # [12, 12]
                diag = cross_sim.diag()  # same-subclass similarity
                class_names = ["TUM"]*3 + ["STR"]*3 + ["LYM"]*3 + ["NEC"]*3
                print(f"  -- Cross-modal alignment (text<->visual cosine sim) --")
                for i, (name, sim_val) in enumerate(zip(class_names, diag)):
                    flag = "OK" if sim_val > 0.3 else "WARN" if sim_val > 0.1 else "BAD"
                    print(f"    [{flag}] {name}[{i%3}]: {sim_val:.4f}")
                print(f"  Mean same-subclass sim: {diag.mean():.4f}")
                text_sim = text_emb_cpu @ text_emb_cpu.t()  # [12, 12]
                off_diag = text_sim[~torch.eye(12, dtype=bool)]
                print(f"  Text-text off-diag sim: mean={off_diag.mean():.4f}, max={off_diag.max():.4f}")
                if off_diag.max() > 0.85:
                    print(f"  WARNING: Some text prompts are near-duplicates (max={off_diag.max():.4f})")
        
        # Backward compat: l_fea will be computed in forward()
        self.l_fea = self.visual_proto  # fallback if text_proto not used

        self.logit_scale1 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale2 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale3 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale4 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)

        # --- SISH-inspired: VQ Codebook Prototypes ---
        self.use_vq = use_vq
        self.vq_cam_weight = vq_cam_weight
        self.vq_warmup_epochs = 0  # text-guided init handles stability, no warmup needed
        self._current_epoch = 0
        _proto_dim = 512  # CONCH/UNI prototype space dimensionality
        if self.use_vq:
            self.vq_head2 = VQCAMHead(self.in_channels[1], self.total_classes,
                                     num_codes=vq_num_codes, code_dim=vq_code_dim,
                                     commitment_weight=vq_commitment_weight,
                                     proto_dim=_proto_dim)
            self.vq_head3 = VQCAMHead(self.in_channels[2], self.total_classes,
                                     num_codes=vq_num_codes, code_dim=vq_code_dim,
                                     commitment_weight=vq_commitment_weight,
                                     proto_dim=_proto_dim)
            self.vq_head4 = VQCAMHead(self.in_channels[3], self.total_classes,
                                     num_codes=vq_num_codes, code_dim=vq_code_dim,
                                     commitment_weight=vq_commitment_weight,
                                     proto_dim=_proto_dim)
            print(f"VQ Codebook enabled: {vq_num_codes} codes, dim={vq_code_dim}, cam_weight={vq_cam_weight}")
            print(f"  Semantic regularization: VQ codes → prototype space ({_proto_dim}d)")

            # --- Text-Guided VQ Initialization ---
            if self.use_text_proto:
                print(f"  Text-guided VQ init: seeding codebooks from {self.text_proto.shape[0]} text prototypes")
                self.vq_head2.init_codebook_from_prototypes(self.text_proto)
                self.vq_head3.init_codebook_from_prototypes(self.text_proto)
                self.vq_head4.init_codebook_from_prototypes(self.text_proto)
            print(f"  VQ warmup: weight=0 for first {self.vq_warmup_epochs} epoch(s)")

        # --- SISH-inspired: Multi-resolution Consistency ---
        self.use_multires_consistency = use_multires_consistency
        if self.use_multires_consistency:
            self.multires_loss_fn = MultiResConsistencyLoss()
            print("Multi-resolution consistency loss enabled")

    @staticmethod
    @torch.no_grad()
    def _encode_text_prompts(prompts):
        """Encode text prompts using CONCH text encoder → [N, 512] frozen embeddings.
        Falls back to MedCLIP if CONCH is not available."""
        if CONCH_AVAILABLE:
            print("  Using CONCH text encoder (histopathology-specialized)")
            conch_model, _ = create_model_from_pretrained(
                "conch_ViT-B-16", "hf_hub:MahmoodLab/conch"
            )
            conch_model.eval()
            if torch.cuda.is_available():
                conch_model = conch_model.cuda()
            tokenizer = get_tokenizer()
            # CONCH get_tokenizer() may return HuggingFace or open_clip tokenizer
            try:
                # Try open_clip style first
                tokens = tokenizer(prompts, context_length=77)
            except TypeError:
                # HuggingFace PreTrainedTokenizerFast
                inputs = tokenizer(prompts, max_length=77, truncation=True,
                                   padding=True, return_tensors="pt")
                tokens = inputs['input_ids']
            if torch.cuda.is_available():
                tokens = tokens.cuda()
            text_features = conch_model.encode_text(tokens)
            assert text_features.shape[-1] == 512, \
                f"CONCH output dim={text_features.shape[-1]}, expected 512. Update AdaptiveLayer!"
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            del conch_model, tokenizer
            print(f"  Encoded {len(prompts)} text prompts via CONCH → shape {text_features.shape}")
            return text_features.float().cpu()
        else:
            # Fallback to MedCLIP
            print("  Falling back to MedCLIP text encoder")
            processor = MedCLIPProcessor()
            clip_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
            clip_model.eval()
            if torch.cuda.is_available():
                clip_model = clip_model.cuda()
            inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
            text_features = clip_model.encode_text(inputs['input_ids'], inputs['attention_mask'])
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            del clip_model, processor
            print(f"  Encoded {len(prompts)} text prompts via MedCLIP → shape {text_features.shape}")
            return text_features.float().cpu()


    def get_param_groups(self):
        regularized = []
        not_regularized = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

    def forward(self, x):
        # --- UNI encoder (frozen weights + trainable LoRA) ---
        uni_feats = self.encoder(x)  # 4 x [B, 1024, 14, 14]

        # --- Trainable SFP ---
        _x = self.sfp(uni_feats)  # P1(56x56), P2(28x28), P3(14x14), P4(7x7)

        logit_scale1 = self.logit_scale1
        logit_scale2 = self.logit_scale2
        logit_scale3 = self.logit_scale3
        logit_scale4 = self.logit_scale4

        imshape = [f.shape for f in _x]
        image_features = [f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]) for f in _x]
        _x1, _x2, _x3, _x4 = image_features

        # --- Hybrid text-visual prototype blending ---
        if self.use_text_proto:
            # Per-parent-class alpha: [4] → expand to [12] via k_list
            parent_alpha = torch.sigmoid(self.proto_alpha_raw)  # [4], each ∈ (0, 1)
            # Expand: e.g. k_list=[3,3,3,3] → [α0,α0,α0, α1,α1,α1, α2,α2,α2, α3,α3,α3]
            alpha = torch.cat([a.expand(k) for a, k in zip(parent_alpha, self.k_list)])  # [12]
            alpha = alpha.unsqueeze(1)  # [12, 1] for broadcasting with [12, 512]
            l_fea = alpha * self.text_proto + (1 - alpha) * self.visual_proto
        else:
            l_fea = self.visual_proto
        l_fea = l_fea.to(x.device)
        l_fea1 = self.l_fc1(l_fea)
        l_fea2 = self.l_fc2(l_fea)
        l_fea3 = self.l_fc3(l_fea)
        l_fea4 = self.l_fc4(l_fea)

        total_vq_loss = torch.tensor(0.0, device=x.device)

        # VQ warmup: ramp up weight over warmup epochs
        if self.use_vq and self._current_epoch < self.vq_warmup_epochs:
            effective_vq_w = 0.0  # VQ off during warmup — let base model settle
        else:
            effective_vq_w = self.vq_cam_weight

        # --- Scale 1 (56x56) — no VQ ---
        _x1 = _x1 / _x1.norm(dim=-1, keepdim=True)
        logits_per_image1 = logit_scale1 * _x1 @ l_fea1.t().float()
        out1 = logits_per_image1.view(imshape[0][0], imshape[0][2], imshape[0][3], -1).permute(0, 3, 1, 2)
        cam1 = out1.clone().detach()
        cls1 = self.pooling(out1, (1, 1)).view(-1, self.total_classes)

        # --- Scale 2 (28x28) ---
        _x2 = _x2 / _x2.norm(dim=-1, keepdim=True)
        logits_per_image2 = logit_scale2 * _x2 @ l_fea2.t().float()
        out2 = logits_per_image2.view(imshape[1][0], imshape[1][2], imshape[1][3], -1).permute(0, 3, 1, 2)
        if self.use_vq:
            vq_cam2, vq_loss2 = self.vq_head2(_x2, imshape[1][2], imshape[1][3])
            total_vq_loss = total_vq_loss + vq_loss2
            out2 = (1 - effective_vq_w) * out2 + effective_vq_w * vq_cam2
        cam2 = out2.clone().detach()
        cls2 = self.pooling(out2, (1, 1)).view(-1, self.total_classes)

        # --- Scale 3 (14x14) ---
        _x3 = _x3 / _x3.norm(dim=-1, keepdim=True)
        logits_per_image3 = logit_scale3 * _x3 @ l_fea3.t().float()
        out3 = logits_per_image3.view(imshape[2][0], imshape[2][2], imshape[2][3], -1).permute(0, 3, 1, 2)
        if self.use_vq:
            vq_cam3, vq_loss3 = self.vq_head3(_x3, imshape[2][2], imshape[2][3])
            total_vq_loss = total_vq_loss + vq_loss3
            out3 = (1 - effective_vq_w) * out3 + effective_vq_w * vq_cam3
        cam3 = out3.clone().detach()
        cls3 = self.pooling(out3, (1, 1)).view(-1, self.total_classes)

        # --- Scale 4 (7x7) ---
        _x4 = _x4 / _x4.norm(dim=-1, keepdim=True)
        logits_per_image4 = logit_scale4 * _x4 @ l_fea4.t().float()
        out4 = logits_per_image4.view(imshape[3][0], imshape[3][2], imshape[3][3], -1).permute(0, 3, 1, 2)
        if self.use_vq:
            vq_cam4, vq_loss4 = self.vq_head4(_x4, imshape[3][2], imshape[3][3])
            total_vq_loss = total_vq_loss + vq_loss4
            out4 = (1 - effective_vq_w) * out4 + effective_vq_w * vq_cam4
        cam4 = out4.clone()
        cls4 = self.pooling(out4, (1, 1)).view(-1, self.total_classes)

        # --- Multi-resolution consistency loss (needs gradients from out, not cam) ---
        multires_loss = torch.tensor(0.0, device=x.device)
        if self.use_multires_consistency:
            multires_loss = (self.multires_loss_fn(out2, out3)
                           + self.multires_loss_fn(out3, out4))

        # --- Semantic regularization loss: link VQ codes to prototypes ---
        sem_reg_loss = torch.tensor(0.0, device=x.device)
        if self.use_vq and self.use_text_proto and self.training:
            sem_reg_loss = (self.vq_head2.compute_sem_reg_loss(l_fea)
                          + self.vq_head3.compute_sem_reg_loss(l_fea)
                          + self.vq_head4.compute_sem_reg_loss(l_fea)) / 3.0

        aux_losses = {'vq_loss': total_vq_loss, 'multires_loss': multires_loss,
                      'sem_reg_loss': sem_reg_loss}
        return cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, self.k_list, aux_losses

# ============================================================
# CONFIG (inline, no external file needed for Kaggle)
# ============================================================
CONFIG_YAML = """
model:
  hf_token: REMOVED_HF_TOKEN
  fpn_out_dims:
  - 64
  - 128
  - 320
  - 512
  label_feature_path: /kaggle/input/datasets/dongnguyenduy/pbip-proto/bcss_label_fea_pro_3333.pkl
  n_ratio: 0.5
  use_lora: false
dataset:
  name: bcss
  train_root: /kaggle/working/BCSS-WSSS/training
  val_root: /kaggle/working/BCSS-WSSS
  test_root: /kaggle/working/BCSS-WSSS
  cls_num_classes: 4
  input_size:
  - 224
  - 224
work_dir:
  ckpt_dir: /kaggle/working/checkpoints
  pred_dir: /kaggle/working/predictions
  train_log_dir: /kaggle/working/train_log
train:
  samples_per_gpu: 4
  accum_steps: 4
  epoch: 8
  l1: 0.0
  l2: 0.1
  l3: 1.0
  l4: 1.0
  l5: 0.05
  mask_adapter_alpha: 0.5
  merge_train: mean
  merge_test: max
  use_tta: false
  # === SISH-Inspired Features ===
  use_vq: true
  vq_num_codes: 128
  vq_code_dim: 256
  vq_commitment_weight: 0.15
  vq_cam_weight: 0.3
  vq_loss_weight: 0.1
  sem_reg_weight: 0.03
  use_multires_consistency: true
  multires_weight: 0.05
  use_binary_hash: true
  hash_dim: 128
  hash_top_k: 16
  hash_quant_weight: 0.01
  hash_bank_capacity: 4096
  # === CRF Post-Processing ===
  use_crf: false
  crf_iter_max: 10
  crf_pos_w: 3
  crf_pos_xy_std: 1
  crf_bi_w: 4
  crf_bi_xy_std: 67
  crf_bi_rgb_std: 3
  # === Hybrid Text-Visual Prototypes ===
  use_text_proto: true
  # === Early Stopping ===
  use_early_stop: true
  patience: 2   # Số lần evaluation không tăng điểm liên tiếp trước khi dừng
optimizer:
  type: AdamW
  learning_rate: 0.00005
  betas:
  - 0.9
  - 0.999
  weight_decay: 0.001
scheduler:
  warmup_iter: 0
  warmup_ratio: 1.0e-06
  power: 1.0
"""

# ============================================================
# FEATURE PATH FOR IMAGE FEATURES
# (used separately in FeatureExtractor / data scripts)
# ============================================================
BCSS_FEATURES_PATH  = "/kaggle/input/datasets/dongnguyenduy/pbip-proto/bcss_features_pro.pkl"
BCSS_LABEL_FEA_PATH = "/kaggle/input/datasets/dongnguyenduy/pbip-proto/bcss_label_fea_pro_3333.pkl"

# ============================================================
# GPU selection (change if needed)
# ============================================================
GPU_ID = 0

start_time = datetime.datetime.now()

# ============================================================
# FRAMEWORK IMAGE EXPORT (for paper figure)
# ============================================================

CLASS_COLORS = np.array([
    [255, 0, 0],     # TUM - red
    [0, 255, 0],     # STR - green
    [0, 0, 255],     # LYM - blue
    [128, 0, 128],   # NEC - purple
    [255, 255, 255], # BACK - white
], dtype=np.uint8)

def save_cam_heatmap(cam_tensor, save_path, size=224, interpolation=cv2.INTER_LINEAR):
    """Save single-channel CAM as jet colormap image.
    Use interpolation=cv2.INTER_NEAREST for VQ CAM to preserve blocky/quantized look.
    """
    cam = cam_tensor.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_resized = cv2.resize(cam, (size, size), interpolation=interpolation)
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(save_path, heatmap)

def save_seg_mask(seg_tensor, save_path, size=224):
    """Save segmentation mask as colored image."""
    seg = seg_tensor.cpu().numpy().astype(np.uint8)
    seg_resized = cv2.resize(seg, (size, size), interpolation=cv2.INTER_NEAREST)
    colored = CLASS_COLORS[seg_resized]  # [H, W, 3]
    cv2.imwrite(save_path, colored[:, :, ::-1])  # RGB→BGR

@torch.no_grad()
def export_framework_images(model, data_loader, cfg, save_dir='/kaggle/working/framework_figs'):
    """Export intermediate outputs for framework diagram.
    Chỉ dùng ảnh chỉ định từ tập TEST (data_loader phải là test_loader).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    MEAN = np.array([0.66791496, 0.47791372, 0.70623304])
    STD = np.array([0.1736589, 0.22564577, 0.19820057])

    TARGET_IMAGE = "TCGA-EW-A1P7-DX1_xmin9808_ymin46395_MPP-0.2500+52.png"
    best_data = None

    print(f"\n[Framework Export] Looking for image: {TARGET_IMAGE}")
    for data in data_loader:
        name, inputs, cls_label, labels = data
        inputs = inputs.cuda().float()
        labels_np = labels.numpy()  # [B, H, W]
        cls_label = cls_label.cuda().float()

        # Forward
        cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list, aux = model(inputs)

        cam1_p = merge_subclass_cams_to_parent(cam1, k_list, method='max')
        cam2_p = merge_subclass_cams_to_parent(cam2, k_list, method='max')
        cam3_p = merge_subclass_cams_to_parent(cam3, k_list, method='max')
        cam4_p = merge_subclass_cams_to_parent(cam4, k_list, method='max')

        cams1 = get_seg_label(cam1_p, inputs, cls_label).cuda()
        cams2 = get_seg_label(cam2_p, inputs, cls_label).cuda()
        cams3 = get_seg_label(cam3_p, inputs, cls_label).cuda()
        cams4 = get_seg_label(cam4_p, inputs, cls_label).cuda()
        fuse = 0.3 * cams2 + 0.3 * cams3 + 0.4 * cams4

        for b in range(inputs.shape[0]):
            # Chỉ lấy đúng ảnh chỉ định (so khớp tên file)
            n = name[b] if isinstance(name[b], str) else name[b][0]
            if n != TARGET_IMAGE and not n.endswith(TARGET_IMAGE):
                continue
            gt = labels_np[b]
            pred = torch.argmax(fuse[b], dim=0).cpu().numpy()
            unique_classes = np.unique(gt)
            fg_classes = [c for c in unique_classes if c < 4]
            best_data = {
                'name': n,
                'input': inputs[b].cpu(),
                'cls_label': cls_label[b].cpu(),
                'gt': gt,
                'pred': pred,
                'cam1_p': cam1_p[b].cpu(),
                'cam2_p': cam2_p[b].cpu(),
                'cam3_p': cam3_p[b].cpu(),
                'cam4_p': cam4_p[b].cpu(),
                'cam1_sub': cam1[b].cpu(),
                'cam4_sub': cam4[b].cpu(),
                'fuse': fuse[b].cpu(),
                'fg_classes': fg_classes,
            }
            break
        if best_data is not None:
            break

    if best_data is None:
        print(f"[Framework Export] Image '{TARGET_IMAGE}' not found in data loader, skipping.")
        model.train()
        return

    d = best_data
    # Dùng Stroma (STR, class 1) thay vì tumor cho CAM
    dom_cls = 1  # STR
    print(f"[Framework Export] Using image: {d['name']} (CAM class: Stroma/STR)")

    # 1. Input image (denormalize)
    img = d['input'].permute(1, 2, 0).numpy()  # [H, W, 3]
    img = (img * STD + MEAN) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, '01_input.png'), img[:, :, ::-1])

    # 2. CAM heatmaps per scale (use Stroma / dom_cls=1)
    for i, (cam_p, scale) in enumerate([
        (d['cam1_p'], '56x56'), (d['cam2_p'], '28x28'),
        (d['cam3_p'], '14x14'), (d['cam4_p'], '7x7')]):
        save_cam_heatmap(cam_p[dom_cls], os.path.join(save_dir, f'02_cam_scale{i+1}_{scale}.png'))

    # 3. Per-class CAM heatmaps (scale 4 = coarsest)
    class_names = ['TUM', 'STR', 'LYM', 'NEC']
    for c in range(4):
        save_cam_heatmap(d['cam4_p'][c], os.path.join(save_dir, f'03_cam4_{class_names[c]}.png'))

    # 4. Fused pseudo segmentation
    save_seg_mask(torch.tensor(d['pred']), os.path.join(save_dir, '04_pseudo_seg.png'))

    # 5. Ground truth
    save_seg_mask(torch.tensor(d['gt']), os.path.join(save_dir, '05_ground_truth.png'))

    # 6. FG/BG masked images (original × mask, like paper figures)
    cam4_max = d['cam4_p'].max(0)[0]  # [H, W]
    cam4_224 = F.interpolate(cam4_max.unsqueeze(0).unsqueeze(0), (224, 224), mode='bilinear')[0, 0]
    thresh = cam4_224.max() * 0.5
    fg_mask = (cam4_224 > thresh).float().numpy()
    bg_mask = 1 - fg_mask
    # FG: original image × foreground mask (black bg)
    fg_img = img * fg_mask[:, :, None]  # [H, W, 3]
    bg_img = img * bg_mask[:, :, None]  # [H, W, 3]
    cv2.imwrite(os.path.join(save_dir, '06_fg_masked.png'), fg_img[:, :, ::-1].astype(np.uint8))
    cv2.imwrite(os.path.join(save_dir, '06_bg_masked.png'), bg_img[:, :, ::-1].astype(np.uint8))
    # Also save binary masks for reference
    cv2.imwrite(os.path.join(save_dir, '06_fg_mask_binary.png'), (fg_mask * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(save_dir, '06_bg_mask_binary.png'), (bg_mask * 255).astype(np.uint8))

    # 7. Overlay CAM on input
    cam_overlay = d['cam4_p'].max(0)[0].numpy()
    cam_overlay = (cam_overlay - cam_overlay.min()) / (cam_overlay.max() - cam_overlay.min() + 1e-8)
    cam_resized = cv2.resize(cam_overlay, (224, 224))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img[:, :, ::-1], 0.5, heatmap, 0.5, 0)
    cv2.imwrite(os.path.join(save_dir, '07_cam_overlay.png'), overlay)

    # 8. Legend
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    for c, name in enumerate(class_names + ['BACK']):
        ax.barh(c, 1, color=CLASS_COLORS[c] / 255.0, edgecolor='black')
        ax.text(0.5, c, name, ha='center', va='center', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1); ax.axis('off')
    fig.savefig(os.path.join(save_dir, '08_legend.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 9. VQ Codebook Detail (for VQ detail figure)
    if hasattr(model, 'use_vq') and model.use_vq and hasattr(model, 'vq_head3'):
        print("[Framework Export] Exporting VQ codebook details...")
        vq_dir = os.path.join(save_dir, 'vq_detail')
        os.makedirs(vq_dir, exist_ok=True)

        # Re-forward the best image to get P3 features (14x14)
        single_input = d['input'].unsqueeze(0).cuda().float()
        with torch.no_grad():
            # Get intermediate features from model
            cls1, c1, cls2, c2, cls3, c3, cls4, c4, lf, kl, _ = model(single_input)

        # Run VQ head3 internals manually on P3 features (14x14)
        vq_head = model.vq_head3
        # Extract P3 from SFP by re-running encoder
        with torch.no_grad():
            feats = model.encoder(single_input)
            p1, p2, p3, p4 = model.sfp(feats)
            B, C, H3, W3 = p3.shape  # 14x14
            _x3 = p3.permute(0, 2, 3, 1).reshape(B * H3 * W3, C)  # [B*196, 320]
            _x3 = _x3 / _x3.norm(dim=-1, keepdim=True)

            # Step 1: Project (320 → 256)
            z_projected = vq_head.project(_x3)  # [196, 256]

            # Step 2: VQ lookup (manually to get indices)
            z_norm = F.normalize(z_projected, dim=-1)
            cb_norm = F.normalize(vq_head.vq.codebook, dim=-1)
            cosine_sim = z_norm @ cb_norm.t()  # [196, 128]
            indices = torch.argmax(cosine_sim, dim=1)  # [196]
            z_q = vq_head.vq.codebook[indices]  # [196, 256]

            # Step 3: CAM head on quantized features
            z_q_spatial = z_q.view(1, H3, W3, -1).permute(0, 3, 1, 2)  # [1, 256, 14, 14]
            vq_cam_out = vq_head.cam_head(z_q_spatial)  # [1, 12, 14, 14]

        # --- Save VQ detail images ---

        # 9a. Pixel features (P3) — mean over channels as heatmap
        # Giữ nguyên nội suy mặc định (LINEAR) -> Hình mượt, liên tục
        p3_vis = p3[0].mean(0).cpu()  # [14, 14]
        save_cam_heatmap(p3_vis, os.path.join(vq_dir, '09a_pixel_features_P3.png'))

        # 9b. Projected features (after MLP) — mean over channels
        # Giữ nguyên nội suy mặc định (LINEAR) -> Hình mượt, liên tục
        z_proj_vis = z_projected.view(H3, W3, -1).mean(-1).cpu()  # [14, 14]
        save_cam_heatmap(z_proj_vis, os.path.join(vq_dir, '09b_projected_features.png'))

        # 9c. Discrete code indices — grid of numbers
        indices_grid = indices.view(H3, W3).cpu().numpy()  # [14, 14]
        fig_idx, ax_idx = plt.subplots(1, 1, figsize=(5, 5))
        ax_idx.imshow(indices_grid, cmap='tab20', interpolation='nearest')
        for i in range(H3):
            for j in range(W3):
                ax_idx.text(j, i, str(indices_grid[i, j]), ha='center', va='center',
                           fontsize=5, color='white', fontweight='bold')
        ax_idx.set_title(f'Discrete Codes ({H3}x{W3})', fontsize=10)
        ax_idx.axis('off')
        fig_idx.savefig(os.path.join(vq_dir, '09c_discrete_codes.png'), dpi=200, bbox_inches='tight')
        plt.close(fig_idx)

        # 9d. Quantized features — mean over channels
        # UPDATE (đã có): Ép dùng NEAREST -> Hình khối vuông (chứng minh đã lượng tử hóa)
        z_q_vis = z_q.view(H3, W3, -1).mean(-1).cpu()  # [14, 14]
        save_cam_heatmap(z_q_vis, os.path.join(vq_dir, '09d_quantized_features.png'), 
                         interpolation=cv2.INTER_NEAREST)

        # 9e. VQ CAM — per class heatmap (parent classes)
        # UPDATE (đã có): Ép dùng NEAREST -> VQ CAM khối vuông
        vq_cam_parent = merge_subclass_cams_to_parent(vq_cam_out, kl, method='max')  # [1, 4, 14, 14]
        for c in range(4):
            save_cam_heatmap(vq_cam_parent[0, c].cpu(),
                           os.path.join(vq_dir, f'09e_vq_cam_{class_names[c]}.png'),
                           interpolation=cv2.INTER_NEAREST)
        # Combined VQ CAM (max across classes) — blocky
        vq_cam_max = vq_cam_parent[0].max(0)[0].cpu()
        save_cam_heatmap(vq_cam_max, os.path.join(vq_dir, '09e_vq_cam_combined.png'),
                         interpolation=cv2.INTER_NEAREST)

        # 9f. Codebook usage histogram
        unique_codes, counts = torch.unique(indices, return_counts=True)
        fig_hist, ax_hist = plt.subplots(1, 1, figsize=(8, 3))
        ax_hist.bar(unique_codes.cpu().numpy(), counts.cpu().numpy(), color='#E67E22')
        ax_hist.set_xlabel('Code Index')
        ax_hist.set_ylabel('Usage Count')
        ax_hist.set_title(f'Codebook Usage ({len(unique_codes)}/{vq_head.vq.num_codes} codes active)')
        fig_hist.savefig(os.path.join(vq_dir, '09f_codebook_usage.png'), dpi=150, bbox_inches='tight')
        plt.close(fig_hist)

        # 9g. Export VQ CAM for ALL 3 heads (for overall framework diagram)
        vq_heads_info = [
            ('vq_head2', p2, 'P2'),
            ('vq_head3', p3, 'P3'),
            ('vq_head4', p4, 'P4'),
        ]
        for head_name, p_feat, p_label in vq_heads_info:
            vq_h = getattr(model, head_name)
            B_h, C_h, H_h, W_h = p_feat.shape
            _xh = p_feat.permute(0, 2, 3, 1).reshape(B_h * H_h * W_h, C_h)
            _xh = _xh / _xh.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                z_h = vq_h.project(_xh)
                z_h_norm = F.normalize(z_h, dim=-1)
                cb_h_norm = F.normalize(vq_h.vq.codebook, dim=-1)
                idx_h = torch.argmax(z_h_norm @ cb_h_norm.t(), dim=1)
                zq_h = vq_h.vq.codebook[idx_h]
                zq_spatial = zq_h.view(1, H_h, W_h, -1).permute(0, 3, 1, 2)
                vq_cam_h = vq_h.cam_head(zq_spatial)
            vq_cam_h_parent = merge_subclass_cams_to_parent(vq_cam_h, kl, method='max')
            vq_cam_h_max = vq_cam_h_parent[0].max(0)[0].cpu()
            
            # UPDATE (đã có): Ép dùng NEAREST
            save_cam_heatmap(vq_cam_h_max, os.path.join(vq_dir, f'09g_vq_cam_{p_label}_combined.png'),
                             interpolation=cv2.INTER_NEAREST)
            # Per-class VQ CAM — blocky
            for c in range(4):
                save_cam_heatmap(vq_cam_h_parent[0, c].cpu(),
                               os.path.join(vq_dir, f'09g_vq_cam_{p_label}_{class_names[c]}.png'),
                               interpolation=cv2.INTER_NEAREST)
            print(f"  VQ CAM {p_label} ({H_h}x{W_h}): saved")

        # ------------------------------------------------------------------------------------
        # 10. BỔ SUNG: Trích xuất trọn bộ Base CAM, VQ CAM và Blended CAM cho sơ đồ Draw.io
        # ------------------------------------------------------------------------------------
        print("[Framework Export] Generating Base, VQ, and Blended CAMs for diagram...")

        # Khôi phục l_fea (Hybrid Text-Visual Prototypes)
        if model.use_text_proto:
            parent_alpha = torch.sigmoid(model.proto_alpha_raw)
            alpha = torch.cat([a.expand(k) for a, k in zip(parent_alpha, model.k_list)]).unsqueeze(1).cuda()
            l_fea_current = alpha * model.text_proto.cuda() + (1 - alpha) * model.visual_proto.cuda()
        else:
            l_fea_current = model.visual_proto.cuda()

        # Định nghĩa các scale: scale1 chỉ có Base CAM (cosine), scale2/3/4 có Base + VQ + Blended
        scales_diagram_info = [
            ('scale1', p1, model.l_fc1, model.logit_scale1, None),   # no VQ at scale1
            ('scale2', p2, model.l_fc2, model.logit_scale2, model.vq_head2),
            ('scale3', p3, model.l_fc3, model.logit_scale3, model.vq_head3),
            ('scale4', p4, model.l_fc4, model.logit_scale4, model.vq_head4),
        ]

        # Chỉ lấy class nổi bật nhất để vẽ
        cls_name = class_names[dom_cls]

        for scale_name, p_feat, l_fc, logit_scale, vq_h in scales_diagram_info:
            B_h, C_h, H_h, W_h = p_feat.shape
            _xh = p_feat.permute(0, 2, 3, 1).reshape(B_h * H_h * W_h, C_h)
            _xh = _xh / _xh.norm(dim=-1, keepdim=True)

            # --- 10.1 Tính toán BASE CAM (cosine similarity: logit_scale * x @ proto.t()) ---
            l_fea_h = l_fc(l_fea_current)
            logits_h = logit_scale * _xh @ l_fea_h.t().float()
            base_cam = logits_h.view(B_h, H_h, W_h, -1).permute(0, 3, 1, 2)  # [B, 12, H, W]
            base_cam_parent = merge_subclass_cams_to_parent(base_cam, kl, method='max')

            save_cam_heatmap(base_cam_parent[0, dom_cls].cpu(),
                             os.path.join(vq_dir, f'10_{scale_name}_1_BaseCAM_{cls_name}.png'))

            if vq_h is not None:
                # --- 10.2 Tính toán VQ CAM ---
                with torch.no_grad():
                    z_h = vq_h.project(_xh)
                    z_h_norm = F.normalize(z_h, dim=-1)
                    cb_h_norm = F.normalize(vq_h.vq.codebook, dim=-1)
                    idx_h = torch.argmax(z_h_norm @ cb_h_norm.t(), dim=1)
                    zq_h = vq_h.vq.codebook[idx_h]
                    zq_spatial = zq_h.view(1, H_h, W_h, -1).permute(0, 3, 1, 2)
                    vq_cam_h = vq_h.cam_head(zq_spatial)  # [1, 12, H, W]

                # --- 10.3 BLENDED CAM: phải blend ở subclass rồi mới merge (giống model) thì mới khớp 02_cam_scale* ---
                eff_vq_w = model.vq_cam_weight if model._current_epoch >= model.vq_warmup_epochs else 0.0
                blended_cam = (1 - eff_vq_w) * base_cam + eff_vq_w * vq_cam_h  # [B, 12, H, W]
                blended_cam_parent = merge_subclass_cams_to_parent(blended_cam, kl, method='max')
                vq_cam_parent = merge_subclass_cams_to_parent(vq_cam_h, kl, method='max')

                # VQ CAM: resize bằng INTER_NEAREST để giữ nét blocky/quantized như trong diagram
                save_cam_heatmap(vq_cam_parent[0, dom_cls].cpu(),
                                 os.path.join(vq_dir, f'10_{scale_name}_2_VQCAM_{cls_name}.png'),
                                 interpolation=cv2.INTER_NEAREST)
                save_cam_heatmap(blended_cam_parent[0, dom_cls].cpu(),
                                 os.path.join(vq_dir, f'10_{scale_name}_3_BlendedCAM_{cls_name}.png'))
                print(f"  -> Saved Base, VQ, and Blended CAM for {scale_name} (Class: {cls_name})")
            else:
                # Scale1: no VQ, only Base CAM (đã lưu ở trên)
                print(f"  -> Saved Base CAM for {scale_name} (Class: {cls_name}, no VQ)")
        # ------------------------------------------------------------------------------------

        print(f"[Framework Export] VQ detail: {len(os.listdir(vq_dir))} images saved to {vq_dir}")

    print(f"[Framework Export] Saved {len(os.listdir(save_dir))} images to {save_dir}")
    model.train()


# ============================================================
# TRAINING
# ============================================================

def cal_eta(time0, cur_iter, total_iter):
    """Calculate elapsed time and estimated time to completion"""
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

def train(cfg):
    
    print("\nInitializing training...")
    torch.backends.cudnn.benchmark = True  # Enable cudnn auto optimization
    
    num_workers = min(10, os.cpu_count())  # Optimize worker count based on CPU cores
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)
    
    # Preload CLIP model to GPU and set to eval mode
    clip_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    # ============ prepare data ============
    print("\nPreparing datasets...")
    train_dataset, val_dataset = get_cls_dataset(cfg, split="val", val_root=cfg.dataset.val_root)
    _, test_dataset = get_cls_dataset(cfg, split="test", val_root=cfg.dataset.test_root, enable_rotation=False, p=0.0)

    # Efficient data loading configuration
    train_loader = DataLoader(train_dataset,
                            batch_size=cfg.train.samples_per_gpu,
                            num_workers=num_workers,
                            pin_memory=True,
                            shuffle=True,
                            prefetch_factor=2,
                            persistent_workers=True)

    val_loader = DataLoader(val_dataset,
                          batch_size=8,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True,
                          persistent_workers=True)

    test_loader = DataLoader(test_dataset,
                           batch_size=8,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=True,
                           persistent_workers=True)
    print(f"✓ Val dataset: {len(val_dataset)} samples | Test dataset: {len(test_dataset)} samples")

    # ============ prepare model ============
    model = ClsNetwork(
                    hf_token=cfg.model.hf_token,
                    cls_num_classes=cfg.dataset.cls_num_classes,
                    fpn_out_dims=tuple(cfg.model.fpn_out_dims),
                    n_ratio=cfg.model.n_ratio,
                    l_fea_path=cfg.model.label_feature_path,
                    use_lora=cfg.model.use_lora,
                    use_vq=cfg.train.use_vq,
                    vq_num_codes=cfg.train.vq_num_codes,
                    vq_code_dim=cfg.train.vq_code_dim,
                    vq_commitment_weight=cfg.train.vq_commitment_weight,
                    vq_cam_weight=cfg.train.vq_cam_weight,
                    use_multires_consistency=cfg.train.use_multires_consistency,
                    use_text_proto=getattr(cfg.train, 'use_text_proto', False))
    
    # Print trainable param count (LoRA + SFP + heads)
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_p/1e6:.1f}M total, {train_p/1e6:.2f}M trainable ({train_p/total_p*100:.2f}%)")
    # Mixed precision training setup
    iters_per_epoch = len(train_loader)
    accum_steps = getattr(cfg.train, 'accum_steps', 1)
    # Each outer iteration consumes accum_steps micro-batches, so divide accordingly
    steps_per_epoch = max(1, iters_per_epoch // accum_steps)
    cfg.train.max_iters = cfg.train.epoch * steps_per_epoch
    cfg.train.eval_iters = steps_per_epoch
    cfg.scheduler.warmup_iter = cfg.scheduler.warmup_iter * steps_per_epoch
    scaler = torch.amp.GradScaler('cuda')
    
    model.to(device)
    model.train()
    model.encoder.model.eval()  # keep frozen UNI in eval (disable Dropout); LoRA still trains

    # Optimizer configuration
    optimizer = PolyWarmupAdamW(
        params=model.parameters(),
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )

    # Loss functions and feature extractor setup
    loss_function = nn.BCEWithLogitsLoss().to(device)
    mask_adapter = MaskAdapter_DynamicThreshold(alpha=cfg.train.mask_adapter_alpha,)
    feature_extractor = FeatureExtractor(mask_adapter=mask_adapter)
    fg_loss_fn = InfoNCELossFG(temperature=0.07).to(device)
    bg_loss_fn = InfoNCELossBG(temperature=0.07).to(device)
    # Hash-guided hard negative mining (additive to InfoNCE)
    hash_miner = None
    if cfg.train.use_binary_hash:
        hash_miner = HashHardNegativeMiner(
            in_dim=512, hash_dim=cfg.train.hash_dim,
            top_k=cfg.train.hash_top_k, quant_weight=cfg.train.hash_quant_weight,
            bank_capacity=cfg.train.hash_bank_capacity
        ).to(device)
    best_fuse234_dice = 0.0
    
    # --- Early Stopping Variables ---
    epochs_no_improve = 0
    patience = getattr(cfg.train, 'patience', 3)
    use_early_stop = getattr(cfg.train, 'use_early_stop', False)

    print("\nStarting training...")
    train_loader_iter = iter(train_loader)
    
    accum_steps = getattr(cfg.train, 'accum_steps', 1)
    
    for n_iter in range(cfg.train.max_iters):
        optimizer.zero_grad(set_to_none=True)
        
        total_loss_for_log = 0.0  # accumulate unscaled loss for logging

        for _micro in range(accum_steps):
            try:
                img_name, inputs, cls_labels, gt_label = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                img_name, inputs, cls_labels, gt_label = next(train_loader_iter)

            inputs = inputs.to(device).float()
            cls_labels = cls_labels.to(device).float()

            with torch.amp.autocast('cuda'):
                cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list, aux_losses = model(inputs)

                # Merge subclass predictions to parent class predictions
                cls1_merge = merge_to_parent_predictions(cls1, k_list, method=cfg.train.merge_train)
                cls2_merge = merge_to_parent_predictions(cls2, k_list, method=cfg.train.merge_train)
                cls3_merge = merge_to_parent_predictions(cls3, k_list, method=cfg.train.merge_train)
                cls4_merge = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_train)

                # Generate binary masks for feature extraction
                subclass_labels = expand_parent_to_subclass_labels(cls_labels, k_list)
                cls4_expand=expand_parent_to_subclass_labels(cls4_merge, k_list)
                cls4_bir=(cls4>cls4_expand).float()*subclass_labels

                # Extract foreground and background features
                batch_info = feature_extractor.process_batch(inputs, cam4, cls4_bir, clip_model)
                if batch_info is not None:
                    fg_features, bg_features = batch_info['fg_features'], batch_info['bg_features']
                    set_info = pair_features(fg_features, bg_features, l_fea, cls4_bir)
                    fg_features, bg_features = set_info['fg_features'], set_info['bg_features']
                    fg_pro, bg_pro = set_info['fg_text'], set_info['bg_text']

                    # InfoNCE always runs (never replaced)
                    fg_loss = fg_loss_fn(fg_features, fg_pro, bg_pro)
                    bg_loss = bg_loss_fn(bg_features, fg_pro, bg_pro)

                    # Hash: ADDITIVE mining loss on top of InfoNCE
                    if cfg.train.use_binary_hash and hash_miner is not None:
                        fg_labels = set_info['labels']
                        cumsum_k = torch.tensor(k_list, device=device).cumsum(0)
                        fg_parent_labels = torch.bucketize(fg_labels, cumsum_k)
                        hash_loss = hash_miner(fg_features, fg_parent_labels)
                    else:
                        hash_loss = torch.tensor(0.0, device=device)
                else:
                    fg_loss = torch.tensor(0.0, device=device)
                    bg_loss = torch.tensor(0.0, device=device)
                    hash_loss = torch.tensor(0.0, device=device)

                # Multi-scale classification losses
                loss1 = loss_function(cls1_merge, cls_labels)
                loss2 = loss_function(cls2_merge, cls_labels)
                loss3 = loss_function(cls3_merge, cls_labels)
                loss4 = loss_function(cls4_merge, cls_labels)
                cls_loss = cfg.train.l1 * loss1 + cfg.train.l2 * loss2 + cfg.train.l3 * loss3 + cfg.train.l4 * loss4
                
                # Raw loss (for logging) and scaled loss (for backward)
                _use_nce = getattr(cfg.train, 'use_contrastive', True)
                _nce_w = 1.0 if _use_nce else 0.0
                raw_loss = cls_loss + (_nce_w * fg_loss + _nce_w * bg_loss + 0.0005*torch.mean(cam4)) * cfg.train.l5

                if cfg.train.use_vq:
                    raw_loss = raw_loss + cfg.train.vq_loss_weight * aux_losses['vq_loss']
                if cfg.train.use_multires_consistency:
                    raw_loss = raw_loss + cfg.train.multires_weight * aux_losses['multires_loss']
                if cfg.train.use_binary_hash:
                    raw_loss = raw_loss + cfg.train.l5 * hash_loss
                sem_reg_w = getattr(cfg.train, 'sem_reg_weight', 0.0)
                if sem_reg_w > 0:
                    raw_loss = raw_loss + sem_reg_w * aux_losses['sem_reg_loss']
                scaled_loss = raw_loss / accum_steps

            # Gradient scaling for mixed precision
            scaler.scale(scaled_loss).backward()
            
            # Accumulate unscaled loss for accurate logging
            total_loss_for_log += raw_loss.item() / accum_steps

        scaler.step(optimizer)
        scaler.update()

        if (n_iter + 1) % 100 == 0:
            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']
            
            # Synchronize for accurate timing measurement
            torch.cuda.synchronize()
            
            cls_pred4 = (torch.sigmoid(cls4_merge) > 0.5).float()
            all_cls_acc4 = (cls_pred4 == cls_labels).all(dim=1).float().mean() * 100
            avg_cls_acc4 = ((cls_pred4 == cls_labels).float().mean(dim=0)).mean() * 100
            
            print(
                f"Iter: {n_iter + 1}/{cfg.train.max_iters}; "
                f"Elapsed: {delta}; ETA: {eta}; "
                f"LR: {cur_lr:.3e}; Loss: {total_loss_for_log:.4f}; "
                f"Acc4: {all_cls_acc4:.2f}/{avg_cls_acc4:.2f}"
            )
        # Regular validation and model saving
        if (n_iter + 1) % cfg.train.eval_iters == 0 or (n_iter + 1) == cfg.train.max_iters:
            epoch_num = (n_iter + 1) // cfg.train.eval_iters
            model._current_epoch = epoch_num  # update for VQ warmup schedule
            print(f"\n{'='*60}")
            print(f"Epoch {epoch_num} Evaluation")
            print(f"{'='*60}")

            # --- Val set ---
            val_all_acc4, val_avg_acc4, val_metrics, val_cls_loss = validate(
                model=model, data_loader=val_loader, cfg=cfg, cls_loss_func=loss_function
            )
            print(f"[VAL]  mIoU: {val_metrics['miou']*100:.2f}% | fwIoU: {val_metrics['fwiou']*100:.2f}% | "
                  f"BdIoU: {val_metrics['boundary_iou']*100:.2f}% | Dice: {val_metrics['dice']*100:.2f}%")
            print(f"       per-class IoU: {[f'{s*100:.2f}' for s in val_metrics['iou'][:-1].tolist()]}")

            # --- Test set ---
            test_all_acc4, test_avg_acc4, test_metrics, test_cls_loss = validate(
                model=model, data_loader=test_loader, cfg=cfg, cls_loss_func=loss_function
            )
            print(f"[TEST] mIoU: {test_metrics['miou']*100:.2f}% | fwIoU: {test_metrics['fwiou']*100:.2f}% | "
                  f"BdIoU: {test_metrics['boundary_iou']*100:.2f}% | Dice: {test_metrics['dice']*100:.2f}%")
            print(f"       per-class IoU: {[f'{s*100:.2f}' for s in test_metrics['iou'][:-1].tolist()]}")

            # Save best model based on val mIOU
            current_val_score = val_metrics['miou']
            if current_val_score > best_fuse234_dice:
                best_fuse234_dice = current_val_score
                epochs_no_improve = 0
                
                save_path = os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth")
                torch.save(
                    {
                        "cfg": cfg,
                        "iter": n_iter,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    },
                    save_path,
                    _use_new_zipfile_serialization=True
                )
                print(f"Saved best model (val mIOU: {best_fuse234_dice:.4f})")
            else:
                epochs_no_improve += 1
                print(f"No improvement in val mIOU for {epochs_no_improve} eval(s). Best remains: {best_fuse234_dice:.4f}")

            if use_early_stop and epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {patience} evaluations without improvement.")
                break

            # Export framework images after epoch 1 (dùng tập TEST)
            if epoch_num == 1:
                export_framework_images(model, test_loader, cfg)

    torch.cuda.empty_cache()
    end_time = datetime.datetime.now()
    total_training_time = end_time - start_time
    print(f'Total training time: {total_training_time}')

    
    print("\n" + "="*80)
    print("POST-TRAINING EVALUATION AND CAM GENERATION")
    print("="*80)

    # ---------- Evaluate on val set ----------
    print("\n1. Evaluating on VAL set...")
    print("-" * 50)
    _, val_dataset_eval = get_cls_dataset(cfg, split="val", val_root=cfg.dataset.val_root, enable_rotation=False, p=0.0)
    val_eval_loader = DataLoader(val_dataset_eval, batch_size=8, shuffle=False,
                                 num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_all_acc4_final, val_avg_acc4_final, val_fuse234_score, val_cls_loss_final = validate(
        model=model, data_loader=val_eval_loader, cfg=cfg, cls_loss_func=loss_function)
    print("Val results:")
    print(f"  all_acc4: {val_all_acc4_final:.6f} | avg_acc4: {val_avg_acc4_final:.6f}")
    print(f"  mIOU: {val_fuse234_score[:-1].mean():.4f} | scores: {val_fuse234_score}")
    print("  Per-class IoU:")
    for i, score in enumerate(val_fuse234_score[:-1]):
        print(f"    Class {i}: {score:.6f}")
    if len(val_fuse234_score) > cfg.dataset.cls_num_classes:
        print(f"    Background: {val_fuse234_score[-1]:.6f}")

    # ---------- Evaluate on test set ----------
    print("\n2. Evaluating on TEST set...")
    print("-" * 50)
    _, test_dataset = get_cls_dataset(cfg, split="test", val_root=cfg.dataset.test_root, enable_rotation=False, p=0.0)
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_all_acc4, test_avg_acc4, fuse234_score, test_cls_loss = validate(
        model=model, data_loader=test_loader, cfg=cfg, cls_loss_func=loss_function)
    print("Test results:")
    print(f"  all_acc4: {test_all_acc4:.6f} | avg_acc4: {test_avg_acc4:.6f}")
    print(f"  mIOU: {fuse234_score[:-1].mean():.4f} | scores: {fuse234_score}")
    print("  Per-class IoU:")
    for i, score in enumerate(fuse234_score[:-1]):
        print(f"    Class {i}: {score:.6f}")
    if len(fuse234_score) > cfg.dataset.cls_num_classes:
        print(f"    Background: {fuse234_score[-1]:.6f}")

    # Also keep reference for train_dataset for CAM generation below
    train_dataset, _ = get_cls_dataset(cfg, split="test", val_root=cfg.dataset.test_root, enable_rotation=False, p=0.0)

    print("\n2. Generating CAMs for complete training dataset...")
    print("-" * 50)
    
    train_cam_loader = DataLoader(train_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            persistent_workers=True)

    print(f"Generating CAMs for all {len(train_dataset)} training samples...")
    print(f"Output directory: {cfg.work_dir.pred_dir}")

    best_model_path = os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        best_iter = checkpoint.get("iter", "unknown")
        print(f"Best model loaded successfully! (Saved at iteration: {best_iter})")
    else:
        print("Warning: Best model checkpoint not found, using current model state")
        print(f"Expected path: {best_model_path}")
    
    generate_cam(model=model, data_loader=train_cam_loader, cfg=cfg)
    
    print("\nFiles generated:")
    print(f"  • Training CAM visualizations: {cfg.work_dir.pred_dir}/*.png")
    print(f"  • Model checkpoint: {cfg.work_dir.ckpt_dir}/best_cam.pth")
    print("="*80)
    
    
if __name__ == "__main__":
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(CONFIG_YAML)))
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.ckpt_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.train_log_dir, exist_ok=True)

    print('\nConfigs: %s' % cfg)

    set_seed(0)
    train(cfg=cfg)