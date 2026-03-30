# ============================================================
# Datasets & Augmentations for Segmentation Training
# Supports: BCSS-WSSS and LUAD-HistoSeg
# ============================================================

import os
import re
import glob

import cv2 as cv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Augmentations ─────────────────────────────────────────────

def get_train_aug(mean, std, size=512):
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            A.HueSaturationValue(10, 25, 20),
        ], p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MedianBlur(blur_limit=3),
        ], p=0.2),
        A.Normalize(mean, std),
        ToTensorV2(transpose_mask=True),
    ])


def get_val_aug(mean, std):
    return A.Compose([
        A.Normalize(mean, std),
        ToTensorV2(transpose_mask=True),
    ])


# ── Label parsing helpers ─────────────────────────────────────

def _parse_bcss_label(filepath):
    """Parse BCSS label from filename: ...[0110]... (no spaces)."""
    term_split = re.split(r'\[|\]', filepath)
    return np.array([int(x) for x in term_split[1]])


def _parse_luad_label(filepath):
    """Parse LUAD label from filename: ...[1 0 1 0]... (space-separated)."""
    fname = os.path.basename(filepath)[:-4]  # remove .png
    label_str = fname.split(']')[0].split('[')[-1].split(' ')
    return np.array([int(x) for x in label_str])


# ── Base Train Dataset ────────────────────────────────────────

class _BaseTrainSeg(Dataset):
    """Base class for pseudo-label training datasets."""

    def __init__(self, img_root, pseudo_dir, transform=None, ignore_index=255,
                 label_parser=None, dataset_name=''):
        self.transform = transform
        self.ignore_index = ignore_index
        self.img_paths = []
        self.mask_paths = []
        self.cls_labels = []

        all_imgs = glob.glob(os.path.join(img_root, '*.png'))
        for ip in sorted(all_imgs):
            mp = os.path.join(pseudo_dir, os.path.basename(ip))
            if not os.path.exists(mp):
                continue
            self.img_paths.append(ip)
            self.mask_paths.append(mp)
            if label_parser:
                self.cls_labels.append(label_parser(ip))

        print(f"  Train ({dataset_name}): {len(self.img_paths)} paired samples")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        ip = self.img_paths[idx]
        mp = self.mask_paths[idx]
        img = cv.cvtColor(cv.imread(ip, cv.IMREAD_UNCHANGED), cv.COLOR_BGR2RGB)
        mask = np.array(Image.open(mp))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask[mask == 4] = self.ignore_index
        mask = mask.astype(np.uint8)
        if self.transform:
            t = self.transform(image=img, mask=mask)
            img = t['image']
            mask = t['mask']
        return img, mask.long()


# ── Base Val/Test Dataset ─────────────────────────────────────

class _BaseValSeg(Dataset):
    """Base class for val/test datasets with GT masks."""

    def __init__(self, img_root, split='val', transform=None, dataset_name=''):
        self.transform = transform
        self.img_paths = []
        self.mask_paths = []

        mask_dir = os.path.join(img_root, split, 'mask')
        for mp in sorted(glob.glob(os.path.join(mask_dir, '*.png'))):
            ip = os.path.join(img_root, split, 'img', os.path.basename(mp))
            if os.path.exists(ip):
                self.mask_paths.append(mp)
                self.img_paths.append(ip)

        print(f"  {split} ({dataset_name}): {len(self.img_paths)} samples")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        ip = self.img_paths[idx]
        mp = self.mask_paths[idx]
        img = cv.cvtColor(cv.imread(ip, cv.IMREAD_UNCHANGED), cv.COLOR_BGR2RGB)
        mask = np.array(Image.open(mp))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask[mask == 4] = 255
        if self.transform:
            t = self.transform(image=img, mask=mask)
            img = t['image']
            mask = t['mask']
        return img, mask.long()


# ── BCSS Concrete Classes ─────────────────────────────────────

class BCSSTrainSeg(_BaseTrainSeg):
    CLASSES = ['TUM', 'STR', 'LYM', 'NEC']
    def __init__(self, img_root, pseudo_dir, transform=None, ignore_index=255):
        super().__init__(img_root, pseudo_dir, transform, ignore_index,
                         label_parser=_parse_bcss_label, dataset_name='BCSS')


class BCSSValSeg(_BaseValSeg):
    CLASSES = ['TUM', 'STR', 'LYM', 'NEC']
    def __init__(self, img_root, split='val', transform=None):
        super().__init__(img_root, split, transform, dataset_name='BCSS')


# ── LUAD Concrete Classes ─────────────────────────────────────

class LUADTrainSeg(_BaseTrainSeg):
    CLASSES = ['TE', 'NEC', 'LYM', 'TAS']
    def __init__(self, img_root, pseudo_dir, transform=None, ignore_index=255):
        super().__init__(img_root, pseudo_dir, transform, ignore_index,
                         label_parser=_parse_luad_label, dataset_name='LUAD')


class LUADValSeg(_BaseValSeg):
    CLASSES = ['TE', 'NEC', 'LYM', 'TAS']
    def __init__(self, img_root, split='val', transform=None):
        super().__init__(img_root, split, transform, dataset_name='LUAD')


# ── Factory ───────────────────────────────────────────────────

def get_datasets(cfg, mean, std):
    """Factory: returns (TrainDataset, ValDataset, TestDataset) for the given config."""
    train_aug = get_train_aug(mean, std, cfg.train.crop_size)
    val_aug = get_val_aug(mean, std)

    ds_name = cfg.dataset.name.lower()
    pseudo_dir = cfg.dataset.pseudo_dir

    if ds_name == 'luad':
        train_ds = LUADTrainSeg(cfg.dataset.train_root, pseudo_dir, train_aug, cfg.dataset.ignore_index)
        val_ds = LUADValSeg(cfg.dataset.val_root, 'val', val_aug)
        test_ds = LUADValSeg(cfg.dataset.test_root, 'test', val_aug)
    else:  # bcss (default)
        train_ds = BCSSTrainSeg(cfg.dataset.train_root, pseudo_dir, train_aug, cfg.dataset.ignore_index)
        val_ds = BCSSValSeg(cfg.dataset.val_root, 'val', val_aug)
        test_ds = BCSSValSeg(cfg.dataset.test_root, 'test', val_aug)

    return train_ds, val_ds, test_ds
