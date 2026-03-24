# ============================================================
# BCSS-WSSS Datasets & Augmentations for Segmentation Training
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


# ── Train Dataset ─────────────────────────────────────────────

class BCSSTrainSeg(Dataset):
    """
    Dataset train BCSS-WSSS dùng pseudo label.
    Cấu trúc thư mục:
        train_root/   ← ảnh gốc (*.png, tên file chứa label: ...[0110]...)
        pseudo_dir/   ← mask pseudo (*.png, cùng tên với ảnh)
    """
    CLASSES = ['TUM', 'STR', 'LYM', 'NEC']

    def __init__(self, img_root, pseudo_dir, transform=None, ignore_index=255):
        self.transform = transform
        self.ignore_index = ignore_index
        self.img_paths = []
        self.mask_paths = []
        self.cls_labels = []

        # Scan images directly in img_root (no 'img' subfolder)
        all_imgs = glob.glob(os.path.join(img_root, '*.png'))
        for ip in sorted(all_imgs):
            mp = os.path.join(pseudo_dir, os.path.basename(ip))
            if not os.path.exists(mp):
                continue
            self.img_paths.append(ip)
            self.mask_paths.append(mp)
            # Parse class label từ tên file: ...[0110]...
            term_split = re.split(r'\[|\]', ip)
            cls_label = np.array([int(x) for x in term_split[1]])
            self.cls_labels.append(cls_label)

        print(f"  Train (BCSS): {len(self.img_paths)} paired samples")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        ip = self.img_paths[idx]
        mp = self.mask_paths[idx]
        img = cv.cvtColor(cv.imread(ip, cv.IMREAD_UNCHANGED), cv.COLOR_BGR2RGB)
        mask = np.array(Image.open(mp))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        # Map BACK(4) → ignore_index, keep 0-3 (TUM, STR, LYM, NEC)
        mask[mask == 4] = self.ignore_index
        mask = mask.astype(np.uint8)
        if self.transform:
            t = self.transform(image=img, mask=mask)
            img = t['image']
            mask = t['mask']
        return img, mask.long()


# ── Val/Test Dataset ──────────────────────────────────────────

class BCSSValSeg(Dataset):
    """
    Dataset val/test BCSS-WSSS với ground-truth mask.
    Cấu trúc thư mục:
        root/
            val/  (or test/)
                img/    ← ảnh (*.png)
                mask/   ← GT mask (*.png)
    """
    CLASSES = ['TUM', 'STR', 'LYM', 'NEC']

    def __init__(self, img_root, split='val', transform=None):
        assert split in ['val', 'valid', 'test'], "split must be one of [val, valid, test]"
        self.transform = transform
        self.img_paths = []
        self.mask_paths = []

        mask_dir = os.path.join(img_root, split, 'mask')
        for mp in sorted(glob.glob(os.path.join(mask_dir, '*.png'))):
            ip = os.path.join(img_root, split, 'img', os.path.basename(mp))
            if os.path.exists(ip):
                self.mask_paths.append(mp)
                self.img_paths.append(ip)

        print(f"  {split} (BCSS): {len(self.img_paths)} samples")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        ip = self.img_paths[idx]
        mp = self.mask_paths[idx]
        img = cv.cvtColor(cv.imread(ip, cv.IMREAD_UNCHANGED), cv.COLOR_BGR2RGB)
        mask = np.array(Image.open(mp))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        # Map BACK(4) → ignore_index=255
        mask[mask == 4] = 255
        if self.transform:
            t = self.transform(image=img, mask=mask)
            img = t['image']
            mask = t['mask']
        return img, mask.long()
