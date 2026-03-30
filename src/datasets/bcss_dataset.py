"""
Dataset classes for BCSS and LUAD-HistoSeg
"""
import os
import glob
import re
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Label parsing helpers ─────────────────────────────────────

def _parse_bcss_label(img_path):
    """Parse BCSS label: ...[0110]... (no spaces)."""
    term_split = re.split(r"\[|\]", img_path)
    return np.array([int(x) for x in term_split[1]])


def _parse_luad_label(img_path):
    """Parse LUAD label: ...[1 0 1 0]... (space-separated)."""
    fname = os.path.basename(img_path)[:-4]
    label_str = fname.split(']')[0].split('[')[-1].split(' ')
    return np.array([int(x) for x in label_str])


# ── BCSS Datasets ─────────────────────────────────────────────

class BCSSTrainingDataset(Dataset):
    """BCSS Training Dataset with image-level labels"""
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
        assert os.path.exists(img_path), f"img_path: {img_path} does not exists"

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return os.path.basename(img_path), img, cls_label, 0

    def get_images_and_labels(self, img_root=None):
        self.img_paths = []
        self.cls_labels = []

        self.img_paths = glob.glob(os.path.join(img_root, "*.png"))
        for img_path in self.img_paths:
            self.cls_labels.append(_parse_bcss_label(img_path))


class BCSSTestDataset(Dataset):
    """BCSS Test/Validation Dataset with pixel-level ground truth"""
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
        assert os.path.exists(img_path), f"img_path: {img_path} does not exists"
        assert os.path.exists(mask_path), f"mask_path: {mask_path} does not exists"

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
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
    """BCSS WSSS Dataset with pseudo labels for refinement"""
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
        assert os.path.exists(img_path), f"image path: {img_path}, does not exist"
        assert os.path.exists(mask_path), f"mask path: {mask_path}, does not exist"

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
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
            self.cls_labels.append(_parse_bcss_label(img_path))


# ── LUAD Datasets ─────────────────────────────────────────────

class LUADTrainingDataset(Dataset):
    """LUAD-HistoSeg Training Dataset with image-level labels"""
    CLASSES = ["TE", "NEC", "LYM", "TAS", "BACK"]

    def __init__(self, img_root="../data/LUAD-HistoSeg/training", transform=None):
        super(LUADTrainingDataset, self).__init__()
        self.get_images_and_labels(img_root)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        cls_label = self.cls_labels[index]
        assert os.path.exists(img_path), f"img_path: {img_path} does not exists"

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return os.path.basename(img_path), img, cls_label, 0

    def get_images_and_labels(self, img_root=None):
        self.img_paths = []
        self.cls_labels = []

        self.img_paths = glob.glob(os.path.join(img_root, "*.png"))
        for img_path in self.img_paths:
            self.cls_labels.append(_parse_luad_label(img_path))


class LUADTestDataset(Dataset):
    """LUAD-HistoSeg Test/Validation Dataset with pixel-level ground truth"""
    CLASSES = ["TE", "NEC", "LYM", "TAS", "BACK"]

    def __init__(self, img_root="../data/LUAD-HistoSeg/", split="val", transform=None):
        assert split in ["test", "val"], "split must be one of [test, val]"
        super(LUADTestDataset, self).__init__()
        self.get_images_and_labels(img_root, split)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        assert os.path.exists(img_path), f"img_path: {img_path} does not exists"
        assert os.path.exists(mask_path), f"mask_path: {mask_path} does not exists"

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


# ── Normalization ─────────────────────────────────────────────

def get_mean_std(dataset_name):
    """Get normalization parameters for dataset."""
    if dataset_name == "luad":
        norm = [[0.7024, 0.5462, 0.6965], [0.2389, 0.2821, 0.2163]]
    else:  # bcss
        norm = [[0.66791496, 0.47791372, 0.70623304], [0.1736589, 0.22564577, 0.19820057]]
    return norm[0], norm[1]


# ── Factory ───────────────────────────────────────────────────

def get_cls_dataset(cfg, split="val", val_root=None, p=0.5, enable_rotation=True):
    """Load training + val/test dataset. Auto-selects BCSS or LUAD based on cfg."""
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

    ds_name = cfg.dataset.name.lower()

    if ds_name == "luad":
        train_dataset = LUADTrainingDataset(cfg.dataset.train_root, transform=transform["train"])
        if split in ["test", "val"]:
            val_dataset = LUADTestDataset(_val_root, split, transform=transform["val"])
        else:
            val_dataset = None
    else:  # bcss
        train_dataset = BCSSTrainingDataset(cfg.dataset.train_root, transform=transform["train"])
        if split in ["test", "valid", "val"]:
            val_dataset = BCSSTestDataset(_val_root, split, transform=transform["val"])
        else:
            val_dataset = None

    return train_dataset, val_dataset
