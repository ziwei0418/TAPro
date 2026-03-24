# ============================================================
# WSSS Stage 2: Segmentation Training
# Backbone: SegFormer mit-b3 (frozen + LoRA) + SegFormer Decoder
# Dataset: BCSS-WSSS | 4 tissue classes + Background
# ============================================================

import os, random, glob, math, re
import numpy as np
import cv2 as cv
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from omegaconf import OmegaConf

try:
    import pydensecrf.densecrf as dcrf
    import pydensecrf.utils as crf_utils
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    print("Warning: pydensecrf not available, CRF disabled")

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Config ────────────────────────────────────────────────────────────────────
SEG_CONFIG = """
model:
  backbone: nvidia/mit-b3
  num_classes: 5
  use_lora: true
  lora_rank: 8
  lora_alpha: 8.0
  deep_supervision: true

dataset:
  name: bcss
  train_root: /kaggle/working/BCSS-WSSS/training
  pseudo_dir: /kaggle/working/BCSS-WSSS/train/pseudo_label
  val_root: /kaggle/working/BCSS-WSSS/
  test_root: /kaggle/working/BCSS-WSSS/
  num_classes: 5
  ignore_index: 255
  mean: [0.7024, 0.5462, 0.6965]
  std:  [0.2389, 0.2821, 0.2163]

train:
  epochs: 30
  batch_size: 8
  accum_steps: 2
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
  ckpt_dir: /kaggle/working/seg_ckpt
  log_dir:  /kaggle/working/seg_log
"""

CLASS_NAMES = ['TUM', 'STR', 'LYM', 'NEC', 'BACK']

# ── LoRA ──────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, frozen_linear: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        self.frozen = frozen_linear
        in_f   = frozen_linear.in_features
        out_f  = frozen_linear.out_features
        device = frozen_linear.weight.device
        dtype  = frozen_linear.weight.dtype

        self.lora_A = nn.Parameter(torch.empty(rank, in_f,  device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        with torch.no_grad():
            h = self.frozen(x)
        return h + self.scale * (x @ self.lora_A.T @ self.lora_B.T)


def inject_lora_segformer(model: nn.Module, rank: int = 8, alpha: float = 8.0):
    count = 0
    for name, module in model.named_modules():
        if type(module).__name__ == 'SegformerEfficientSelfAttention':
            if isinstance(module.query, nn.Linear):
                module.query = LoRALinear(module.query, rank, alpha)
                count += 1
            if isinstance(module.value, nn.Linear):
                module.value = LoRALinear(module.value, rank, alpha)
                count += 1
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA injected into {count} layers | Trainable params: {lora_params/1e6:.2f}M")
    return model


# ── SegFormer Model ───────────────────────────────────────────────────────────

class SegFormerSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        num_classes = cfg.model.num_classes
        backbone_id = cfg.model.backbone

        seg_cfg = SegformerConfig.from_pretrained(backbone_id)
        seg_cfg.num_labels          = num_classes
        seg_cfg.id2label            = {i: n for i, n in enumerate(CLASS_NAMES)}
        seg_cfg.label2id            = {n: i for i, n in enumerate(CLASS_NAMES)}
        seg_cfg.decoder_hidden_size = 512

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            backbone_id,
            config=seg_cfg,
            ignore_mismatched_sizes=True,
        )

        for p in self.model.segformer.parameters():
            p.requires_grad_(False)

        self.deep_sup = cfg.model.deep_supervision
        if cfg.model.use_lora:
            inject_lora_segformer(
                self.model.segformer,
                rank=cfg.model.lora_rank,
                alpha=cfg.model.lora_alpha,
            )

        # SegFormer mit-b3 hidden dims per stage: [64, 128, 320, 512]
        if self.deep_sup:
            self.aux_head1 = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, num_classes, 1),
            )
            self.aux_head2 = nn.Sequential(
                nn.Conv2d(320, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, num_classes, 1),
            )

        total_p = sum(p.numel() for p in self.parameters())
        train_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total params: {total_p/1e6:.2f}M | Trainable: {train_p/1e6:.2f}M")

    def forward(self, x):
        H, W = x.shape[2:]
        encoder_out = self.model.segformer(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = encoder_out.hidden_states   # 4 stages

        logits = self.model.decode_head(hidden_states)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        if not self.deep_sup or not self.training:
            return logits, None, None

        aux1 = F.interpolate(
            self.aux_head1(hidden_states[1]), size=(H, W),
            mode='bilinear', align_corners=False,
        )
        aux2 = F.interpolate(
            self.aux_head2(hidden_states[2]), size=(H, W),
            mode='bilinear', align_corners=False,
        )
        return logits, aux1, aux2

    def get_param_groups(self, lr: float, backbone_scale: float = 0.1):
        lora_params, decoder_params, aux_params = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad: continue
            if 'model.segformer' in name:
                lora_params.append(p)
            elif 'aux_head' in name:
                aux_params.append(p)
            else:
                decoder_params.append(p)
        groups = [
            {'params': lora_params,    'lr': lr * backbone_scale, 'name': 'lora'},
            {'params': decoder_params, 'lr': lr,                   'name': 'decoder'},
        ]
        if aux_params:
            groups.append({'params': aux_params, 'lr': lr, 'name': 'aux'})
        return groups


# ── Dataset ───────────────────────────────────────────────────────────────────

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


class BCSSTrainSeg(Dataset):
    """
    Dataset train BCSS-WSSS dùng pseudo label.
    Cấu trúc thư mục:
        train_root/
            img/        ← ảnh gốc  (*.png, tên file chứa label: ...[0110]...)
            pseudo_label/ ← mask pseudo (*.png, cùng tên với ảnh)
    """
    CLASSES = ['TUM', 'STR', 'LYM', 'NEC', 'BACK']

    def __init__(self, img_root, pseudo_dir, transform=None, ignore_index=255):
        self.transform    = transform
        self.ignore_index = ignore_index
        self.img_paths    = []
        self.mask_paths   = []
        self.cls_labels   = []

        all_imgs = glob.glob(os.path.join(img_root, 'img', '*.png'))
        for ip in sorted(all_imgs):
            mp = os.path.join(pseudo_dir, os.path.basename(ip))
            if not os.path.exists(mp):
                continue
            self.img_paths.append(ip)
            self.mask_paths.append(mp)
            # Parse class label từ tên file: ...[0110]...
            term_split = re.split(r'\[|\]', ip)
            cls_label  = np.array([int(x) for x in term_split[1]])
            self.cls_labels.append(cls_label)

        print(f"  Train (BCSS): {len(self.img_paths)} paired samples")

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        ip  = self.img_paths[idx]
        mp  = self.mask_paths[idx]
        img = cv.cvtColor(cv.imread(ip, cv.IMREAD_UNCHANGED), cv.COLOR_BGR2RGB)
        mask = np.array(Image.open(mp))
        if mask.ndim == 3: mask = mask[:, :, 0]
        mask = np.clip(mask, 0, 4).astype(np.uint8)
        if self.transform:
            t    = self.transform(image=img, mask=mask)
            img  = t['image']
            mask = t['mask']
        return img, mask.long()


class BCSSValSeg(Dataset):
    """
    Dataset val/test BCSS-WSSS với ground-truth mask.
    Cấu trúc thư mục:
        root/
            val/
                img/    ← ảnh (*.png)
                mask/   ← GT mask (*.png)
            test/
                img/
                mask/
    """
    CLASSES = ['TUM', 'STR', 'LYM', 'NEC', 'BACK']

    def __init__(self, img_root, split='val', transform=None):
        assert split in ['val', 'valid', 'test'], "split must be one of [val, valid, test]"
        self.transform = transform
        self.img_paths  = []
        self.mask_paths = []

        mask_dir = os.path.join(img_root, split, 'mask')
        for mp in sorted(glob.glob(os.path.join(mask_dir, '*.png'))):
            ip = os.path.join(img_root, split, 'img', os.path.basename(mp))
            if os.path.exists(ip):
                self.mask_paths.append(mp)
                self.img_paths.append(ip)

        print(f"  {split} (BCSS): {len(self.img_paths)} samples")

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        ip  = self.img_paths[idx]
        mp  = self.mask_paths[idx]
        img  = cv.cvtColor(cv.imread(ip, cv.IMREAD_UNCHANGED), cv.COLOR_BGR2RGB)
        mask = np.array(Image.open(mp))
        if mask.ndim == 3: mask = mask[:, :, 0]
        if self.transform:
            t    = self.transform(image=img, mask=mask)
            img  = t['image']
            mask = t['mask']
        return img, mask.long()


# ── Loss ──────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, smooth=1.0):
        super().__init__()
        self.C      = num_classes
        self.ig     = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        valid = (targets != self.ig)
        t     = targets.clone(); t[~valid] = 0
        oh    = F.one_hot(t, self.C).permute(0, 3, 1, 2).float()
        v     = valid.unsqueeze(1).float()
        inter = (probs * oh * v).sum((2, 3))
        union = ((probs + oh) * v).sum((2, 3))
        dice  = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BootstrappedCE(nn.Module):
    def __init__(self, beta=0.1, ignore_index=255):
        super().__init__()
        self.beta = beta
        self.ig   = ignore_index

    def forward(self, logits, targets):
        valid    = (targets != self.ig)
        ce_gt    = F.cross_entropy(logits, targets, ignore_index=self.ig, reduction='none')
        with torch.no_grad():
            pseudo = logits.detach().argmax(dim=1)
        ce_model = F.cross_entropy(logits, pseudo, ignore_index=self.ig, reduction='none')
        loss = (1.0 - self.beta) * ce_gt + self.beta * ce_model
        return loss[valid].mean()


class SegLoss(nn.Module):
    def __init__(self, cfg, class_weights=None):
        super().__init__()
        self.ig       = cfg.dataset.ignore_index
        self.ce_w     = cfg.train.ce_weight
        self.dice_w   = cfg.train.dice_weight
        self.deep_sup = cfg.model.deep_supervision
        nc            = cfg.dataset.num_classes

        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=cfg.train.label_smoothing,
            ignore_index=self.ig,
        )
        self.dice = DiceLoss(nc, self.ig)
        self.bootstrap = BootstrappedCE(cfg.train.noise_beta, self.ig) \
            if cfg.train.use_noise_robust_loss else None

    def forward(self, main, aux1, aux2, targets):
        if self.bootstrap is not None:
            loss = self.ce_w * self.bootstrap(main, targets)
        else:
            loss = self.ce_w * self.ce(main, targets)
        loss = loss + self.dice_w * self.dice(main, targets)
        if self.deep_sup and aux1 is not None:
            loss = loss + 0.3 * self.ce(aux1, targets) + 0.3 * self.ce(aux2, targets)
        return loss


# ── Metrics ───────────────────────────────────────────────────────────────────

class SegMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.C  = num_classes
        self.ig = ignore_index
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.C, self.C), dtype=np.int64)

    def update(self, pred, gt):
        valid = gt != self.ig
        p, g  = pred[valid], gt[valid]
        self.cm += np.bincount(g * self.C + p, minlength=self.C**2).reshape(self.C, self.C)

    def compute(self):
        h    = self.cm.astype(np.float64)
        diag = np.diag(h)
        iou  = diag / (h.sum(1) + h.sum(0) - diag + 1e-8)
        dice = 2 * diag / (h.sum(1) + h.sum(0) + 1e-8)
        freq = h.sum(1) / (h.sum() + 1e-8)
        return {
            'iou'  : iou,
            'miou' : iou[:-1].mean(),   # exclude BACK
            'dice' : dice[:-1].mean(),
            'fwiou': (freq * iou).sum(),
        }


# ── CRF Post-processing ───────────────────────────────────────────────────────

def apply_crf(img_rgb, prob_map, iter_max=10):
    if not CRF_AVAILABLE: return prob_map.argmax(0)
    C, H, W = prob_map.shape
    U = crf_utils.unary_from_softmax(prob_map)
    U = np.ascontiguousarray(U)
    d = dcrf.DenseCRF2D(W, H, C)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=1, compat=3)
    d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=np.ascontiguousarray(img_rgb), compat=4)
    Q = np.array(d.inference(iter_max)).reshape(C, H, W)
    return Q.argmax(0)


# ── Scheduler ─────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg, steps_per_epoch):
    total  = cfg.train.epochs * steps_per_epoch
    warmup = cfg.train.warmup_epochs * steps_per_epoch

    def lr_fn(step):
        if step < warmup:
            return max(1e-6, step / warmup)
        t = (step - warmup) / max(total - warmup, 1)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * t)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


# ── Validate ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device, cfg, use_crf=False):
    model.eval()
    metrics = SegMetrics(cfg.dataset.num_classes, cfg.dataset.ignore_index)
    mean    = np.array(cfg.dataset.mean)
    std     = np.array(cfg.dataset.std)

    for imgs, masks in tqdm(loader, ncols=80, desc='Val', leave=False):
        imgs = imgs.to(device).float()
        out, _, _ = model(imgs)
        probs = F.softmax(out, dim=1)

        for i in range(imgs.shape[0]):
            if use_crf and CRF_AVAILABLE:
                img_np  = imgs[i].cpu().numpy().transpose(1, 2, 0)
                img_rgb = ((img_np * std + mean) * 255).clip(0, 255).astype(np.uint8)
                pred    = apply_crf(img_rgb, probs[i].cpu().numpy())
            else:
                pred = out[i].argmax(0).cpu().numpy()
            metrics.update(pred, masks[i].numpy())

    model.train()
    return metrics.compute()


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(pseudo_dir, num_classes=5, max_samples=1000):
    counts = np.zeros(num_classes, dtype=np.float64)
    paths  = glob.glob(os.path.join(pseudo_dir, '*.png'))[:max_samples]
    for p in paths:
        m = np.array(Image.open(p))
        if m.ndim == 3: m = m[:, :, 0]
        m = np.clip(m, 0, num_classes - 1)
        for c in range(num_classes):
            counts[c] += (m == c).sum()
    freq    = counts / (counts.sum() + 1e-8)
    median  = np.median(freq[freq > 1e-6])
    weights = np.where(freq > 1e-6, median / (freq + 1e-8), 1.0)
    weights = np.clip(weights, 0.5, 4.0)
    print(f"  Class weights: { {n: f'{w:.2f}' for n, w in zip(CLASS_NAMES, weights)} }")
    return torch.tensor(weights, dtype=torch.float32)


# ── Train ─────────────────────────────────────────────────────────────────────

def train_seg(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.log_dir,  exist_ok=True)

    random.seed(42); np.random.seed(42)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

    mean = list(cfg.dataset.mean)
    std  = list(cfg.dataset.std)

    print("\nLoading datasets...")
    train_ds = BCSSTrainSeg(
        img_root=cfg.dataset.train_root,
        pseudo_dir=cfg.dataset.pseudo_dir,
        transform=get_train_aug(mean, std, size=512),
        ignore_index=cfg.dataset.ignore_index,
    )
    val_ds  = BCSSValSeg(cfg.dataset.val_root,  split='val',  transform=get_val_aug(mean, std))
    test_ds = BCSSValSeg(cfg.dataset.test_root, split='test', transform=get_val_aug(mean, std))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader  = DataLoader(val_ds,  batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    print("\nBuilding SegFormer mit-b3 model...")
    model = SegFormerSeg(cfg).to(device)
    model.model.segformer.eval()   # giữ BN encoder ở eval mode

    print("\nComputing class weights...")
    class_weights = compute_class_weights(
        cfg.dataset.pseudo_dir, cfg.dataset.num_classes
    ).to(device)
    criterion = SegLoss(cfg, class_weights).to(device)

    param_groups = model.get_param_groups(
        lr=cfg.train.lr,
        backbone_scale=cfg.train.backbone_lr_scale,
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)

    accum = cfg.train.accum_steps
    actual_steps_per_epoch = max(1, len(train_loader) // accum)
    scheduler = build_scheduler(optimizer, cfg, actual_steps_per_epoch)
    scaler    = torch.amp.GradScaler('cuda')

    best_val_miou  = 0.0
    best_test_miou = 0.0

    print(f"\nTraining {cfg.train.epochs} epochs | BS={cfg.train.batch_size} accum={accum} | Steps/epoch={len(train_loader)}")
    print("=" * 70)

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        model.model.segformer.eval()   # giữ BN encoder ở eval
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            ncols=100,
            desc=f'Ep {epoch:02d}/{cfg.train.epochs}',
        )

        for step, (imgs, masks) in pbar:
            imgs  = imgs.to(device).float()
            masks = masks.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                main, aux1, aux2 = model(imgs)
                loss = criterion(main, aux1, aux2, masks) / accum

            scaler.scale(loss).backward()
            total_loss += loss.item() * accum

            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            pbar.set_postfix({
                'loss': f'{total_loss/(step+1):.4f}',
                'lr_d': f'{optimizer.param_groups[1]["lr"]:.1e}',
            })

        avg_loss = total_loss / len(train_loader)

        use_crf = getattr(cfg.train, 'use_crf_val', False)
        val_m   = validate(model, val_loader,  device, cfg, use_crf)
        test_m  = validate(model, test_loader, device, cfg, use_crf)

        print(f"\nEp {epoch:02d} | Loss={avg_loss:.4f} | LR={optimizer.param_groups[1]['lr']:.2e}")
        print(f"  [VAL]  mIoU={val_m['miou']*100:.2f}% | fwIoU={val_m['fwiou']*100:.2f}% | Dice={val_m['dice']*100:.2f}%")
        print(f"         {' | '.join(f'{n}:{v*100:.1f}%' for n, v in zip(CLASS_NAMES, val_m['iou']))}")
        print(f"  [TEST] mIoU={test_m['miou']*100:.2f}% | fwIoU={test_m['fwiou']*100:.2f}% | Dice={test_m['dice']*100:.2f}%")
        print(f"         {' | '.join(f'{n}:{v*100:.1f}%' for n, v in zip(CLASS_NAMES, test_m['iou']))}")

        if val_m['miou'] > best_val_miou:
            best_val_miou  = val_m['miou']
            best_test_miou = test_m['miou']
            torch.save({
                'epoch'    : epoch,
                'model'    : model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_miou' : best_val_miou,
                'test_miou': best_test_miou,
                'cfg'      : cfg,
            }, os.path.join(cfg.work_dir.ckpt_dir, 'best_seg.pth'))
            print(f"  ✓ Best saved — val={best_val_miou*100:.2f}% test={best_test_miou*100:.2f}%")
        print("=" * 70)

    print("\nFinal evaluation with best checkpoint...")
    ckpt = torch.load(
        os.path.join(cfg.work_dir.ckpt_dir, 'best_seg.pth'),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt['model'])

    for split, loader in [('VAL', val_loader), ('TEST', test_loader)]:
        m = validate(model, loader, device, cfg, use_crf=CRF_AVAILABLE)
        print(f"\n[{split}] mIoU={m['miou']*100:.2f}% | Dice={m['dice']*100:.2f}% | fwIoU={m['fwiou']*100:.2f}%")
        for name, iou in zip(CLASS_NAMES, m['iou']):
            print(f"  {name:4s}: {iou*100:.2f}%")


if __name__ == '__main__':
    cfg = OmegaConf.create(SEG_CONFIG)
    print(f"Config: {cfg}")
    train_seg(cfg)