# ============================================================
# WSSS Stage 2: Segmentation Training
# Backbone: UNI ViT-L/16 (frozen + LoRA) + UPerNet Decoder
# Dataset: LUAD-HistoSeg | 4 tissue classes + Background
# ============================================================

import os, datetime, random, glob, math
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from huggingface_hub import login as hf_login

import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf

try:
    import pydensecrf.densecrf as dcrf
    import pydensecrf.utils as crf_utils
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    print("Warning: pydensecrf not available, CRF disabled")

# ── Config ────────────────────────────────────────────────────────────────────
SEG_CONFIG = """
model:
  hf_token: REMOVED_HF_TOKEN
  num_classes: 5
  use_lora: true
  lora_rank: 8
  lora_alpha: 8.0
  decoder_channels: 256
  deep_supervision: true

dataset:
  name: luad
  train_root: /kaggle/working/LUAD-HistoSeg/training
  pseudo_dir: /kaggle/working/predictions
  val_root: /kaggle/working/LUAD-HistoSeg/
  test_root: /kaggle/working/LUAD-HistoSeg/
  num_classes: 5
  ignore_index: 255
  mean: [0.7024, 0.5462, 0.6965]
  std:  [0.2389, 0.2821, 0.2163]

train:
  epochs: 30            
  batch_size: 8
  accum_steps: 2
  num_workers: 8
  lr: 0.00003            
  backbone_lr_scale: 0.05
  weight_decay: 0.05     
  warmup_epochs: 3
  label_smoothing: 0.0   
  ce_weight: 1.0
  dice_weight: 0.5
  use_crf_val: false
  use_noise_robust_loss: true
  noise_beta: 0.1        # SỬA: Hạ từ 0.3 xuống 0.1

work_dir:
  ckpt_dir: /kaggle/working/seg_ckpt
  log_dir:  /kaggle/working/seg_log
"""

CLASS_NAMES = ['TE', 'NEC', 'LYM', 'TAS', 'BACK']

# ── LoRA ──────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, frozen_linear, rank=8, alpha=8.0):
        super().__init__()
        self.frozen = frozen_linear
        in_f  = frozen_linear.in_features
        out_f = frozen_linear.out_features
        device = frozen_linear.weight.device
        dtype  = frozen_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, in_f, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        with torch.no_grad():
            h = self.frozen(x)
        return h + self.scale * (x @ self.lora_A.T @ self.lora_B.T)


# ── UNI Encoder ───────────────────────────────────────────────────────────────

class UNIEncoderSeg(nn.Module):
    BLOCK_IDXS = (5, 11, 17, 23)
    EMBED_DIM  = 1024

    def __init__(self, hf_token, img_size=224, use_lora=True,
                 lora_rank=8, lora_alpha=8.0):
        super().__init__()
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
        self.grid = img_size // self.model.patch_embed.patch_size[0]

        if use_lora:
            for idx in self.BLOCK_IDXS:
                blk = self.model.blocks[idx]
                blk.attn.qkv = LoRALinear(blk.attn.qkv, lora_rank, lora_alpha)
            lora_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"  UNI LoRA injected: {lora_p/1e6:.2f}M trainable params")

    def forward(self, x):
        B, _, H, W = x.shape
        G = self.grid
        D = self.EMBED_DIM
        captured = {}

        def make_hook(idx):
            def hook(m, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if t.ndim == 3 and t.shape[1] == G*G + 1:
                    t = t[:, 1:, :]
                captured[idx] = t.reshape(B, G, G, D).permute(0,3,1,2).contiguous()
            return hook

        hooks = [self.model.blocks[i].register_forward_hook(make_hook(i))
                 for i in self.BLOCK_IDXS]
        try:
            self.model(x)
        finally:
            for h in hooks: h.remove()

        return [captured[i] for i in self.BLOCK_IDXS]


# ── UPerNet Decoder ───────────────────────────────────────────────────────────

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins=(1, 2, 3, 6)):
        super().__init__()
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(b),
                nn.Conv2d(in_dim, reduction_dim, 1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True),
            ) for b in bins
        ])

    def forward(self, x):
        H, W = x.shape[2:]
        out  = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), (H, W), mode='bilinear', align_corners=True))
        return torch.cat(out, dim=1)


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ) for c in in_channels_list
        ])
        self.outputs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ) for _ in in_channels_list
        ])

    def forward(self, feats):
        laterals = [l(f) for l, f in zip(self.laterals, feats)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i+1], size=laterals[i].shape[2:],
                mode='bilinear', align_corners=True
            )
        return [o(l) for o, l in zip(self.outputs, laterals)]


class UPerNetDecoder(nn.Module):
    def __init__(self, in_channels=1024, num_scales=4,
                 decoder_channels=256, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

        ppm_out = in_channels + 4 * (decoder_channels // 4)
        self.ppm = PPM(in_channels, decoder_channels // 4, bins=(1,2,3,6))

        self.ppm_bottleneck = nn.Sequential(
            nn.Conv2d(ppm_out, decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )

        fpn_in = [in_channels] * (num_scales - 1) + [decoder_channels]
        self.fpn = FPN(fpn_in, decoder_channels)

        self.fusion = nn.Sequential(
            nn.Conv2d(decoder_channels * num_scales, decoder_channels,
                      3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.seg_head = nn.Conv2d(decoder_channels, num_classes, 1)

        self.aux_head2 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )
        self.aux_head3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, feats, img_size, deep_sup=True):
        f0, f1, f2, f3 = feats
        f3_ppm = self.ppm_bottleneck(self.ppm(f3))
        fpn_feats  = self.fpn([f0, f1, f2, f3_ppm])
        target_size = fpn_feats[0].shape[2:]
        fused = torch.cat([
            F.interpolate(f, target_size, mode='bilinear', align_corners=True)
            for f in fpn_feats
        ], dim=1)

        fused = self.fusion(fused)
        out = F.interpolate(self.seg_head(fused), img_size,
                            mode='bilinear', align_corners=True)

        if not deep_sup or not self.training:
            return out, None, None

        aux2 = F.interpolate(self.aux_head2(f1), img_size,
                             mode='bilinear', align_corners=True)
        aux3 = F.interpolate(self.aux_head3(f2), img_size,
                             mode='bilinear', align_corners=True)
        return out, aux2, aux3


# ── Full Model ────────────────────────────────────────────────────────────────

class UNISegNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = UNIEncoderSeg(
            hf_token   = cfg.model.hf_token,
            img_size   = 224,
            use_lora   = cfg.model.use_lora,
            lora_rank  = cfg.model.lora_rank,
            lora_alpha = cfg.model.lora_alpha,
        )
        self.decoder = UPerNetDecoder(
            in_channels      = 1024,
            num_scales       = 4,
            decoder_channels = cfg.model.decoder_channels,
            num_classes      = cfg.model.num_classes,
        )
        self.deep_sup = cfg.model.deep_supervision

    def forward(self, x):
        feats = self.encoder(x)
        return self.decoder(feats, x.shape[2:], self.deep_sup)

    def get_param_groups(self, lr, backbone_scale=0.05):
        lora_params    = []
        decoder_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad: continue
            if 'encoder' in name:
                lora_params.append(p)
            else:
                decoder_params.append(p)
        return [
            {'params': lora_params,    'lr': lr * backbone_scale, 'name': 'lora'},
            {'params': decoder_params, 'lr': lr,                  'name': 'decoder'},
        ]


# ── Dataset ───────────────────────────────────────────────────────────────────

def get_train_aug(mean, std, size=224):
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


class LUADTrainSeg(Dataset):
    def __init__(self, img_root, pseudo_dir, transform=None, ignore_index=255):
        self.transform    = transform
        self.ignore_index = ignore_index
        imgs  = sorted(glob.glob(os.path.join(img_root, '*.png')))
        self.pairs = []
        for ip in imgs:
            mp = os.path.join(pseudo_dir, os.path.basename(ip))
            if os.path.exists(mp):
                self.pairs.append((ip, mp))
        print(f"  Train: {len(self.pairs)} paired samples")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img    = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        mask   = np.array(Image.open(mp))
        if mask.ndim == 3: mask = mask[:, :, 0]
        mask = np.clip(mask, 0, 4).astype(np.uint8)
        if self.transform:
            t    = self.transform(image=img, mask=mask)
            img  = t['image']
            mask = t['mask']
        return img, mask.long()


class LUADValSeg(Dataset):
    def __init__(self, img_root, split='val', transform=None):
        self.transform = transform
        mps = sorted(glob.glob(os.path.join(img_root, split, 'mask', '*.png')))
        self.pairs = []
        for mp in mps:
            ip = os.path.join(img_root, split, 'img', os.path.basename(mp))
            if os.path.exists(ip):
                self.pairs.append((ip, mp))
        print(f"  {split}: {len(self.pairs)} samples")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img    = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        mask   = np.array(Image.open(mp))
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
        self.C = num_classes
        self.ig = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        valid = (targets != self.ig)
        t     = targets.clone(); t[~valid] = 0
        oh    = F.one_hot(t, self.C).permute(0,3,1,2).float()
        v     = valid.unsqueeze(1).float()
        inter = (probs * oh * v).sum((2,3))
        union = ((probs + oh) * v).sum((2,3))
        dice  = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BootstrappedCE(nn.Module):
    """
    CHUẨN BOOTSTRAP: Trộn loss thực tế từ GT (pseudo label) và từ chính dự đoán của Model.
    Giúp model không bị overfit vào nhiễu của pseudo label.
    """
    def __init__(self, beta=0.3, ignore_index=255):
        super().__init__()
        self.beta = beta
        self.ig = ignore_index

    def forward(self, logits, targets):
        valid = (targets != self.ig)
        
        # Loss tính trên Pseudo Label (Ground Truth dỏm)
        ce_gt = F.cross_entropy(logits, targets, ignore_index=self.ig, reduction='none')
        
        # Loss tính trên dự đoán hiện tại của Model (tự học tự sửa)
        with torch.no_grad():
            pseudo = logits.detach().argmax(dim=1)
        ce_model = F.cross_entropy(logits, pseudo, ignore_index=self.ig, reduction='none')
        
        # Kết hợp theo tỉ lệ beta
        loss = (1.0 - self.beta) * ce_gt + self.beta * ce_model
        return loss[valid].mean()


class SegLoss(nn.Module):
    def __init__(self, cfg, class_weights=None):
        super().__init__()
        self.ig        = cfg.dataset.ignore_index
        self.ce_w      = cfg.train.ce_weight
        self.dice_w    = cfg.train.dice_weight
        self.deep_sup  = cfg.model.deep_supervision
        nc             = cfg.dataset.num_classes

        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=cfg.train.label_smoothing,
            ignore_index=self.ig,
        )
        self.dice = DiceLoss(nc, self.ig)

        if cfg.train.use_noise_robust_loss:
            self.bootstrap = BootstrappedCE(cfg.train.noise_beta, self.ig)
        else:
            self.bootstrap = None

    def forward(self, main, aux2, aux3, targets):
        # Ưu tiên dùng Bootstrap CE nếu có bật tính năng noise robust
        if self.bootstrap is not None:
            loss = self.ce_w * self.bootstrap(main, targets)
        else:
            loss = self.ce_w * self.ce(main, targets)
            
        loss = loss + self.dice_w * self.dice(main, targets)

        # Deep supervision
        if self.deep_sup and aux2 is not None:
            loss = loss \
                 + 0.3 * self.ce(aux2, targets) \
                 + 0.3 * self.ce(aux3, targets)

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
        dice = 2*diag / (h.sum(1) + h.sum(0) + 1e-8)
        freq = h.sum(1) / (h.sum() + 1e-8)
        return {
            'iou'  : iou,
            'miou' : iou[:-1].mean(),     # exclude BG
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
    total   = cfg.train.epochs * steps_per_epoch
    warmup  = cfg.train.warmup_epochs * steps_per_epoch
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
        imgs  = imgs.to(device).float()
        out, _, _ = model(imgs)
        probs     = F.softmax(out, dim=1)

        for i in range(imgs.shape[0]):
            if use_crf and CRF_AVAILABLE:
                img_np  = imgs[i].cpu().numpy().transpose(1,2,0)
                img_rgb = ((img_np * std + mean) * 255).clip(0,255).astype(np.uint8)
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
        if m.ndim == 3: m = m[:,:,0]
        m = np.clip(m, 0, num_classes-1)
        for c in range(num_classes):
            counts[c] += (m == c).sum()
    freq    = counts / (counts.sum() + 1e-8)
    median  = np.median(freq[freq > 1e-6])
    weights = np.where(freq > 1e-6, median / (freq + 1e-8), 1.0)
    
    # FIX: Hạ trần class weight xuống 4.0 để tránh Gradient bùng nổ cho class quá hiếm
    weights = np.clip(weights, 0.5, 4.0) 
    print(f"  Class weights: { {n:f'{w:.2f}' for n,w in zip(CLASS_NAMES, weights)} }")
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
    train_ds = LUADTrainSeg(cfg.dataset.train_root, cfg.dataset.pseudo_dir,
                            transform=get_train_aug(mean, std), ignore_index=cfg.dataset.ignore_index)
    val_ds   = LUADValSeg(cfg.dataset.val_root,  'val',  transform=get_val_aug(mean, std))
    test_ds  = LUADValSeg(cfg.dataset.test_root, 'test', transform=get_val_aug(mean, std))

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, 
                              num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    print("\nBuilding UNI + UPerNet model...")
    model = UNISegNet(cfg).to(device)
    model.encoder.model.eval() 

    print("\nComputing class weights...")
    class_weights = compute_class_weights(cfg.dataset.pseudo_dir, cfg.dataset.num_classes).to(device)
    criterion = SegLoss(cfg, class_weights).to(device)

    param_groups = model.get_param_groups(lr=cfg.train.lr, backbone_scale=cfg.train.backbone_lr_scale)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)
    
    # FIX: Chia cho accum_steps để Scheduler chạy đúng lộ trình epoch
    actual_steps_per_epoch = max(1, len(train_loader) // cfg.train.accum_steps)
    scheduler = build_scheduler(optimizer, cfg, actual_steps_per_epoch)
    scaler    = torch.amp.GradScaler('cuda')

    best_val_miou  = 0.0
    best_test_miou = 0.0
    accum          = cfg.train.accum_steps

    print(f"\nTraining {cfg.train.epochs} epochs | BS={cfg.train.batch_size} accum={accum} | Steps/epoch={len(train_loader)}")
    print("=" * 70)

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        model.encoder.model.eval() 
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, desc=f'Ep {epoch:02d}/{cfg.train.epochs}')

        for step, (imgs, masks) in pbar:
            imgs  = imgs.to(device).float()
            masks = masks.to(device)

            # Cập nhật syntax Autocast chuẩn không bị deprecate
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                main, aux2, aux3 = model(imgs)
                loss = criterion(main, aux2, aux3, masks) / accum

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
        print(f"         {' | '.join(f'{n}:{v*100:.1f}%' for n,v in zip(CLASS_NAMES, val_m['iou']))}")
        print(f"  [TEST] mIoU={test_m['miou']*100:.2f}% | fwIoU={test_m['fwiou']*100:.2f}% | Dice={test_m['dice']*100:.2f}%")
        print(f"         {' | '.join(f'{n}:{v*100:.1f}%' for n,v in zip(CLASS_NAMES, test_m['iou']))}")

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
    ckpt = torch.load(os.path.join(cfg.work_dir.ckpt_dir, 'best_seg.pth'), map_location=device, weights_only=False)
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