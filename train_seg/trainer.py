# ============================================================
# Main Training Loop — Multi-backbone Segmentation Fine-tuning
# ============================================================

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CLASS_NAMES
from .model import build_seg_model
from .dataset import BCSSTrainSeg, BCSSValSeg, get_train_aug, get_val_aug
from .losses import SegLoss
from .scheduler import build_scheduler
from .crf import CRF_AVAILABLE
from .utils import compute_class_weights, validate, fix_double_ext


def train_seg(cfg):
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.log_dir, exist_ok=True)

    random.seed(42); np.random.seed(42)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

    mean = list(cfg.dataset.mean)
    std = list(cfg.dataset.std)

    # Fix double-extension pseudo labels nếu cần
    fix_double_ext(cfg.dataset.pseudo_dir)

    print("\nLoading datasets...")
    crop_size = cfg.train.get('crop_size', 512)
    train_ds = BCSSTrainSeg(
        img_root=cfg.dataset.train_root,
        pseudo_dir=cfg.dataset.pseudo_dir,
        transform=get_train_aug(mean, std, size=crop_size),
        ignore_index=cfg.dataset.ignore_index,
    )
    val_ds = BCSSValSeg(cfg.dataset.val_root, split='val', transform=get_val_aug(mean, std))
    test_ds = BCSSValSeg(cfg.dataset.test_root, split='test', transform=get_val_aug(mean, std))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    model = build_seg_model(cfg).to(device)

    # Giữ BN encoder ở eval mode
    def _freeze_encoder_bn(m):
        model_type = cfg.model.get('type', 'segformer')
        if model_type == 'segformer':
            m.model.segformer.eval()
        elif model_type == 'phikon':
            m.backbone.eval()
        elif model_type == 'ctranspath':
            m.backbone.eval()
    _freeze_encoder_bn(model)

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
    scaler = torch.amp.GradScaler('cuda')

    best_val_miou = 0.0
    best_test_miou = 0.0

    print(f"\nTraining {cfg.train.epochs} epochs | BS={cfg.train.batch_size} accum={accum} | Steps/epoch={len(train_loader)}")
    print("=" * 70)

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        _freeze_encoder_bn(model)
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            ncols=100,
            desc=f'Ep {epoch:02d}/{cfg.train.epochs}',
        )

        for step, (imgs, masks) in pbar:
            imgs = imgs.to(device).float()
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

        # use_crf = getattr(cfg.train, 'use_crf_val', False)
        use_crf = False
        val_m = validate(model, val_loader, device, cfg, use_crf)
        test_m = validate(model, test_loader, device, cfg, use_crf)

        print(f"\nEp {epoch:02d} | Loss={avg_loss:.4f} | LR={optimizer.param_groups[1]['lr']:.2e}")
        print(f"  [VAL]  mIoU={val_m['miou']*100:.2f}% | fwIoU={val_m['fwiou']*100:.2f}% | Dice={val_m['dice']*100:.2f}%")
        print(f"         {' | '.join(f'{n}:{v*100:.1f}%' for n, v in zip(CLASS_NAMES, val_m['iou']))}")
        print(f"  [TEST] mIoU={test_m['miou']*100:.2f}% | fwIoU={test_m['fwiou']*100:.2f}% | Dice={test_m['dice']*100:.2f}%")
        print(f"         {' | '.join(f'{n}:{v*100:.1f}%' for n, v in zip(CLASS_NAMES, test_m['iou']))}")

        if val_m['miou'] > best_val_miou:
            best_val_miou = val_m['miou']
            best_test_miou = test_m['miou']
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_miou': best_val_miou,
                'test_miou': best_test_miou,
                'cfg': cfg,
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
