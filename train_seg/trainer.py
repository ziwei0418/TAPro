# ============================================================
# Main Training Loop — Multi-backbone Segmentation Fine-tuning
# ============================================================

import os
import sys
import random
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import build_seg_model
from .dataset import get_datasets
from .losses import SegLoss
from .scheduler import build_scheduler
from .crf import CRF_AVAILABLE
from .utils import compute_class_weights, validate, fix_double_ext, regenerate_pseudo_labels


def cutmix_seg(imgs, masks, alpha=1.0):
    """CutMix for segmentation: paste a random patch from one sample onto another.
    Both image and mask are cut identically.
    """
    B, C, H, W = imgs.shape
    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(B, device=imgs.device)

    # Random box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cy = np.random.randint(H)
    cx = np.random.randint(W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    imgs_mix = imgs.clone()
    masks_mix = masks.clone()
    imgs_mix[:, :, y1:y2, x1:x2] = imgs[indices, :, y1:y2, x1:x2]
    masks_mix[:, y1:y2, x1:x2] = masks[indices, y1:y2, x1:x2]
    return imgs_mix, masks_mix


class ModelEMA:
    """Exponential Moving Average of model weights for more stable predictions."""
    def __init__(self, model, decay=0.999):
        import copy
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)
        for ema_b, model_b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)


class Logger:
    """Tee stdout to both console and log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'a', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def train_seg(cfg):
    """Main training loop with file logging."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.log_dir, exist_ok=True)

    # Setup file logging
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(cfg.work_dir.log_dir, f'training_{timestamp}.txt')
    sys.stdout = Logger(log_path)
    print(f"Logging to: {log_path}")
    print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    random.seed(42); np.random.seed(42)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

    mean = list(cfg.dataset.mean)
    std = list(cfg.dataset.std)

    # Fix double-extension pseudo labels nếu cần
    fix_double_ext(cfg.dataset.pseudo_dir)

    class_names = list(cfg.dataset.class_names)
    print(f"\nLoading datasets... ({cfg.dataset.name.upper()}: {class_names})")
    train_ds, val_ds, test_ds = get_datasets(cfg, mean, std)

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
    opt_name = str(cfg.train.get('optimizer', 'adamw')).lower()
    if opt_name == 'ano':
        from anon import Ano
        optimizer = Ano(
            param_groups,
            lr=cfg.train.lr,
            betas=(cfg.train.get('ano_beta1', 0.92), cfg.train.get('ano_beta2', 0.99)),
            weight_decay=cfg.train.weight_decay,
            logarithmic_schedule=cfg.train.get('ano_logarithmic_schedule', False),
        )
        print(
            "Optimizer: Ano "
            f"(betas={cfg.train.get('ano_beta1', 0.92):.3f},{cfg.train.get('ano_beta2', 0.99):.3f}, "
            f"log_schedule={cfg.train.get('ano_logarithmic_schedule', False)})"
        )
    else:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)
        print("Optimizer: AdamW")

    accum = cfg.train.accum_steps
    actual_steps_per_epoch = max(1, len(train_loader) // accum)
    scheduler = build_scheduler(optimizer, cfg, actual_steps_per_epoch)
    scaler = torch.amp.GradScaler('cuda')

    # EMA
    use_ema = cfg.train.get('use_ema', True)
    ema = ModelEMA(model, decay=cfg.train.get('ema_decay', 0.999)) if use_ema else None
    if use_ema:
        print(f"EMA enabled (decay={cfg.train.get('ema_decay', 0.999)})")

    best_val_miou = 0.0
    best_test_miou = 0.0
    best_ema_val_miou = 0.0
    patience = cfg.train.get('patience', 10)
    epochs_no_improve = 0

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

            # CutMix augmentation (50% chance)
            use_cutmix = cfg.train.get('use_cutmix', True)
            if use_cutmix and random.random() < 0.5:
                imgs, masks = cutmix_seg(imgs, masks, alpha=1.0)

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
                if ema is not None:
                    ema.update(model)

            pbar.set_postfix({
                'loss': f'{total_loss/(step+1):.4f}',
                'lr_d': f'{optimizer.param_groups[1]["lr"]:.1e}',
            })

        avg_loss = total_loss / len(train_loader)

        use_tta = cfg.train.get('use_tta_val', False)
        val_m = validate(model, val_loader, device, cfg, use_tta=use_tta)
        test_m = validate(model, test_loader, device, cfg, use_tta=use_tta)

        print(f"\nEp {epoch:02d} | Loss={avg_loss:.4f} | LR={optimizer.param_groups[1]['lr']:.2e}")
        print(f"  [VAL]  mIoU={val_m['miou']*100:.2f}% | fwIoU={val_m['fwiou']*100:.2f}% | Dice={val_m['dice']*100:.2f}%")
        print(f"         {' | '.join(f'{n}:{v*100:.1f}%' for n, v in zip(class_names, val_m['iou']))}")
        print(f"  [TEST] mIoU={test_m['miou']*100:.2f}% | fwIoU={test_m['fwiou']*100:.2f}% | Dice={test_m['dice']*100:.2f}%")
        print(f"         {' | '.join(f'{n}:{v*100:.1f}%' for n, v in zip(class_names, test_m['iou']))}")

        # EMA validation
        if ema is not None:
            ema_val_m = validate(ema.ema_model, val_loader, device, cfg, use_tta=use_tta)
            ema_test_m = validate(ema.ema_model, test_loader, device, cfg, use_tta=use_tta)
            print(f"  [EMA-VAL]  mIoU={ema_val_m['miou']*100:.2f}% | Dice={ema_val_m['dice']*100:.2f}%")
            print(f"  [EMA-TEST] mIoU={ema_test_m['miou']*100:.2f}% | Dice={ema_test_m['dice']*100:.2f}%")
            if ema_val_m['miou'] > best_ema_val_miou:
                best_ema_val_miou = ema_val_m['miou']
                torch.save({
                    'epoch': epoch,
                    'model': ema.ema_model.state_dict(),
                    'val_miou': best_ema_val_miou,
                    'test_miou': ema_test_m['miou'],
                    'cfg': cfg,
                }, os.path.join(cfg.work_dir.ckpt_dir, 'best_ema.pth'))
                print(f"  ✓ EMA best saved — val={best_ema_val_miou*100:.2f}% test={ema_test_m['miou']*100:.2f}%")

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
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  No improve for {epochs_no_improve}/{patience} epochs. Best val={best_val_miou*100:.2f}%")
        print("=" * 70)

        if epochs_no_improve >= patience:
            print(f"\n⏹ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Final evaluation: both regular and EMA
    print("\nFinal evaluation with TTA + best checkpoint...")
    ckpt = torch.load(
        os.path.join(cfg.work_dir.ckpt_dir, 'best_seg.pth'),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt['model'])

    for split, loader in [('VAL', val_loader), ('TEST', test_loader)]:
        m = validate(model, loader, device, cfg, use_tta=True)
        print(f"\n[{split}+TTA] mIoU={m['miou']*100:.2f}% | Dice={m['dice']*100:.2f}% | fwIoU={m['fwiou']*100:.2f}%")
        for name, iou in zip(class_names, m['iou']):
            print(f"  {name:4s}: {iou*100:.2f}%")

    if ema is not None:
        ema_path = os.path.join(cfg.work_dir.ckpt_dir, 'best_ema.pth')
        if os.path.exists(ema_path):
            ema_ckpt = torch.load(ema_path, map_location=device, weights_only=False)
            model.load_state_dict(ema_ckpt['model'])
            print("\n--- EMA Model Final ---")
            for split, loader in [('VAL', val_loader), ('TEST', test_loader)]:
                m = validate(model, loader, device, cfg, use_tta=True)
                print(f"[EMA-{split}+TTA] mIoU={m['miou']*100:.2f}% | Dice={m['dice']*100:.2f}% | fwIoU={m['fwiou']*100:.2f}%")
                for name, iou in zip(class_names, m['iou']):
                    print(f"  {name:4s}: {iou*100:.2f}%")

    return model, device, cfg


def train_seg_with_self_training(cfg):
    """
    Self-training: train → re-generate pseudo labels → train again.
    Each round improves pseudo label quality.
    """
    n_rounds = cfg.train.get('self_training_rounds', 1)
    pseudo_dir = cfg.dataset.pseudo_dir

    for rnd in range(1, n_rounds + 1):
        print(f"\n{'#'*70}")
        print(f"# Self-Training Round {rnd}/{n_rounds}")
        print(f"# Pseudo dir: {pseudo_dir}")
        print(f"{'#'*70}")

        # Update pseudo_dir for this round
        from omegaconf import OmegaConf
        cfg_round = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_round.dataset.pseudo_dir = pseudo_dir

        model, device, _ = train_seg(cfg_round)

        # Re-generate pseudo labels for next round
        if rnd < n_rounds:
            new_pseudo_dir = pseudo_dir + f'_round{rnd}'
            regenerate_pseudo_labels(model, None, device, cfg_round, new_pseudo_dir)
            pseudo_dir = new_pseudo_dir

    return model
