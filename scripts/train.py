"""
Training script for WSSS network
"""
import os
import sys
import json
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from medclip import MedCLIPModel, MedCLIPVisionModelViT

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets import get_cls_dataset
from src.engine import generate_cam
from src.models import ClsNetwork
from src.losses import (
    InfoNCELossFG,
    InfoNCELossBG,
    BackgroundPromptSuppressionLoss,
)
from src.utils import (
    set_seed,
    PolyWarmupAdamW,
    validate,
    expand_parent_to_subclass_labels,
    pair_features,
    FeatureExtractor,
    MaskAdapter_DynamicThreshold,
)


def cal_eta(time0, cur_iter, total_iter):
    """Calculate elapsed time and estimated time to completion."""
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def linear_warmup_factor(progress, warmup_progress):
    """Linearly ramp from 0 to 1 over the requested warmup progress."""
    if warmup_progress is None or warmup_progress <= 0:
        return 1.0
    return float(min(1.0, progress / warmup_progress))


def train(cfg, args):
    """Main training function."""
    print("\nInitializing training...")
    torch.backends.cudnn.benchmark = True
    
    num_workers = min(10, os.cpu_count())
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(args.seed)
    
    # Preload CLIP model to GPU
    clip_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    # ============ prepare data ============
    print("\nPreparing datasets...")
    train_dataset, val_dataset = get_cls_dataset(cfg, split="val", val_root=cfg.dataset.val_root)
    _, test_dataset = get_cls_dataset(cfg, split="test", val_root=cfg.dataset.test_root, enable_rotation=False, p=0.0)

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
                    use_text_proto=getattr(cfg.train, 'use_text_proto', False),
                    dataset_name=cfg.dataset.name,
                    text_prompt_mode=getattr(cfg.train, 'text_prompt_mode', 'pathology'),
                    text_encoder_mode=getattr(cfg.train, 'text_encoder_mode', 'auto'),
                    use_hybrid_proto_blend=getattr(cfg.train, 'use_hybrid_proto_blend', False),
                    hybrid_alpha_mode=getattr(cfg.train, 'hybrid_alpha_mode', 'learnable'),
                    hybrid_alpha_fixed=getattr(cfg.train, 'hybrid_alpha_fixed', 0.5),
                    hybrid_alpha_parent_values=getattr(cfg.train, 'hybrid_alpha_parent_values', None),
                    use_bg_prompt_suppression=getattr(cfg.train, 'use_bg_prompt_suppression', False),
                    bg_prompt_mode=getattr(cfg.train, 'bg_prompt_mode', 'pathology'))
    
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_p/1e6:.1f}M total, {train_p/1e6:.2f}M trainable ({train_p/total_p*100:.2f}%)")
    
    # Setup training parameters
    iters_per_epoch = len(train_loader)
    accum_steps = getattr(cfg.train, 'accum_steps', 1)
    steps_per_epoch = max(1, iters_per_epoch // accum_steps)
    cfg.train.max_iters = cfg.train.epoch * steps_per_epoch
    cfg.train.eval_iters = steps_per_epoch
    cfg.scheduler.warmup_iter = cfg.scheduler.warmup_iter * steps_per_epoch
    scaler = torch.amp.GradScaler('cuda')
    
    model.to(device)
    model.train()
    model.encoder.model.eval()  # keep frozen UNI in eval mode

    # Optimizer configuration
    base_lr = cfg.optimizer.learning_rate
    optimizer_params = [{"params": [p for p in model.parameters() if p.requires_grad]}]

    optimizer = PolyWarmupAdamW(
        params=optimizer_params,
        lr=base_lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )

    # Loss functions
    loss_function = nn.BCEWithLogitsLoss().to(device)
    mask_adapter = MaskAdapter_DynamicThreshold(alpha=cfg.train.mask_adapter_alpha)
    feature_extractor = FeatureExtractor(mask_adapter=mask_adapter)
    
    nce_temp = getattr(cfg.train, 'info_nce_temp', 0.07)
    fg_loss_fn = InfoNCELossFG(temperature=nce_temp).to(device)
    bg_loss_fn = InfoNCELossBG(temperature=nce_temp).to(device)
    use_bg_prompt_suppression = getattr(cfg.train, 'use_bg_prompt_suppression', False)
    if use_bg_prompt_suppression:
        bg_prompt_loss_fn = BackgroundPromptSuppressionLoss(
            temperature=getattr(cfg.train, 'bg_prompt_temp', 0.07),
            margin=getattr(cfg.train, 'bg_prompt_margin', 0.10)
        ).to(device)
    else:
        bg_prompt_loss_fn = None
    
    best_fuse234_dice = 0.0
    epochs_no_improve = 0
    patience = getattr(cfg.train, 'patience', 3)
    use_early_stop = getattr(cfg.train, 'use_early_stop', False)

    print("\nStarting training...")
    train_loader_iter = iter(train_loader)
    
    for n_iter in range(cfg.train.max_iters):
        optimizer.zero_grad(set_to_none=True)
        total_loss_for_log = 0.0
        bg_prompt_total_for_log = 0.0
        bg_align_for_log = 0.0
        bg_repel_for_log = 0.0
        bg_stability_for_log = 0.0

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

                cls1_merge = model.aggregate_parent_predictions(cls1, method=cfg.train.merge_train)
                cls2_merge = model.aggregate_parent_predictions(cls2, method=cfg.train.merge_train)
                cls3_merge = model.aggregate_parent_predictions(cls3, method=cfg.train.merge_train)
                cls4_merge = model.aggregate_parent_predictions(cls4, method=cfg.train.merge_train)

                subclass_labels = expand_parent_to_subclass_labels(cls_labels, k_list)
                cls4_expand = expand_parent_to_subclass_labels(cls4_merge, k_list)
                cls4_bir = (cls4 > cls4_expand).float() * subclass_labels

                # Extract FG/BG features
                batch_info = feature_extractor.process_batch(inputs, cam4, cls4_bir, clip_model)
                if batch_info is not None:
                    fg_features = batch_info['fg_features']
                    bg_features = batch_info['bg_features']
                    set_info = pair_features(fg_features, bg_features, l_fea, cls4_bir)
                    fg_features, bg_features = set_info['fg_features'], set_info['bg_features']
                    fg_pro, bg_pro = set_info['fg_text'], set_info['bg_text']

                    fg_loss = fg_loss_fn(fg_features, fg_pro, bg_pro)
                    bg_loss = bg_loss_fn(bg_features, fg_pro, bg_pro)

                    if use_bg_prompt_suppression:
                        parent_indices = model.subclass_to_parent[set_info['labels']].to(device)
                        bg_prompt_bank = model.get_background_prompt_bank(device)
                        bg_prompt_selected = bg_prompt_bank[parent_indices]
                        bg_prompt_anchor = model.bg_prompt_init.to(device)[parent_indices]
                        bg_prompt_losses, bg_prompt_stats = bg_prompt_loss_fn(
                            fg_img_feature=fg_features,
                            bg_img_feature=bg_features,
                            fg_proto_feature=fg_pro,
                            bg_prompt_feature=bg_prompt_selected,
                            parent_indices=parent_indices,
                            all_bg_prompt_bank=bg_prompt_bank,
                            bg_prompt_anchor=bg_prompt_anchor,
                        )
                        bg_prompt_total = (
                            getattr(cfg.train, 'bg_prompt_align_weight', 1.0) * bg_prompt_losses['align']
                            + getattr(cfg.train, 'bg_prompt_repel_weight', 0.5) * bg_prompt_losses['repel']
                            + getattr(cfg.train, 'bg_prompt_stability_weight', 0.1) * bg_prompt_losses['stability']
                        )
                    else:
                        bg_prompt_total = torch.tensor(0.0, device=device)
                        bg_prompt_stats = None
                else:
                    fg_loss = torch.tensor(0.0, device=device)
                    bg_loss = torch.tensor(0.0, device=device)
                    bg_prompt_total = torch.tensor(0.0, device=device)
                    bg_prompt_stats = None

                # Multi-scale classification losses
                loss1 = loss_function(cls1_merge, cls_labels)
                loss2 = loss_function(cls2_merge, cls_labels)
                loss3 = loss_function(cls3_merge, cls_labels)
                loss4 = loss_function(cls4_merge, cls_labels)
                cls_loss = getattr(cfg.train, 'l1', 0.1) * loss1 + cfg.train.l2 * loss2 + cfg.train.l3 * loss3 + cfg.train.l4 * loss4
                
                _use_nce = getattr(cfg.train, 'use_contrastive', True)
                _nce_w = 1.0 if _use_nce else 0.0
                bg_prompt_schedule = linear_warmup_factor(
                    progress=(n_iter + 1) / max(steps_per_epoch, 1),
                    warmup_progress=getattr(cfg.train, 'bg_prompt_warmup_epochs', 0.0),
                )
                bg_prompt_weight = getattr(cfg.train, 'bg_prompt_weight', 0.0) * bg_prompt_schedule
                raw_loss = cls_loss + (_nce_w * fg_loss + _nce_w * bg_loss + 0.0005*torch.mean(cam4)) * getattr(cfg.train, 'l5', 1.0)
                raw_loss = raw_loss + bg_prompt_weight * bg_prompt_total
                
                scaled_loss = raw_loss / accum_steps

            scaler.scale(scaled_loss).backward()
            total_loss_for_log += raw_loss.item() / accum_steps
            if bg_prompt_stats is not None:
                bg_prompt_total_for_log += (bg_prompt_weight * bg_prompt_total).item() / accum_steps
                bg_align_for_log += bg_prompt_stats['bg_align'].item() / accum_steps
                bg_repel_for_log += bg_prompt_stats['bg_repel'].item() / accum_steps
                bg_stability_for_log += bg_prompt_stats['bg_stability'].item() / accum_steps

        scaler.step(optimizer)
        scaler.update()

        # Logging
        if (n_iter + 1) % 100 == 0:
            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']
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
            if use_bg_prompt_suppression:
                print(
                    f"       BGPrompt: total={bg_prompt_total_for_log:.4f} | "
                    f"align={bg_align_for_log:.4f} | repel={bg_repel_for_log:.4f} | "
                    f"stability={bg_stability_for_log:.4f} | "
                    f"w={bg_prompt_weight:.4f}"
                )
            if getattr(cfg.train, 'use_text_proto', False) and getattr(cfg.train, 'use_hybrid_proto_blend', False):
                alpha_analysis = model.get_hybrid_alpha_analysis()
                if alpha_analysis is not None:
                    alpha_values = " ".join(
                        f"{name}={value:.3f}" for name, value in alpha_analysis["values"].items()
                    )
                    print(
                        f"       HybridAlpha[{alpha_analysis['mode']}]: {alpha_values} | "
                        f"mean={alpha_analysis['mean']:.3f}"
                    )
        # Regular validation and model saving
        if (n_iter + 1) % cfg.train.eval_iters == 0 or (n_iter + 1) == cfg.train.max_iters:
            epoch_num = (n_iter + 1) // cfg.train.eval_iters
            model._current_epoch = epoch_num
            print(f"\n{'='*60}")
            print(f"Epoch {epoch_num} Evaluation")
            print(f"{'='*60}")

            val_all_acc4, val_avg_acc4, val_metrics, val_cls_loss = validate(
                model=model, data_loader=val_loader, cfg=cfg, cls_loss_func=loss_function
            )
            print(f"[VAL]  mIoU: {val_metrics['miou']*100:.2f}% | fwIoU: {val_metrics['fwiou']*100:.2f}% | "
                  f"BdIoU: {val_metrics['boundary_iou']*100:.2f}% | Dice: {val_metrics['dice']*100:.2f}%")
            print(f"       per-class IoU: {[f'{s*100:.2f}' for s in val_metrics['iou'][:-1].tolist()]}")

            test_all_acc4, test_avg_acc4, test_metrics, test_cls_loss = validate(
                model=model, data_loader=test_loader, cfg=cfg, cls_loss_func=loss_function
            )
            print(f"[TEST] mIoU: {test_metrics['miou']*100:.2f}% | fwIoU: {test_metrics['fwiou']*100:.2f}% | "
                  f"BdIoU: {test_metrics['boundary_iou']*100:.2f}% | Dice: {test_metrics['dice']*100:.2f}%")
            print(f"       per-class IoU: {[f'{s*100:.2f}' for s in test_metrics['iou'][:-1].tolist()]}")

            if use_bg_prompt_suppression:
                analysis = model.get_background_prompt_analysis()
                if analysis is not None:
                    print(
                        "  BG prompt analysis: "
                        f"drift_mean={analysis['drift_mean']:.4f} | "
                        f"drift_max={analysis['drift_max']:.4f} | "
                        f"inter_mean={analysis['inter_mean']:.4f} | "
                        f"inter_max={analysis['inter_max']:.4f}"
                    )
            if getattr(cfg.train, 'use_text_proto', False) and getattr(cfg.train, 'use_hybrid_proto_blend', False):
                alpha_analysis = model.get_hybrid_alpha_analysis()
                if alpha_analysis is not None:
                    alpha_values = " ".join(
                        f"{name}={value:.4f}" for name, value in alpha_analysis["values"].items()
                    )
                    print(
                        f"  Hybrid alpha analysis ({alpha_analysis['mode']}): {alpha_values} | "
                        f"mean={alpha_analysis['mean']:.4f} | "
                        f"min={alpha_analysis['min']:.4f} | max={alpha_analysis['max']:.4f}"
                    )

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

            stop_after_epoch = getattr(args, "stop_after_epoch", None)
            if stop_after_epoch is not None and epoch_num >= stop_after_epoch:
                print(f"\nStopping after epoch {epoch_num} as requested (--stop-after-epoch={stop_after_epoch}).")
                break

    torch.cuda.empty_cache()
    end_time = datetime.datetime.now()
    total_training_time = end_time - time0
    print(f'Total training time: {total_training_time}')

    if getattr(cfg.train, 'skip_post_train_eval', False):
        print("Skipping post-training evaluation and CAM generation (skip_post_train_eval=true).")
        return model

    print("\n" + "="*80)
    print("POST-TRAINING EVALUATION AND CAM GENERATION")
    print("="*80)

    print("\n1. Evaluating on VAL set...")
    print("-" * 50)
    _, val_dataset_eval = get_cls_dataset(cfg, split="val", val_root=cfg.dataset.val_root, enable_rotation=False, p=0.0)
    val_eval_loader = DataLoader(val_dataset_eval, batch_size=8, shuffle=False,
                                 num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_all_acc4_final, val_avg_acc4_final, val_metrics_final, val_cls_loss_final = validate(
        model=model, data_loader=val_eval_loader, cfg=cfg, cls_loss_func=loss_function)
    val_fuse234_score = val_metrics_final['iou']
    print("Val results:")
    print(f"  all_acc4: {val_all_acc4_final:.6f} | avg_acc4: {val_avg_acc4_final:.6f}")
    print(f"  mIOU: {val_fuse234_score[:-1].mean():.4f} | scores: {val_fuse234_score}")
    print("  Per-class IoU:")
    for i, score in enumerate(val_fuse234_score[:-1]):
        print(f"    Class {i}: {score:.6f}")
    if len(val_fuse234_score) > cfg.dataset.cls_num_classes:
        print(f"    Background: {val_fuse234_score[-1]:.6f}")

    print("\n2. Evaluating on TEST set...")
    print("-" * 50)
    _, test_dataset = get_cls_dataset(cfg, split="test", val_root=cfg.dataset.test_root, enable_rotation=False, p=0.0)
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_all_acc4, test_avg_acc4, test_metrics_final, test_cls_loss = validate(
        model=model, data_loader=test_loader, cfg=cfg, cls_loss_func=loss_function)
    fuse234_score = test_metrics_final['iou']
    print("Test results:")
    print(f"  all_acc4: {test_all_acc4:.6f} | avg_acc4: {test_avg_acc4:.6f}")
    print(f"  mIOU: {fuse234_score[:-1].mean():.4f} | scores: {fuse234_score}")
    print("  Per-class IoU:")
    for i, score in enumerate(fuse234_score[:-1]):
        print(f"    Class {i}: {score:.6f}")
    if len(fuse234_score) > cfg.dataset.cls_num_classes:
        print(f"    Background: {fuse234_score[-1]:.6f}")

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

    generate_cam(model=model, data_loader=train_cam_loader, cfg=cfg, device=device)

    print("\nFiles generated:")
    print(f"  • Training CAM visualizations: {cfg.work_dir.pred_dir}/*.png")
    print(f"  • Model checkpoint: {cfg.work_dir.ckpt_dir}/best_cam.pth")
    print("="*80)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device id')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Override cfg.train.epoch without editing the config')
    parser.add_argument('--stop-after-epoch', type=int, default=None,
                        help='Stop training after this evaluated epoch, then continue to post-training CAM generation unless skipped')
    parser.add_argument('--generate-pseudo', action='store_true',
                        help='Force post-training evaluation and pseudo-mask generation even if the config skips it')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.epoch is not None:
        cfg.train.epoch = int(args.epoch)
    if args.generate_pseudo:
        cfg.train.skip_post_train_eval = False

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.ckpt_dir, f"{timestamp}_seed{args.seed}")

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.train_log_dir, exist_ok=True)

    print(f'\nConfigs: {cfg}')
    print(f'Seed: {args.seed}')
    set_seed(args.seed)
    train(cfg, args)


if __name__ == "__main__":
    main()
