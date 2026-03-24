"""
Training script for WSSS network
"""
import os
import sys
import datetime
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
from medclip import MedCLIPModel, MedCLIPVisionModelViT

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.bcss_dataset import get_cls_dataset
from src.models.network import ClsNetwork
from src.models.hash_miner import HashHardNegativeMiner
from src.losses.contrast_loss import InfoNCELossFG, InfoNCELossBG
from src.utils.pyutils import set_seed, AverageMeter
from src.utils.optimizer import PolyWarmupAdamW
from src.utils.evaluate import validate
from src.utils.hierarchical import merge_to_parent_predictions, expand_parent_to_subclass_labels, pair_features
from src.utils.fgbg_feature import FeatureExtractor, MaskAdapter_DynamicThreshold


def cal_eta(time0, cur_iter, total_iter):
    """Calculate elapsed time and ETA."""
    import datetime
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    delta = time_now - time0
    time_per_iter = delta.total_seconds() / max(cur_iter, 1)
    remaining_iters = total_iter - cur_iter
    eta = datetime.timedelta(seconds=int(time_per_iter * remaining_iters))
    return str(delta), str(eta)


def train(cfg, args):
    """Main training function."""
    print("\nInitializing training...")
    torch.backends.cudnn.benchmark = True
    
    num_workers = min(10, os.cpu_count())
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)
    
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

    # Optimizer
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

    # Loss functions
    loss_function = nn.BCEWithLogitsLoss().to(device)
    mask_adapter = MaskAdapter_DynamicThreshold(alpha=cfg.train.mask_adapter_alpha)
    feature_extractor = FeatureExtractor(mask_adapter=mask_adapter)
    fg_loss_fn = InfoNCELossFG(temperature=0.07).to(device)
    bg_loss_fn = InfoNCELossBG(temperature=0.07).to(device)
    
    # Hash miner (optional)
    hash_miner = None
    if cfg.train.use_binary_hash:
        hash_miner = HashHardNegativeMiner(
            in_dim=512, hash_dim=cfg.train.hash_dim,
            top_k=cfg.train.hash_top_k, quant_weight=cfg.train.hash_quant_weight,
            bank_capacity=cfg.train.hash_bank_capacity
        ).to(device)
    
    best_fuse234_dice = 0.0
    epochs_no_improve = 0
    patience = getattr(cfg.train, 'patience', 3)
    use_early_stop = getattr(cfg.train, 'use_early_stop', False)

    print("\nStarting training...")
    train_loader_iter = iter(train_loader)
    
    for n_iter in range(cfg.train.max_iters):
        optimizer.zero_grad(set_to_none=True)
        total_loss_for_log = 0.0

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

                cls1_merge = merge_to_parent_predictions(cls1, k_list, method=cfg.train.merge_train)
                cls2_merge = merge_to_parent_predictions(cls2, k_list, method=cfg.train.merge_train)
                cls3_merge = merge_to_parent_predictions(cls3, k_list, method=cfg.train.merge_train)
                cls4_merge = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_train)

                subclass_labels = expand_parent_to_subclass_labels(cls_labels, k_list)
                cls4_expand = expand_parent_to_subclass_labels(cls4_merge, k_list)
                cls4_bir = (cls4 > cls4_expand).float() * subclass_labels

                # Extract FG/BG features
                batch_info = feature_extractor.process_batch(inputs, cam4, cls4_bir, clip_model)
                if batch_info is not None:
                    fg_features, bg_features = batch_info['fg_features'], batch_info['bg_features']
                    set_info = pair_features(fg_features, bg_features, l_fea, cls4_bir)
                    fg_features, bg_features = set_info['fg_features'], set_info['bg_features']
                    fg_pro, bg_pro = set_info['fg_text'], set_info['bg_text']

                    fg_loss = fg_loss_fn(fg_features, fg_pro, bg_pro)
                    bg_loss = bg_loss_fn(bg_features, fg_pro, bg_pro)

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

            scaler.scale(scaled_loss).backward()
            total_loss_for_log += raw_loss.item() / accum_steps

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
        
        # Regular validation
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

            # Save best model
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
                print(f"No improvement for {epochs_no_improve} eval(s). Best: {best_fuse234_dice:.4f}")

            if use_early_stop and epochs_no_improve >= patience:
                print(f"\nEarly stopping after {patience} evaluations without improvement.")
                break

    torch.cuda.empty_cache()
    print(f'\nTotal training time: {datetime.datetime.now() - time0}')
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device id')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.ckpt_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.train_log_dir, exist_ok=True)

    print('\nConfigs: %s' % cfg)
    set_seed(0)
    train(cfg, args)


if __name__ == "__main__":
    main()
