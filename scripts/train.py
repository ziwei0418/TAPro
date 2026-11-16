"""
WSSS（弱监督语义分割）网络训练脚本。

整体流程：
1. 构建 train / val / test 数据集与分类网络；
2. 使用多尺度分类损失和前景/背景对比损失进行训练；
3. 定期在验证集与测试集上评估，并按验证集 mIoU 保存最优模型；
4. 训练结束后用最优模型为训练集生成 CAM（类激活图）伪标签。
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
    """计算已用时间和预计剩余时间（ETA），用于训练日志输出。"""
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def linear_warmup_factor(progress, warmup_progress):
    """线性 warmup：在 warmup 进度内从 0 线性升到 1；未配置 warmup 时直接返回 1.0。"""
    if warmup_progress is None or warmup_progress <= 0:
        return 1.0
    return float(min(1.0, progress / warmup_progress))


def train(cfg, args):
    """训练主流程：构建数据/模型/损失，执行训练循环与验证，并完成训练后的 CAM 生成。"""
    print("\nInitializing training...")
    torch.backends.cudnn.benchmark = True
    
    # Windows 下 multiprocessing worker 每个进程都会加载 torch/CUDA DLL，
    # 6 个 DataLoader 叠加常驻 worker 会把页面文件耗尽（WinError 1455）。
    # 这里强制单进程加载：稳定优先，代价是数据加载稍慢。
    num_workers = 0
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(args.seed)
    
    # 预先将 MedCLIP 加载到 GPU，并固定为 eval 模式（后续只做特征提取）
    clip_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    # 记录训练起始时间，用于估算 ETA
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    # ============ 准备数据集 ============
    print("\nPreparing datasets...")
    train_dataset, val_dataset = get_cls_dataset(cfg, split="val", val_root=cfg.dataset.val_root)
    _, test_dataset = get_cls_dataset(cfg, split="test", val_root=cfg.dataset.test_root, enable_rotation=False, p=0.0)

    # 训练集需要打乱并常驻 worker；验证/测试集保持顺序即可
    train_loader = DataLoader(train_dataset,
                            batch_size=cfg.train.samples_per_gpu,
                            num_workers=num_workers,
                            pin_memory=True,
                            shuffle=True)

    val_loader = DataLoader(val_dataset,
                          batch_size=8,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True)

    test_loader = DataLoader(test_dataset,
                           batch_size=8,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=True)
    print(f"✓ Val dataset: {len(val_dataset)} samples | Test dataset: {len(test_dataset)} samples")

    # ============ 构建模型 ============
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
                    bg_prompt_mode=getattr(cfg.train, 'bg_prompt_mode', 'pathology'),
                    use_text_anchored_mask=getattr(cfg.train, 'use_text_anchored_mask', False),
                    text_anchor_tau=getattr(cfg.train, 'text_anchor_tau', 0.1),
                    text_anchor_map_bias=getattr(cfg.train, 'text_anchor_map_bias', False),
                    text_anchor_sparsity_weight=getattr(cfg.train, 'text_anchor_sparsity_weight', 0.0))
    
    # 统计模型总参数量与可训练参数量
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_p/1e6:.1f}M total, {train_p/1e6:.2f}M trainable ({train_p/total_p*100:.2f}%)")
    
    # 训练调度参数：梯度累积后每个 epoch 的优化步数、总迭代数与评估间隔
    iters_per_epoch = len(train_loader)
    accum_steps = getattr(cfg.train, 'accum_steps', 1)
    steps_per_epoch = max(1, iters_per_epoch // accum_steps)
    cfg.train.max_iters = cfg.train.epoch * steps_per_epoch
    cfg.train.eval_iters = steps_per_epoch
    # warmup 配置以 epoch 为单位，这里换算成优化步数
    cfg.scheduler.warmup_iter = cfg.scheduler.warmup_iter * steps_per_epoch
    # 混合精度训练的梯度缩放器，用于防止 FP16 下梯度下溢
    scaler = torch.amp.GradScaler('cuda')
    
    model.to(device)
    model.train()
    model.encoder.model.eval()  # 冻结的 UNI 编码器保持在 eval 模式（不更新 BN 统计量）

    # 优化器配置：仅优化 requires_grad 为 True 的参数（冻结部分不参与更新）
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

    # 损失函数：多尺度分类使用 BCE；前景/背景使用 InfoNCE；可选背景提示抑制损失
    loss_function = nn.BCEWithLogitsLoss().to(device)
    mask_adapter = MaskAdapter_DynamicThreshold(alpha=cfg.train.mask_adapter_alpha)
    feature_extractor = FeatureExtractor(mask_adapter=mask_adapter)
    
    nce_temp = getattr(cfg.train, 'info_nce_temp', 0.07)
    fg_loss_fn = InfoNCELossFG(temperature=nce_temp).to(device)  # 前景对比损失
    bg_loss_fn = InfoNCELossBG(temperature=nce_temp).to(device)  # 背景对比损失
    use_bg_prompt_suppression = getattr(cfg.train, 'use_bg_prompt_suppression', False)
    # 若启用背景提示词抑制，则实例化其损失函数；temperature 控制对比的锐利度，margin 控制背景与前景原型的间隔
    if use_bg_prompt_suppression:
        bg_prompt_loss_fn = BackgroundPromptSuppressionLoss(
            temperature=getattr(cfg.train, 'bg_prompt_temp', 0.07),  # 对比损失温度系数
            margin=getattr(cfg.train, 'bg_prompt_margin', 0.10)      # 背景-前景原型的间隔阈值
        ).to(device)
    else:
        bg_prompt_loss_fn = None  # 未启用时为 None，前向循环里会跳过该损失
    
    # 记录验证集最优 mIoU；epochs_no_improve 用于早停计数
    best_fuse234_dice = 0.0
    epochs_no_improve = 0
    patience = getattr(cfg.train, 'patience', 3)
    use_early_stop = getattr(cfg.train, 'use_early_stop', False)

    print("\nStarting training...")
    # 训练集迭代器：一个 epoch 取完所有 batch 后重新创建
    train_loader_iter = iter(train_loader)
    
    # 主训练循环：每个 iteration 处理 accum_steps 个 micro-batch 后更新一次参数
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

            # 前向计算使用自动混合精度（AMP），降低显存占用并加速
            with torch.amp.autocast('cuda'):
                cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list, aux_losses = model(inputs)

                # 将子类预测聚合为父类预测，以便与父类标签计算分类损失
                cls1_merge = model.aggregate_parent_predictions(cls1, method=cfg.train.merge_train)
                cls2_merge = model.aggregate_parent_predictions(cls2, method=cfg.train.merge_train)
                cls3_merge = model.aggregate_parent_predictions(cls3, method=cfg.train.merge_train)
                cls4_merge = model.aggregate_parent_predictions(cls4, method=cfg.train.merge_train)

                # 将父类标签扩展为子类标签；仅保留“子类预测强于父类预测”的位置作为伪标签
                subclass_labels = expand_parent_to_subclass_labels(cls_labels, k_list)
                cls4_expand = expand_parent_to_subclass_labels(cls4_merge, k_list)
                cls4_bir = (cls4 > cls4_expand).float() * subclass_labels

                # 提取前景/背景特征（借助 CLIP 模型与第 4 层 CAM）
                batch_info = feature_extractor.process_batch(inputs, cam4, cls4_bir, clip_model)
                if batch_info is not None:
                    fg_features = batch_info['fg_features']
                    bg_features = batch_info['bg_features']
                    set_info = pair_features(fg_features, bg_features, l_fea, cls4_bir)
                    # fg_pro / bg_pro 分别是前景类与背景类的文本原型特征
                    fg_features, bg_features = set_info['fg_features'], set_info['bg_features']
                    fg_pro, bg_pro = set_info['fg_text'], set_info['bg_text']

                    fg_loss = fg_loss_fn(fg_features, fg_pro, bg_pro)
                    bg_loss = bg_loss_fn(bg_features, fg_pro, bg_pro)

                    # 可选：背景提示抑制损失，由对齐、互斥、稳定性三项加权组成
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

                # 多尺度分类损失（4 个 FPN 层级的 BCE 加权求和）
                loss1 = loss_function(cls1_merge, cls_labels)
                loss2 = loss_function(cls2_merge, cls_labels)
                loss3 = loss_function(cls3_merge, cls_labels)
                loss4 = loss_function(cls4_merge, cls_labels)
                cls_loss = getattr(cfg.train, 'l1', 0.1) * loss1 + cfg.train.l2 * loss2 + cfg.train.l3 * loss3 + cfg.train.l4 * loss4
                
                # 对比损失开关；背景提示损失权重按 epoch 线性 warmup
                _use_nce = getattr(cfg.train, 'use_contrastive', True)
                _nce_w = 1.0 if _use_nce else 0.0
                bg_prompt_schedule = linear_warmup_factor(
                    progress=(n_iter + 1) / max(steps_per_epoch, 1),
                    warmup_progress=getattr(cfg.train, 'bg_prompt_warmup_epochs', 0.0),
                )
                bg_prompt_weight = getattr(cfg.train, 'bg_prompt_weight', 0.0) * bg_prompt_schedule
                raw_loss = cls_loss + (_nce_w * fg_loss + _nce_w * bg_loss + 0.0005*torch.mean(cam4)) * getattr(cfg.train, 'l5', 1.0)
                raw_loss = raw_loss + bg_prompt_weight * bg_prompt_total
                if aux_losses:
                    raw_loss = raw_loss + sum(aux_losses.values())

                # 梯度累积：将 loss 除以累积步数后再反向传播
                scaled_loss = raw_loss / accum_steps

            scaler.scale(scaled_loss).backward()
            total_loss_for_log += raw_loss.item() / accum_steps
            if bg_prompt_stats is not None:
                bg_prompt_total_for_log += (bg_prompt_weight * bg_prompt_total).item() / accum_steps
                bg_align_for_log += bg_prompt_stats['bg_align'].item() / accum_steps
                bg_repel_for_log += bg_prompt_stats['bg_repel'].item() / accum_steps
                bg_stability_for_log += bg_prompt_stats['bg_stability'].item() / accum_steps

        # 只有累计完成 accum_steps 个 micro-batch 后才执行一次参数更新
        scaler.step(optimizer)
        scaler.update()

        # 每 100 个 iteration 打印一次训练日志
        if (n_iter + 1) % 100 == 0:
            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']
            torch.cuda.synchronize()
            
            # 分类准确率：all 为整张图全部类别都预测正确，avg 为各类别准确率的平均
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
        # 定期验证与模型保存
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
            if getattr(cfg.train, 'use_text_anchored_mask', False):
                ta = model.get_text_anchor_analysis()
                if ta is not None:
                    subs = " ".join(f"{v:.3f}" for v in ta["per_subclass_mean"])
                    thr = " ".join(f"{v:.3f}" for v in ta["threshold"])
                    print(
                        f"  TextAnchor mask: mean={ta['mean']:.4f} | "
                        f"min={ta['min']:.4f} | max={ta['max']:.4f}"
                    )
                    print(f"    per-subclass mean: {subs}")
                    print(f"    threshold:         {thr}")

            # 根据验证集 mIoU 保存最优模型
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

            # 连续 patience 次评估都没有提升时提前结束训练
            if use_early_stop and epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {patience} evaluations without improvement.")
                break

            # 支持通过命令行限制最多训练到第几个 epoch
            stop_after_epoch = getattr(args, "stop_after_epoch", None)
            if stop_after_epoch is not None and epoch_num >= stop_after_epoch:
                print(f"\nStopping after epoch {epoch_num} as requested (--stop-after-epoch={stop_after_epoch}).")
                break

    # 训练结束：清理显存缓存并统计总耗时
    torch.cuda.empty_cache()
    end_time = datetime.datetime.now()
    total_training_time = end_time - time0
    print(f'Total training time: {total_training_time}')

    # 配置允许跳过训练后评估与 CAM 生成（例如只关注训练指标时）
    if getattr(cfg.train, 'skip_post_train_eval', False):
        print("Skipping post-training evaluation and CAM generation (skip_post_train_eval=true).")
        return model

    # ============ 训练后处理：最终评估 + CAM 伪标签生成 ============
    print("\n" + "="*80)
    print("POST-TRAINING EVALUATION AND CAM GENERATION")
    print("="*80)

    # 1) 在验证集上进行最终评估
    print("\n1. Evaluating on VAL set...")
    print("-" * 50)
    _, val_dataset_eval = get_cls_dataset(cfg, split="val", val_root=cfg.dataset.val_root, enable_rotation=False, p=0.0)
    val_eval_loader = DataLoader(val_dataset_eval, batch_size=8, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
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

    # 2) 在测试集上进行最终评估
    print("\n2. Evaluating on TEST set...")
    print("-" * 50)
    _, test_dataset = get_cls_dataset(cfg, split="test", val_root=cfg.dataset.test_root, enable_rotation=False, p=0.0)
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
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

    # 3) 加载训练集（不做旋转增强），用其生成 CAM 伪标签（batch=1，供后续分割监督使用）
    train_dataset, _ = get_cls_dataset(cfg, split="test", val_root=cfg.dataset.test_root, enable_rotation=False, p=0.0)
    print("\n2. Generating CAMs for complete training dataset...")
    print("-" * 50)
    train_cam_loader = DataLoader(train_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)

    print(f"Generating CAMs for all {len(train_dataset)} training samples...")
    print(f"Output directory: {cfg.work_dir.pred_dir}")

    # 优先加载验证集最优的 checkpoint（best_cam.pth）用于生成 CAM
    best_model_path = os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        best_iter = checkpoint.get("iter", "unknown")
        print(f"Best model loaded successfully! (Saved at iteration: {best_iter})")
    else:
        print("Warning: Best model checkpoint not found, using current model state")
        print(f"Expected path: {best_model_path}")

    # 用最优模型为训练集生成 CAM 可视化与伪标签，保存到 cfg.work_dir.pred_dir
    generate_cam(model=model, data_loader=train_cam_loader, cfg=cfg, device=device)

    print("\nFiles generated:")
    print(f"  • Training CAM visualizations: {cfg.work_dir.pred_dir}/*.png")
    print(f"  • Model checkpoint: {cfg.work_dir.ckpt_dir}/best_cam.pth")
    print("="*80)

    return model


def main():
    # 命令行参数：配置文件路径、GPU、随机种子以及训练轮数等覆盖项
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device id')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--epoch', type=int, default=8,
                        help='Override cfg.train.epoch without editing the config')
    parser.add_argument('--stop-after-epoch', type=int, default=None,
                        help='Stop training after this evaluated epoch, then continue to post-training CAM generation unless skipped')
    parser.add_argument('--generate-pseudo', action='store_true',
                        help='Force post-training evaluation and pseudo-mask generation even if the config skips it')
    args = parser.parse_args()

    # 读取配置文件，并按命令行参数覆盖相应字段
    cfg = OmegaConf.load(args.config)
    if args.epoch is not None:
        cfg.train.epoch = int(args.epoch)
    if args.generate_pseudo:
        cfg.train.skip_post_train_eval = False

    # 为本次运行创建带时间戳的独立输出目录，避免多次实验互相覆盖
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.ckpt_dir, f"{timestamp}_seed{args.seed}")

    # 确保 checkpoint、预测结果与日志目录存在
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.train_log_dir, exist_ok=True)

    print(f'\nConfigs: {cfg}')
    print(f'Seed: {args.seed}')
    set_seed(args.seed)
    # 启动训练
    train(cfg, args)


if __name__ == "__main__":
    main()
