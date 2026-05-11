"""
Evaluation metrics and utilities
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
try:
    import ttach as tta
except ImportError:
    tta = None
    print("Warning: ttach not installed. Test-time augmentation will be disabled.")

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False

from .pyutils import AverageMeter
from .hierarchical import merge_to_parent_predictions, merge_subclass_cams_to_parent
from .cam_refine import apply_cam_power_calibration, feature_region_refine_scores


class DenseCRF:
    """Dense CRF post-processing for semantic segmentation."""
    def __init__(self, iter_max=10, pos_w=3, pos_xy_std=1, bi_w=4, bi_xy_std=67, bi_rgb_std=3):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape
        U = unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)
        image = np.ascontiguousarray(image)
        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )
        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))
        return Q


def compute_boundary_iou(gt, pred, num_classes, dilation=3):
    """Compute per-class Boundary IoU for a single sample."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation+1, 2*dilation+1))
    per_class_intersection = np.zeros(num_classes, dtype=np.float64)
    per_class_union = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        gt_c = (gt == c).astype(np.uint8)
        pred_c = (pred == c).astype(np.uint8)
        # boundary = mask - erode(mask)
        gt_boundary = gt_c - cv2.erode(gt_c, kernel)
        pred_boundary = pred_c - cv2.erode(pred_c, kernel)
        per_class_intersection[c] = (gt_boundary & pred_boundary).sum()
        per_class_union[c] = (gt_boundary | pred_boundary).sum()

    valid = per_class_union > 0
    per_class_biou = np.zeros(num_classes)
    per_class_biou[valid] = per_class_intersection[valid] / per_class_union[valid]
    return per_class_biou


class ConfusionMatrixAllClass(object):
    """Confusion matrix for multi-class segmentation"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat1 = None
        self.mat2 = None

    def update(self, a, b):
        """
        :param a: ground truth
        :param b: pred
        :return:
        """
        n = self.num_classes
        if self.mat1 is None:
            self.mat1 = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        if self.mat2 is None:
            self.mat2 = torch.zeros((2, 2), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k].to(torch.int64)
            self.mat1 += torch.bincount(inds, minlength=n**2).reshape(n, n)
            a[a != 0] = 1
            b[b != 0] = 1
            k = (a >= 0) & (a < 2)
            inds = 2 * a[k].to(torch.int64) + b[k].to(torch.int64)
            self.mat2 += torch.bincount(inds, minlength=2**2).reshape(2, 2)
            del a
            del b

    def reset(self):
        if self.mat1 is not None:
            self.mat1.zero_()
        if self.mat2 is not None:
            self.mat2.zero_()

    def compute(self):
        h = self.mat1.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        dice = 2 * torch.diag(h) / (h.sum(1) + h.sum(0))
        # Frequency-weighted IoU
        freq = h.sum(1) / h.sum()
        fwiu = (freq * iu).sum()
        # Boundary IoU (binary: foreground vs background)
        h_bg_fg = self.mat2.float()
        biu = torch.diag(h_bg_fg) / (h_bg_fg.sum(1) + h_bg_fg.sum(0) - torch.diag(h_bg_fg))
        dice_bg_fg = 2 * torch.diag(h_bg_fg) / (h_bg_fg.sum(1) + h_bg_fg.sum(0))
        return acc_global, acc, iu, dice, dice_bg_fg, fwiu, biu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat1)
        torch.distributed.all_reduce(self.mat2)

    def __str__(self):
        acc_global, acc, iu, dice, fg_bg_dice, fwiu, biu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}\n'
            'dice: {}\n'
            'mean dice: {}\n'
            'fg_bg_dice: {}\n'
            'mean_fg_bg: {}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu[:-1].mean().item() * 100,
                ['{:.1f}'.format(i) for i in (dice * 100).tolist()],
                dice[:-1].mean().item() * 100,
                ['{:.1f}'.format(i) for i in (fg_bg_dice * 100).tolist()],
                fg_bg_dice.mean().item() * 100
            )


def get_seg_label(cams, inputs, label):
    """Generate segmentation labels from CAMs
    
    Args:
        cams: [B, C, H', W'] Raw CAM activations
        inputs: [B, 3, H, W] Input images
        label: [B, C] Class labels (1 for present, 0 for absent)
    
    Returns:
        torch.Tensor: [B, C, H, W] Normalized CAMs resized to input size
    """
    with torch.no_grad():
        b, c, h, w = inputs.shape
        label = label.view(b, -1, 1, 1).cpu().data.numpy()
        cams = cams.cpu().data.numpy()
        cams = np.maximum(cams, 0)
        
        # Normalize CAMs to [0,1]
        channel_max = np.max(cams, axis=(2, 3), keepdims=True)
        channel_min = np.min(cams, axis=(2, 3), keepdims=True)
        cams = (cams - channel_min) / (channel_max - channel_min + 1e-6)
        cams = cams * label
        cams = torch.from_numpy(cams).float()
        
        # Resize to input dimensions
        cams = F.interpolate(cams, size=(h, w), mode="bilinear", align_corners=True)
        cam_max = torch.max(cams, dim=1, keepdim=True)[0]
        bg_cam = (1 - cam_max) ** 10
        cam_all = torch.cat([cams, bg_cam], dim=1)

    return cams


def validate(model=None, data_loader=None, cfg=None, cls_loss_func=None):
    """Validation function with test-time augmentation
    
    Args:
        model: Neural network model
        data_loader: Validation data loader
        cfg: Configuration object
        cls_loss_func: Classification loss function
    
    Returns:
        tuple: (all_cls_acc4, avg_cls_acc4, metrics, cls_loss)
            - all_cls_acc4: Accuracy when all classes are correct
            - avg_cls_acc4: Average per-class accuracy
            - metrics: dict with iou, miou, fwiou, binary_iou, boundary_iou, dice
            - cls_loss: Classification loss value
    """
    model.eval()
    avg_meter = AverageMeter("all_cls_acc4", "avg_cls_acc4", "cls_loss")
    fuse234_matrix = ConfusionMatrixAllClass(num_classes=cfg.dataset.cls_num_classes + 1)
    boundary_iou_sum = np.zeros(5, dtype=np.float64)  # 4 classes + BG
    boundary_iou_count = 0
    
    # Test-time augmentation setup
    use_tta = getattr(cfg.train, 'use_tta', True) and tta is not None
    if use_tta:
        tta_transform = tta.Compose([
            tta.HorizontalFlip(),
            tta.Multiply(factors=[0.9, 1.0, 1.1])
        ])
    
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data
            inputs = inputs.cuda().float()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda().float()

            cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list, _ = model(inputs)

            cls1 = merge_to_parent_predictions(cls1, k_list, method=cfg.train.merge_test)
            cls2 = merge_to_parent_predictions(cls2, k_list, method=cfg.train.merge_test)
            cls3 = merge_to_parent_predictions(cls3, k_list, method=cfg.train.merge_test)
            cls4 = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_test)

            cls_loss1 = cls_loss_func(cls1, cls_label)
            cls_loss2 = cls_loss_func(cls2, cls_label)
            cls_loss3 = cls_loss_func(cls3, cls_label)
            cls_loss4 = cls_loss_func(cls4, cls_label)
            cls_loss = getattr(cfg.train, 'l1', 0.1) * cls_loss1 + cfg.train.l2 * cls_loss2 + cfg.train.l3 * cls_loss3 + cfg.train.l4 * cls_loss4

            cls4 = (torch.sigmoid(cls4) > 0.5).float()
            all_cls_acc4 = (cls4 == cls_label).all(dim=1).float().sum() / cls4.shape[0] * 100
            avg_cls_acc4 = ((cls4 == cls_label).sum(dim=0) / cls4.shape[0]).mean() * 100
            avg_meter.add({"all_cls_acc4": all_cls_acc4, "avg_cls_acc4": avg_cls_acc4, "cls_loss": cls_loss})
            
            if use_tta:
                # Generate evaluation CAMs with TTA
                cams1 = []
                cams2 = []
                cams3 = []
                cams4 = []
                for tta_trans in tta_transform:
                    augmented_tensor = tta_trans.augment_image(inputs)
                    cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list, _ = model(augmented_tensor)

                    cam1 = merge_subclass_cams_to_parent(cam1, k_list, method=cfg.train.merge_test)
                    cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test)
                    cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test)
                    cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test)

                    cam1 = get_seg_label(cam1, augmented_tensor, cls_label).cuda()
                    cam1 = tta_trans.deaugment_mask(cam1).unsqueeze(dim=0)
                    cams1.append(cam1)
                    cam2 = get_seg_label(cam2, augmented_tensor, cls_label).cuda()
                    cam2 = tta_trans.deaugment_mask(cam2).unsqueeze(dim=0)
                    cams2.append(cam2)
                    cam3 = get_seg_label(cam3, augmented_tensor, cls_label).cuda()
                    cam3 = tta_trans.deaugment_mask(cam3).unsqueeze(dim=0)
                    cams3.append(cam3)
                    cam4 = get_seg_label(cam4, augmented_tensor, cls_label).cuda()
                    cam4 = tta_trans.deaugment_mask(cam4).unsqueeze(dim=0)
                    cams4.append(cam4)

                cams1 = torch.cat(cams1, dim=0).mean(dim=0)
                cams2 = torch.cat(cams2, dim=0).mean(dim=0)
                cams3 = torch.cat(cams3, dim=0).mean(dim=0)
                cams4 = torch.cat(cams4, dim=0).mean(dim=0)
            else:
                # No TTA: single forward pass for CAMs
                cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list, _ = model(inputs)

                cam1 = merge_subclass_cams_to_parent(cam1, k_list, method=cfg.train.merge_test)
                cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test)
                cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test)
                cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test)

                cams1 = get_seg_label(cam1, inputs, cls_label).cuda()
                cams2 = get_seg_label(cam2, inputs, cls_label).cuda()
                cams3 = get_seg_label(cam3, inputs, cls_label).cuda()
                cams4 = get_seg_label(cam4, inputs, cls_label).cuda()

            # Fuse multi-scale predictions (same as wsss copy 5: only scales 2,3,4)
            fuse234 = 0.3 * cams2 + 0.3 * cams3 + 0.4 * cams4
            fuse234 = apply_cam_power_calibration(fuse234, cfg)
            if getattr(cfg.train, 'use_feature_region_refine', False):
                if getattr(cfg.train, 'use_attention_boundary_refine', False):
                    uni_feats, attention_distance = model.encoder.forward_with_attention_distance(
                        inputs,
                        attn_block_idx=int(getattr(cfg.train, 'attention_boundary_block', 23)),
                        target_size=int(getattr(cfg.train, 'attention_boundary_target_size', 7)),
                    )
                    feature_map = model.sfp(uni_feats)[3]
                else:
                    attention_distance = None
                    feature_map = model.sfp(model.encoder(inputs))[3]
                fuse234 = feature_region_refine_scores(
                    fuse234,
                    feature_map,
                    cfg,
                    attention_distance=attention_distance,
                )

            # CRF post-processing
            use_crf = getattr(cfg.train, 'use_crf', False) and CRF_AVAILABLE
            if use_crf:
                from src.datasets.bcss_dataset import get_mean_std
                MEAN_arr, STD_arr = get_mean_std(getattr(cfg.dataset, 'name', 'bcss'))
                MEAN_np = np.array(MEAN_arr)
                STD_np = np.array(STD_arr)
                crf_processor = DenseCRF(
                    iter_max=getattr(cfg.train, 'crf_iter_max', 10),
                    pos_w=getattr(cfg.train, 'crf_pos_w', 3),
                    pos_xy_std=getattr(cfg.train, 'crf_pos_xy_std', 1),
                    bi_w=getattr(cfg.train, 'crf_bi_w', 4),
                    bi_xy_std=getattr(cfg.train, 'crf_bi_xy_std', 67),
                    bi_rgb_std=getattr(cfg.train, 'crf_bi_rgb_std', 3),
                )
                probs = F.softmax(fuse234, dim=1)
                refined_labels = []
                for i in range(probs.shape[0]):
                    img_np = inputs[i].cpu().numpy().transpose(1, 2, 0)
                    img_rgb = ((img_np * STD_np + MEAN_np) * 255).clip(0, 255).astype(np.uint8)
                    refined = crf_processor(img_rgb, probs[i].cpu().numpy().astype(np.float32))
                    refined_labels.append(np.argmax(refined, axis=0))
                fuse_label234 = torch.from_numpy(np.stack(refined_labels)).long().cuda()
            else:
                fuse_label234 = torch.argmax(fuse234, dim=1).cuda()
            
            fuse234_matrix.update(labels.detach().clone(), fuse_label234.clone())

            # Accumulate Boundary IoU
            gt_np = labels.cpu().numpy()    # [B, H, W]
            pred_np = fuse_label234.cpu().numpy()
            for b in range(gt_np.shape[0]):
                b_biou = compute_boundary_iou(gt_np[b], pred_np[b], num_classes=5)  # 4 classes + BG
                boundary_iou_sum += b_biou
                boundary_iou_count += 1

    all_cls_acc4, avg_cls_acc4, cls_loss = avg_meter.pop('all_cls_acc4'), avg_meter.pop("avg_cls_acc4"), avg_meter.pop("cls_loss")
    acc_global, acc, iu, dice, dice_bg_fg, fwiu, biu = fuse234_matrix.compute()
    boundary_iou_per_class = boundary_iou_sum / max(boundary_iou_count, 1)
    
    # Per-class frequency-weighted IoU
    freq = fuse234_matrix.mat1.float().sum(1) / fuse234_matrix.mat1.float().sum()
    fwiou_per_class = freq * iu

    metrics = {
        'iou': iu,                          # per-class IoU [C+1] (includes BG)
        'miou': iu[:-1].mean(),
        'dice_per_class': dice,              # per-class Dice [C+1]
        'dice': dice[:-1].mean(),
        'fwiou': fwiu,                       # scalar
        'fwiou_per_class': fwiou_per_class,  # per-class freq-weighted IoU [C+1]
        'binary_iou': biu[1] if len(biu) > 1 else biu.mean(),
        'boundary_iou': boundary_iou_per_class[:4].mean(),
        'boundary_iou_per_class': boundary_iou_per_class,  # numpy [5]
    }
    
    model.train()
    return all_cls_acc4, avg_cls_acc4, metrics, cls_loss
