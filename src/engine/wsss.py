import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.utils.cam_refine import apply_cam_power_calibration, feature_region_refine_scores
from src.utils.evaluate import get_seg_label
from src.utils.hierarchical import merge_subclass_cams_to_parent


PALETTE = [
    [255, 0, 0],      # TUM / TE
    [0, 255, 0],      # STR / NEC
    [0, 0, 255],      # LYM
    [153, 0, 255],    # NEC / TAS
    [255, 255, 255],  # BACK
    [0, 0, 0],
]


def generate_cam(model=None, data_loader=None, cfg=None, device=None):
    """Generate fused CAM segmentation masks and save them to cfg.work_dir.pred_dir."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    save_confidence = getattr(cfg.train, "save_pseudo_confidence", False)
    save_soft = getattr(cfg.train, "save_pseudo_soft", False)
    if save_confidence:
        conf_dir = getattr(cfg.work_dir, "confidence_dir", cfg.work_dir.pred_dir + "_confidence")
        margin_dir = getattr(cfg.work_dir, "margin_dir", cfg.work_dir.pred_dir + "_margin")
        os.makedirs(conf_dir, exist_ok=True)
        os.makedirs(margin_dir, exist_ok=True)
    else:
        conf_dir = None
        margin_dir = None
    if save_soft:
        soft_dir = getattr(cfg.work_dir, "soft_dir", cfg.work_dir.pred_dir + "_soft")
        os.makedirs(soft_dir, exist_ok=True)
    else:
        soft_dir = None

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data

            inputs = inputs.to(device).float()
            cls_label = cls_label.to(device).float()

            cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list, _ = model(inputs)

            cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test)
            cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test)
            cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test)

            cam2 = get_seg_label(cam2, inputs, cls_label).to(device)
            cam3 = get_seg_label(cam3, inputs, cls_label).to(device)
            cam4 = get_seg_label(cam4, inputs, cls_label).to(device)

            fuse234 = 0.3 * cam2 + 0.3 * cam3 + 0.4 * cam4
            fuse234 = apply_cam_power_calibration(fuse234, cfg)
            if getattr(cfg.train, "use_feature_region_refine", False):
                if getattr(cfg.train, "use_attention_boundary_refine", False):
                    uni_feats, attention_distance = model.encoder.forward_with_attention_distance(
                        inputs,
                        attn_block_idx=int(getattr(cfg.train, "attention_boundary_block", 23)),
                        target_size=int(getattr(cfg.train, "attention_boundary_target_size", 7)),
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
            probs = torch.softmax(fuse234, dim=1)
            top2 = torch.topk(probs, k=min(2, probs.shape[1]), dim=1).values
            confidence = top2[:, 0]
            margin = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else top2[:, 0]
            output_fuse234 = torch.argmax(fuse234, dim=1).long()

            for i in range(len(output_fuse234)):
                pred_mask = Image.fromarray(
                    output_fuse234[i].cpu().numpy().astype(np.uint8)
                ).convert("P")
                flat_palette = [val for sublist in PALETTE for val in sublist]
                pred_mask.putpalette(flat_palette)
                pred_mask.save(os.path.join(cfg.work_dir.pred_dir, name[i] + ".png"))
                if save_confidence:
                    conf_img = (confidence[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    margin_img = (margin[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    Image.fromarray(conf_img).save(os.path.join(conf_dir, name[i] + ".png"))
                    Image.fromarray(margin_img).save(os.path.join(margin_dir, name[i] + ".png"))
                if save_soft:
                    np.save(
                        os.path.join(soft_dir, name[i] + ".npy"),
                        probs[i].cpu().numpy().astype(np.float16),
                    )

    model.train()
    return
