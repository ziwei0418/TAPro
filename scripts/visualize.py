"""
Visualization script for exporting framework images
"""
import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.bcss_dataset import get_cls_dataset
from src.models.network import ClsNetwork
from src.utils.hierarchical import merge_subclass_cams_to_parent
from src.utils.evaluate import get_seg_label


# Color scheme for classes
CLASS_COLORS = np.array([
    [255, 0, 0],     # TUM - red
    [0, 255, 0],     # STR - green
    [0, 0, 255],     # LYM - blue
    [128, 0, 128],   # NEC - purple
    [255, 255, 255], # BACK - white
], dtype=np.uint8)


def save_cam_heatmap(cam_tensor, save_path, size=224, interpolation=cv2.INTER_LINEAR):
    """Save single-channel CAM as jet colormap image.
    Use interpolation=cv2.INTER_NEAREST for VQ CAM to preserve blocky/quantized look.
    """
    cam = cam_tensor.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_resized = cv2.resize(cam, (size, size), interpolation=interpolation)
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(save_path, heatmap)


def save_seg_mask(seg_tensor, save_path, size=224):
    """Save segmentation mask as colored image."""
    seg = seg_tensor.cpu().numpy().astype(np.uint8)
    colored = CLASS_COLORS[seg]  # [H, W, 3]
    colored_resized = cv2.resize(colored, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(save_path, colored_resized[:, :, ::-1])  # RGB→BGR


@torch.no_grad()
def export_framework_images(model, data_loader, cfg, save_dir='framework_figs', target_image=None):
    """Export intermediate outputs for framework diagram.
    Only uses specified image from TEST set (data_loader must be test_loader).
    
    Args:
        model: Trained WSSS model
        data_loader: DataLoader for test dataset
        cfg: Configuration object
        save_dir: Directory to save visualization outputs
        target_image: Specific image filename to visualize (if None, uses first available)
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    MEAN = np.array([0.66791496, 0.47791372, 0.70623304])
    STD = np.array([0.1736589, 0.22564577, 0.19820057])
    class_names = ['TUM', 'STR', 'LYM', 'NEC']

    if target_image is None:
        target_image = "TCGA-EW-A1P7-DX1_xmin9808_ymin46395_MPP-0.2500+52.png"
    
    best_data = None
    print(f"\n[Framework Export] Looking for image: {target_image}")
    
    for data in data_loader:
        name, inputs, cls_label, labels = data
        inputs = inputs.cuda().float()
        labels_np = labels.numpy()
        cls_label = cls_label.cuda().float()

        # Forward pass
        cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list, aux = model(inputs)

        cam2_p = merge_subclass_cams_to_parent(cam2, k_list, method='max')
        cam3_p = merge_subclass_cams_to_parent(cam3, k_list, method='max')
        cam4_p = merge_subclass_cams_to_parent(cam4, k_list, method='max')

        cams2 = get_seg_label(cam2_p, inputs, cls_label).cuda()
        cams3 = get_seg_label(cam3_p, inputs, cls_label).cuda()
        cams4 = get_seg_label(cam4_p, inputs, cls_label).cuda()
        fuse = 0.3 * cams2 + 0.3 * cams3 + 0.4 * cams4

        for b in range(inputs.shape[0]):
            n = name[b] if isinstance(name[b], str) else name[b][0]
            if n != target_image and not n.endswith(target_image):
                continue
            
            gt = labels_np[b]
            pred = torch.argmax(fuse[b], dim=0).cpu().numpy()
            unique_classes = np.unique(gt)
            fg_classes = [c for c in unique_classes if c < 4]
            
            best_data = {
                'name': n,
                'input': inputs[b].cpu(),
                'cls_label': cls_label[b].cpu(),
                'gt': gt,
                'pred': pred,
                'cam2_p': cam2_p[b].cpu(),
                'cam3_p': cam3_p[b].cpu(),
                'cam4_p': cam4_p[b].cpu(),
                'cam4_sub': cam4[b].cpu(),
                'fuse': fuse[b].cpu(),
                'fg_classes': fg_classes,
            }
            break
        if best_data is not None:
            break

    if best_data is None:
        print(f"[Framework Export] Image '{target_image}' not found, using first available image")
        # Fallback: use first image
        data = next(iter(data_loader))
        name, inputs, cls_label, labels = data
        inputs = inputs.cuda().float()
        cls_label = cls_label.cuda().float()
        
        cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list, aux = model(inputs)
        cam2_p = merge_subclass_cams_to_parent(cam2, k_list, method='max')
        cam3_p = merge_subclass_cams_to_parent(cam3, k_list, method='max')
        cam4_p = merge_subclass_cams_to_parent(cam4, k_list, method='max')
        
        cams2 = get_seg_label(cam2_p, inputs, cls_label).cuda()
        cams3 = get_seg_label(cam3_p, inputs, cls_label).cuda()
        cams4 = get_seg_label(cam4_p, inputs, cls_label).cuda()
        fuse = 0.3 * cams2 + 0.3 * cams3 + 0.4 * cams4
        
        best_data = {
            'name': name[0],
            'input': inputs[0].cpu(),
            'cls_label': cls_label[0].cpu(),
            'gt': labels[0].numpy(),
            'pred': torch.argmax(fuse[0], dim=0).cpu().numpy(),
            'cam2_p': cam2_p[0].cpu(),
            'cam3_p': cam3_p[0].cpu(),
            'cam4_p': cam4_p[0].cpu(),
            'cam4_sub': cam4[0].cpu(),
            'fuse': fuse[0].cpu(),
            'fg_classes': list(range(4)),
        }

    d = best_data
    dom_cls = 1  # Use Stroma (STR) class for CAM visualization
    print(f"[Framework Export] Using image: {d['name']} (CAM class: {class_names[dom_cls]})")

    # 1. Input image (denormalize)
    img = d['input'].permute(1, 2, 0).numpy()
    img = (img * STD + MEAN) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, '01_input.png'), img[:, :, ::-1])

    # 2. CAM heatmaps per scale
    for i, (cam_p, scale) in enumerate([
        (d['cam2_p'], '28x28'),
        (d['cam3_p'], '14x14'), (d['cam4_p'], '7x7')]):
        save_cam_heatmap(cam_p[dom_cls], os.path.join(save_dir, f'02_cam_scale{i+2}_{scale}.png'))

    # 3. Per-class CAM heatmaps (scale 4)
    for c in range(4):
        save_cam_heatmap(d['cam4_p'][c], os.path.join(save_dir, f'03_cam4_{class_names[c]}.png'))

    # 4. Fused pseudo segmentation
    save_seg_mask(torch.tensor(d['pred']), os.path.join(save_dir, '04_pseudo_seg.png'))

    # 5. Ground truth
    save_seg_mask(torch.tensor(d['gt']), os.path.join(save_dir, '05_ground_truth.png'))

    # 6. FG/BG masked images
    # Đối với cơ chế Contrastive Loss: Mask Foreground của class C chính là CAM của class C.
    # Mask Background của class C chính là toàn bộ các pixel KHÔNG thuộc class C.
    cam4_dom = d['cam4_p'][dom_cls]  # [H, W] cho class dom_cls (ví dụ: STR)
    
    # Normalize the heatmap to [0, 1] 
    cam4_norm = cam4_dom - cam4_dom.min()
    cam4_norm = cam4_norm / (cam4_norm.max() + 1e-8)
    cam4_224 = F.interpolate(cam4_norm.unsqueeze(0).unsqueeze(0), (224, 224), mode='bilinear')[0, 0]
    
    # Dùng Threshold = 0.2 để lấy vùng Attention (Foreground) rộng ra một chút, giống viền Pseudo Mask
    thresh = 0.2
    fg_mask = (cam4_224 > thresh).float().numpy()
    bg_mask = 1.0 - fg_mask
    
    fg_img = img * fg_mask[:, :, None]
    bg_img = img * bg_mask[:, :, None]
    cv2.imwrite(os.path.join(save_dir, '06_fg_masked.png'), fg_img[:, :, ::-1].astype(np.uint8))
    cv2.imwrite(os.path.join(save_dir, '06_bg_masked.png'), bg_img[:, :, ::-1].astype(np.uint8))
    cv2.imwrite(os.path.join(save_dir, '06_fg_mask_binary.png'), (fg_mask * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(save_dir, '06_bg_mask_binary.png'), (bg_mask * 255).astype(np.uint8))

    # 7. Overlay CAM on input
    cam_overlay = d['cam4_p'].max(0)[0].numpy()
    cam_overlay = (cam_overlay - cam_overlay.min()) / (cam_overlay.max() - cam_overlay.min() + 1e-8)
    cam_resized = cv2.resize(cam_overlay, (224, 224))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img[:, :, ::-1], 0.5, heatmap, 0.5, 0)
    cv2.imwrite(os.path.join(save_dir, '07_cam_overlay.png'), overlay)

    # 8. Legend
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    for c, name in enumerate(class_names + ['BACK']):
        ax.barh(c, 1, color=CLASS_COLORS[c] / 255.0, edgecolor='black')
        ax.text(0.5, c, name, ha='center', va='center', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.axis('off')
    fig.savefig(os.path.join(save_dir, '08_legend.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 9. VQ Codebook Details (if VQ is enabled)
    if hasattr(model, 'use_vq') and model.use_vq and hasattr(model, 'vq_head3'):
        print("[Framework Export] Exporting VQ codebook details...")
        vq_dir = os.path.join(save_dir, 'vq_detail')
        os.makedirs(vq_dir, exist_ok=True)

        single_input = d['input'].unsqueeze(0).cuda().float()
        with torch.no_grad():
            feats = model.encoder(single_input)
            p1, p2, p3, p4 = model.sfp(feats)
            
            vq_head = model.vq_head3
            B, C, H3, W3 = p3.shape
            _x3 = p3.permute(0, 2, 3, 1).reshape(B * H3 * W3, C)
            _x3 = _x3 / _x3.norm(dim=-1, keepdim=True)

            z_projected = vq_head.project(_x3)
            z_norm = F.normalize(z_projected, dim=-1)
            cb_norm = F.normalize(vq_head.vq.codebook, dim=-1)
            cosine_sim = z_norm @ cb_norm.t()
            indices = torch.argmax(cosine_sim, dim=1)
            z_q = vq_head.vq.codebook[indices]
            
            z_q_spatial = z_q.view(1, H3, W3, -1).permute(0, 3, 1, 2)
            vq_cam_out = vq_head.cam_head(z_q_spatial)

        # Save VQ visualizations
        p3_vis = p3[0].mean(0).cpu()
        save_cam_heatmap(p3_vis, os.path.join(vq_dir, '09a_pixel_features_P3.png'))

        z_proj_vis = z_projected.view(H3, W3, -1).mean(-1).cpu()
        save_cam_heatmap(z_proj_vis, os.path.join(vq_dir, '09b_projected_features.png'))

        # Discrete code indices
        indices_grid = indices.view(H3, W3).cpu().numpy()
        fig_idx, ax_idx = plt.subplots(1, 1, figsize=(5, 5))
        ax_idx.imshow(indices_grid, cmap='tab20', interpolation='nearest')
        for i in range(H3):
            for j in range(W3):
                ax_idx.text(j, i, str(indices_grid[i, j]), ha='center', va='center',
                           fontsize=5, color='white', fontweight='bold')
        ax_idx.set_title(f'Discrete Codes ({H3}x{W3})', fontsize=10)
        ax_idx.axis('off')
        fig_idx.savefig(os.path.join(vq_dir, '09c_discrete_codes.png'), dpi=200, bbox_inches='tight')
        plt.close(fig_idx)

        z_q_vis = z_q.view(H3, W3, -1).mean(-1).cpu()
        save_cam_heatmap(z_q_vis, os.path.join(vq_dir, '09d_quantized_features.png'),
                         interpolation=cv2.INTER_NEAREST)

        # VQ CAM per class
        k_list = model.k_list
        vq_cam_parent = merge_subclass_cams_to_parent(vq_cam_out, k_list, method='max')
        for c in range(4):
            save_cam_heatmap(vq_cam_parent[0, c].cpu(),
                           os.path.join(vq_dir, f'09e_vq_cam_{class_names[c]}.png'),
                           interpolation=cv2.INTER_NEAREST)
        
        vq_cam_max = vq_cam_parent[0].max(0)[0].cpu()
        save_cam_heatmap(vq_cam_max, os.path.join(vq_dir, '09e_vq_cam_combined.png'),
                         interpolation=cv2.INTER_NEAREST)

        # Codebook usage histogram
        unique_codes, counts = torch.unique(indices, return_counts=True)
        fig_hist, ax_hist = plt.subplots(1, 1, figsize=(8, 3))
        ax_hist.bar(unique_codes.cpu().numpy(), counts.cpu().numpy(), color='#E67E22')
        ax_hist.set_xlabel('Code Index')
        ax_hist.set_ylabel('Usage Count')
        ax_hist.set_title(f'Codebook Usage ({len(unique_codes)}/{vq_head.vq.num_codes} codes active)')
        fig_hist.savefig(os.path.join(vq_dir, '09f_codebook_usage.png'), dpi=150, bbox_inches='tight')
        plt.close(fig_hist)

        print(f"[Framework Export] VQ detail: {len(os.listdir(vq_dir))} images saved to {vq_dir}")

    print(f"[Framework Export] Saved {len(os.listdir(save_dir))} images to {save_dir}")
    model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='framework_figs',
                        help='Directory to save visualizations')
    parser.add_argument('--target_image', type=str, default=None,
                        help='Specific image filename to visualize')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device id')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test dataset
    print("\nLoading test dataset...")
    _, test_dataset = get_cls_dataset(cfg, split="test", val_root=cfg.dataset.test_root,
                                      enable_rotation=False, p=0.0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                             num_workers=min(10, os.cpu_count()),
                             pin_memory=True)
    print(f"✓ Loaded {len(test_dataset)} test samples")

    # Load model
    print("\nLoading model...")
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
        use_text_proto=getattr(cfg.train, 'use_text_proto', False)
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    print(f"✓ Model loaded from {args.checkpoint}")

    # Export framework images
    export_framework_images(model, test_loader, cfg, 
                           save_dir=args.output_dir,
                           target_image=args.target_image)


if __name__ == "__main__":
    main()
