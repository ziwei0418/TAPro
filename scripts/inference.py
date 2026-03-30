"""
Inference script for generating CAMs
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.bcss_dataset import get_cls_dataset, get_mean_std
from src.models.network import ClsNetwork
from src.utils.hierarchical import merge_subclass_cams_to_parent
from src.utils.evaluate import get_seg_label

# Optional CRF
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    print("WARNING: pydensecrf not installed. CRF post-processing disabled.")


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
        """
        image: [H, W, 3], uint8 RGB image
        probmap: [C, H, W], float32 softmax probabilities
        Returns: [C, H, W] refined probabilities
        """
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


def generate_cam(model, data_loader, cfg, device='cuda:0'):
    """Generate CAMs for all images in data_loader."""
    model.eval()
    
    # CRF post-processing setup
    use_crf = getattr(cfg.train, 'use_crf', False) and CRF_AVAILABLE
    if use_crf:
        crf_processor = DenseCRF(
            iter_max=getattr(cfg.train, 'crf_iter_max', 10),
            pos_w=getattr(cfg.train, 'crf_pos_w', 3),
            pos_xy_std=getattr(cfg.train, 'crf_pos_xy_std', 1),
            bi_w=getattr(cfg.train, 'crf_bi_w', 4),
            bi_xy_std=getattr(cfg.train, 'crf_bi_xy_std', 67),
            bi_rgb_std=getattr(cfg.train, 'crf_bi_rgb_std', 3),
        )
        MEAN_arr, STD_arr = get_mean_std(cfg.dataset.name)
        MEAN = np.array(MEAN_arr)
        STD = np.array(STD_arr)
        print("CRF post-processing enabled")

    PALETTE = [
        [255, 0, 0],      # TUM
        [0, 255, 0],      # STR
        [0, 0, 255],      # LYM
        [153, 0, 255],    # NEC
        [255, 255, 255],  # BACK
        [0, 0, 0],
    ]

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader), desc="Generating CAMs"):
            name, inputs, cls_label, labels = data

            inputs = inputs.to(device).float()
            cls_label = cls_label.to(device).float()

            cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list, _ = model(inputs)

            # Merge subclass CAMs to parent
            cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test)
            cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test)
            cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test)

            # Get segmentation labels
            cam2 = get_seg_label(cam2, inputs, cls_label).to(device)
            cam3 = get_seg_label(cam3, inputs, cls_label).to(device)
            cam4 = get_seg_label(cam4, inputs, cls_label).to(device)

            # Fuse multi-scale CAMs
            fuse234 = 0.3 * cam2 + 0.3 * cam3 + 0.4 * cam4
            
            if use_crf:
                # Apply CRF per sample
                probs = F.softmax(fuse234, dim=1)
                for i in range(len(name)):
                    img_np = inputs[i].cpu().numpy().transpose(1, 2, 0)
                    img_rgb = ((img_np * STD + MEAN) * 255).clip(0, 255).astype(np.uint8)
                    refined = crf_processor(img_rgb, probs[i].cpu().numpy().astype(np.float32))
                    label = np.argmax(refined, axis=0).astype(np.uint8)
                    pred_mask = Image.fromarray(label).convert('P')
                    flat_palette = [val for sublist in PALETTE for val in sublist]
                    pred_mask.putpalette(flat_palette)
                    pred_mask.save(os.path.join(cfg.work_dir.pred_dir, name[i] + ".png"))
            else:
                output_fuse234 = torch.argmax(fuse234, dim=1).long()
                for i in range(len(output_fuse234)):
                    pred_mask = Image.fromarray(output_fuse234[i].cpu().numpy().astype(np.uint8)).convert('P')
                    flat_palette = [val for sublist in PALETTE for val in sublist]
                    pred_mask.putpalette(flat_palette)
                    pred_mask.save(os.path.join(cfg.work_dir.pred_dir, name[i] + ".png"))
    
    model.train()
    print(f"✓ CAMs generated and saved to {cfg.work_dir.pred_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device id')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to generate CAMs for')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    if args.split == 'train':
        dataset, _ = get_cls_dataset(cfg, split="train", val_root=cfg.dataset.train_root,
                                     enable_rotation=False, p=0.0)
    elif args.split == 'val':
        _, dataset = get_cls_dataset(cfg, split="val", val_root=cfg.dataset.val_root,
                                     enable_rotation=False, p=0.0)
    else:  # test
        _, dataset = get_cls_dataset(cfg, split="test", val_root=cfg.dataset.test_root,
                                     enable_rotation=False, p=0.0)
    
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False,
                             num_workers=min(10, os.cpu_count()),
                             pin_memory=True)
    print(f"✓ Loaded {len(dataset)} samples")

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
        use_text_proto=getattr(cfg.train, 'use_text_proto', False),
        dataset_name=cfg.dataset.name,
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    print(f"✓ Model loaded from {args.checkpoint}")

    # Generate CAMs
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    generate_cam(model, data_loader, cfg, device=device)


if __name__ == "__main__":
    main()
