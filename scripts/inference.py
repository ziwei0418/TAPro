"""
Inference script for generating CAMs.
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets import get_cls_dataset
from src.engine import generate_cam
from src.models import ClsNetwork


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
        text_anchor_sparsity_weight=getattr(cfg.train, 'text_anchor_sparsity_weight', 0.0),
    )
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    print(f"✓ Model loaded from {args.checkpoint}")

    # Generate CAMs
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    generate_cam(model, data_loader, cfg, device=device)


if __name__ == "__main__":
    main()
