import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets import get_cls_dataset
from src.models import ClsNetwork
from src.utils import validate

CFG = 'configs/experiments/bcss_text_anchor.yaml'
CKPT = 'train_log/bcss_text_anchor/checkpoints/2026-11-14-07-09_seed42/best_cam.pth'


def main():
    cfg = OmegaConf.load(CFG)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}', flush=True)

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

    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    print('model loaded', flush=True)

    loss_function = nn.BCEWithLogitsLoss().to(device)

    for split, root in [('val', cfg.dataset.val_root), ('test', cfg.dataset.test_root)]:
        if split == 'val':
            _, ds = get_cls_dataset(cfg, split='val', val_root=root)
        else:
            _, ds = get_cls_dataset(cfg, split='test', val_root=root, enable_rotation=False, p=0.0)
        loader = DataLoader(ds, batch_size=8, shuffle=False,
                            num_workers=0, pin_memory=True)
        _, _, m, _ = validate(model, loader, cfg, loss_function)

        print(f'\n[{split.upper()}]', flush=True)
        print(f"  Acc       : {m['acc'].item()*100:.2f}%", flush=True)
        print(f"  Acc_class : {m['acc_class'].item()*100:.2f}%", flush=True)
        print(f"  fwIoU     : {m['fwiou'].item()*100:.2f}%", flush=True)
        print(f"  mIoU      : {m['miou'].item()*100:.2f}%", flush=True)
        print(f"  per-class IoU: {[f'{s*100:.2f}' for s in m['iou'][:-1].tolist()]}", flush=True)
        print(f"  per-class Acc: {[f'{s*100:.2f}' for s in m['acc_per_class'][:-1].tolist()]}", flush=True)


if __name__ == '__main__':
    main()
