#!/usr/bin/env python3
# ============================================================
# Entry point — run: python train_seg/train_seg.py
# ============================================================

import sys
import os

# Ensure project root is on sys.path so `train_seg` package resolves
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from train_seg.config import load_config
from train_seg.trainer import train_seg

if __name__ == '__main__':
    cfg = load_config()
    print(f"Config:\n{cfg}")
    train_seg(cfg)