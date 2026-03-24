# Hướng dẫn sử dụng dự án WSSS_2026_RESEARCH

## Cài đặt dependencies

```bash
cd /home/s12gbn5/Desktop/WSSS_2026_RESEARCH
pip install -r requirements.txt
```

## Cấu trúc dự án đã hoàn thành

```
WSSS_2026_RESEARCH/
├── src/                          # Source code modules
│   ├── datasets/                 # Dataset loading
│   │   ├── __init__.py
│   │   └── bcss_dataset.py       # ✅ BCSS dataset classes
│   ├── losses/                   # Loss functions
│   │   ├── __init__.py
│   │   └── contrast_loss.py      # ✅ InfoNCE, consistency losses
│   ├── models/                   # Network architecture
│   │   ├── __init__.py
│   │   ├── encoder.py            # ✅ UNI encoder + LoRA
│   │   ├── fpn.py                # ✅ Feature Pyramid Network
│   │   ├── vq_modules.py         # ✅ Vector Quantization
│   │   ├── hash_miner.py         # ✅ Hard negative mining
│   │   ├── text_encoder.py       # ✅ Text prototype encoding
│   │   └── network.py            # ✅ Main ClsNetwork
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── pyutils.py            # ✅ Basic utilities
│       ├── optimizer.py          # ✅ PolyWarmupAdamW
│       ├── evaluate.py           # ✅ Metrics calculation
│       ├── hierarchical.py       # ✅ Hierarchical merging
│       └── fgbg_feature.py       # ✅ FG/BG feature extraction
├── scripts/                      # Training & inference scripts  
│   ├── train.py                  # ✅ Training script
│   ├── inference.py              # ✅ CAM generation
│   └── visualize.py              # ✅ Framework visualization
├── configs/
│   └── default.yaml              # ✅ Configuration file
├── docs/                         # Documentation
│   ├── RESTRUCTURING.md
│   ├── TODO.md
│   ├── SUMMARY_VI.md
│   └── PROJECT_TREE.md
├── requirements.txt              # ✅ Dependencies
├── setup.py                      # ✅ Package setup
├── .gitignore                    # ✅ Git ignore rules
├── README.md                     # ✅ Project README
└── wsss.py                       # Original monolithic file (reference)
```

## Sử dụng

### 1. Training

```bash
# Train model with default config
python scripts/train.py --config configs/default.yaml --gpu 0

# Train với config tùy chỉnh
python scripts/train.py --config configs/my_config.yaml --gpu 0
```

### 2. Inference (Generate CAMs)

```bash
# Generate CAMs cho test set
python scripts/inference.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/2024-01-15-10-30/best_cam.pth \
    --split test \
    --gpu 0

# Generate CAMs cho val set
python scripts/inference.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best_cam.pth \
    --split val \
    --gpu 0
```

### 3. Visualization

```bash
# Export framework images (cho paper)
python scripts/visualize.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best_cam.pth \
    --output_dir framework_figs \
    --gpu 0

# Visualize một ảnh cụ thể
python scripts/visualize.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best_cam.pth \
    --target_image "TCGA-EW-A1P7-DX1_xmin9808_ymin46395_MPP-0.2500+52.png" \
    --output_dir my_visualizations \
    --gpu 0
```

## Cấu hình trong configs/default.yaml

Các tham số quan trọng:

```yaml
model:
  hf_token: "YOUR_HUGGINGFACE_TOKEN"  # Token để download UNI model
  use_lora: false                      # Bật/tắt LoRA adapters

train:
  samples_per_gpu: 4                   # Batch size
  accum_steps: 4                       # Gradient accumulation
  epoch: 8                             # Số epochs
  
  # VQ Codebook
  use_vq: true
  vq_num_codes: 128
  vq_cam_weight: 0.3
  
  # Text prototypes
  use_text_proto: true
  
  # Early stopping
  use_early_stop: true
  patience: 2
```

## Import module trong code

```python
# Import datasets
from src.datasets.bcss_dataset import BCSSTrainingDataset, get_cls_dataset

# Import models
from src.models.network import ClsNetwork
from src.models.encoder import UNIEncoder
from src.models.vq_modules import VQCodebook

# Import losses
from src.losses.contrast_loss import InfoNCELossFG, InfoNCELossBG

# Import utils
from src.utils.evaluate import validate, ConfusionMatrixAllClass
from src.utils.hierarchical import merge_subclass_cams_to_parent
```

## Outputs

### Training outputs
- `checkpoints/YYYY-MM-DD-HH-MM/best_cam.pth` - Best model checkpoint
- `train_log/` - Training logs

### Inference outputs  
- `predictions/*.png` - CAM predictions (colored masks)

### Visualization outputs
- `framework_figs/01_input.png` - Input image
- `framework_figs/02_cam_scale*.png` - Multi-scale CAMs
- `framework_figs/03_cam4_*.png` - Per-class CAMs
- `framework_figs/04_pseudo_seg.png` - Pseudo segmentation
- `framework_figs/05_ground_truth.png` - Ground truth
- `framework_figs/vq_detail/` - VQ codebook visualizations

## Kiểm tra imports

```bash
# Test imports
python -c "from src.models.network import ClsNetwork; print('✓ Model imports OK')"
python -c "from src.datasets.bcss_dataset import get_cls_dataset; print('✓ Dataset imports OK')"
python -c "from src.losses.contrast_loss import InfoNCELossFG; print('✓ Loss imports OK')"
python -c "from src.utils.evaluate import validate; print('✓ Utils imports OK')"
```

## Git workflow

```bash
# Initialize git (nếu chưa có)
cd /home/s12gbn5/Desktop/WSSS_2026_RESEARCH
git init
git add .
git commit -m "Initial commit: Restructured WSSS project"

# Tạo repository trên GitHub rồi push
git remote add origin https://github.com/YOUR_USERNAME/WSSS_2026_RESEARCH.git
git branch -M main
git push -u origin main
```

## Troubleshooting

### Lỗi import torch/numpy
```bash
pip install torch torchvision numpy opencv-python
```

### Lỗi CUDA out of memory
```yaml
# Giảm batch size trong config
train:
  samples_per_gpu: 2  # Từ 4 → 2
  accum_steps: 8      # Từ 4 → 8 (giữ nguyên effective batch size)
```

### Lỗi HuggingFace token
```yaml
# Lấy token tại https://huggingface.co/settings/tokens
model:
  hf_token: "hf_YOUR_NEW_TOKEN_HERE"
```

## Tài liệu tham khảo

- `docs/RESTRUCTURING.md` - Chi tiết quá trình tái cấu trúc
- `docs/TODO.md` - Danh sách các nhiệm vụ đã hoàn thành
- `docs/SUMMARY_VI.md` - Tóm tắt dự án (tiếng Việt)
- `README.md` - Tài liệu tổng quan

## Hoàn thành ✅

Tất cả các file đã được tạo thành công:
- ✅ 7 model files trong src/models/
- ✅ 3 script files trong scripts/
- ✅ 6 utility files trong src/utils/
- ✅ 2 loss files trong src/losses/
- ✅ 1 dataset file trong src/datasets/
- ✅ Config, docs, setup files

Dự án sẵn sàng để đẩy lên GitHub! 🚀
