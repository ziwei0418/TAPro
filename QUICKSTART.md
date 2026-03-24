# 🚀 QUICKSTART - Hướng dẫn cho người mới

Hướng dẫn từng bước để train, validate và generate masks cho dự án WSSS.

---

## 📋 Mục lục

1. [Thiết lập môi trường lần đầu](#1-thiết-lập-môi-trường-lần-đầu)
2. [Kích hoạt môi trường hàng ngày](#2-kích-hoạt-môi-trường-hàng-ngày)
3. [Training model](#3-training-model)
4. [Generate CAM/Masks](#4-generate-cammasks)
5. [Visualization](#5-visualization)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Thiết lập môi trường lần đầu

### Bước 1.1: Clone/Copy project về server

```bash
cd ~
# Nếu từ Git:
git clone <repository_url> WSSS_2026_RESEARCH
# Hoặc đã có sẵn thì bỏ qua bước này

cd WSSS_2026_RESEARCH
```

### Bước 1.2: Tắt conda base (nếu đang dùng Anaconda)

```bash
# Kiểm tra prompt có (base) không
# Nếu có thì tắt auto-activate:
conda deactivate
conda config --set auto_activate_base false

# Đóng terminal và mở lại để áp dụng
```

### Bước 1.3: Cài đặt system dependencies (chỉ làm 1 lần)

```bash
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-venv

# Kiểm tra Python version (nên >= 3.9)
python3 --version
```

### Bước 1.4: Tạo virtual environment (chỉ làm 1 lần)

```bash
cd ~/WSSS_2026_RESEARCH

# Tạo venv
python3 -m venv venv

# Kích hoạt lần đầu
source venv/bin/activate

# Prompt sẽ thành: (venv) username@server:~/WSSS_2026_RESEARCH$
```

### Bước 1.5: Cài đặt Python packages (chỉ làm 1 lần)

**Cách 1: Dùng installation script (Khuyến nghị)**

```bash
# Đảm bảo venv đã active (có dòng (venv) ở prompt)
chmod +x install.sh
bash install.sh

# Script sẽ:
# - Kiểm tra Python version
# - Cài từng nhóm dependencies
# - Hỏi có muốn cài CONCH/pydensecrf không (optional)
# - Verify installation tự động
# Thời gian: ~10-15 phút
```

**Cách 2: Cài thủ công (nếu script lỗi)**

```bash
# Đảm bảo venv đã active
pip install --upgrade pip setuptools wheel

# 1. Core dependencies
pip install torch>=2.0.0 torchvision>=0.15.0
pip install numpy>=1.24.0 opencv-python>=4.7.0 pillow>=9.5.0 tqdm>=4.65.0
pip install protobuf==3.20.3

# 2. Model libraries
pip install timm>=0.9.0 transformers==4.35.2

# 3. Medical imaging (MedCLIP version 0.0.3, not 0.0.4!)
pip install medclip==0.0.3

# 4. CONCH (optional, recommended)
pip install git+https://github.com/Mahmoodlab/CONCH.git

# 5. Configuration
pip install omegaconf>=2.3.0 pyyaml>=6.0

# 6. Utilities
pip install albumentations>=1.3.0 scikit-learn>=1.2.0
pip install matplotlib>=3.7.0 seaborn>=0.12.0 icecream>=2.1.0
pip install ttach>=0.0.3 fast_slic>=0.4.0

# 7. CRF (optional, requires build tools)
sudo apt-get install build-essential python3-dev
pip install git+https://github.com/lucasb-eyer/pydensecrf.git

# 8. HuggingFace
pip install huggingface-hub>=0.15.0
```

**Verify cài đặt thành công:**

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import timm; print('timm OK')"
python -c "import medclip; print('medclip OK')"
python -c "from src.models.network import ClsNetwork; print('Project imports OK')"
```

---

## 2. Kích hoạt môi trường hàng ngày

**Mỗi khi mở terminal mới**, làm theo các bước sau:

```bash
# Bước 1: Di chuyển vào thư mục project
cd ~/Desktop/WSSS_2026_RESEARCH
# Hoặc nếu ở home:
# cd ~/WSSS_2026_RESEARCH

# Bước 2: Kích hoạt venv
source venv/bin/activate

# Prompt sẽ thành: (venv) username@server:~/WSSS_2026_RESEARCH$

# Bước 3: Kiểm tra Python đang dùng (optional)
which python
# Output phải là: /home/.../WSSS_2026_RESEARCH/venv/bin/python
```

**Hoàn tất! Bây giờ có thể chạy training.**

---

## 3. Training Model

### Bước 3.1: Chuẩn bị dữ liệu

```bash
# Đảm bảo data đã có ở đúng đường dẫn
# Kiểm tra config:
cat configs/default.yaml | grep -A5 "dataset:"

# Đường dẫn cần có:
# - train_root: /path/to/training
# - val_root: /path/to/validation  
# - test_root: /path/to/test
# - label_feature_path: /path/to/bcss_label_fea_pro_3333.pkl
```

### Bước 3.2: Kiểm tra config

```bash
# Xem toàn bộ config
cat configs/default.yaml

# Các tham số quan trọng:
# - model.hf_token: "hf_xxx"  # HuggingFace token
# - train.samples_per_gpu: 4   # Batch size (giảm nếu OOM)
# - train.epoch: 8             # Số epochs
# - train.use_vq: true         # Bật VQ codebook
# - train.use_text_proto: true # Bật text prototypes
```

### Bước 3.3: Chạy training

```bash
# Training với config mặc định
python scripts/train.py \
    --config configs/default.yaml \
    --gpu 0

# Training với GPU khác (nếu có nhiều GPU)
python scripts/train.py \
    --config configs/default.yaml \
    --gpu 1

# Training với custom config
python scripts/train.py \
    --config configs/my_experiment.yaml \
    --gpu 0
```

### Bước 3.4: Theo dõi training

```bash
# Training log sẽ hiển thị:
# - Elapsed time / ETA
# - Learning rate
# - Loss values
# - Accuracy (all_acc4 / avg_acc4)
# - Validation metrics mỗi epoch:
#   * mIoU (mean Intersection over Union)
#   * fwIoU (frequency-weighted IoU)
#   * Boundary IoU
#   * Dice score

# Ví dụ output:
# Iter: 100/1000; Elapsed: 0:05:23; ETA: 0:48:30; LR: 5.000e-05; Loss: 0.3421; Acc4: 78.42/85.31
# 
# [VAL]  mIoU: 65.32% | fwIoU: 71.23% | BdIoU: 58.45% | Dice: 78.91%
# [TEST] mIoU: 63.87% | fwIoU: 69.56% | BdIoU: 57.12% | Dice: 77.34%
```

### Bước 3.5: Outputs sau khi training

```bash
# Checkpoint được lưu tại:
checkpoints/YYYY-MM-DD-HH-MM/best_cam.pth

# Ví dụ:
# checkpoints/2026-03-09-14-30/best_cam.pth

# Kiểm tra checkpoint vừa tạo:
ls -lh checkpoints/*/best_cam.pth
```

---

## 4. Generate CAM/Masks

### Bước 4.1: Generate cho tập test

```bash
# Sử dụng best checkpoint vừa train
python scripts/inference.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/2026-03-09-14-30/best_cam.pth \
    --split test \
    --gpu 0

# Hoặc dùng wildcard để tự tìm checkpoint mới nhất:
python scripts/inference.py \
    --config configs/default.yaml \
    --checkpoint $(ls -t checkpoints/*/best_cam.pth | head -1) \
    --split test \
    --gpu 0
```

### Bước 4.2: Generate cho tập train (để làm pseudo labels)

```bash
python scripts/inference.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/2026-03-09-14-30/best_cam.pth \
    --split train \
    --gpu 0
```

### Bước 4.3: Generate cho tập validation

```bash
python scripts/inference.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/2026-03-09-14-30/best_cam.pth \
    --split val \
    --gpu 0
```

### Bước 4.4: Kiểm tra outputs

```bash
# Masks được lưu tại:
ls predictions/*.png

# Xem số lượng masks đã tạo:
ls predictions/*.png | wc -l

# Xem mask bằng image viewer:
eog predictions/sample_image.png  # Ubuntu
# hoặc
feh predictions/sample_image.png  # Lightweight viewer
```

**Định dạng mask:**
- File PNG với palette colors
- Mỗi pixel = 1 class:
  - 0: TUM (Tumor) - Đỏ
  - 1: STR (Stroma) - Xanh lá
  - 2: LYM (Lymphocyte) - Xanh dương
  - 3: NEC (Necrosis) - Tím
  - 4: BACK (Background) - Trắng

---

## 5. Visualization

### Bước 5.1: Generate framework images (cho paper/presentation)

```bash
python scripts/visualize.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/2026-03-09-14-30/best_cam.pth \
    --output_dir framework_figs \
    --gpu 0

# Với ảnh cụ thể:
python scripts/visualize.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/2026-03-09-14-30/best_cam.pth \
    --target_image "TCGA-EW-A1P7-DX1_xmin9808_ymin46395_MPP-0.2500+52.png" \
    --output_dir my_visualizations \
    --gpu 0
```

### Bước 5.2: Kiểm tra visualizations

```bash
# Xem danh sách file đã generate:
ls framework_figs/

# Output sẽ có:
# - 01_input.png                    # Input image
# - 02_cam_scale1_56x56.png         # CAM scale 1
# - 02_cam_scale2_28x28.png         # CAM scale 2
# - 02_cam_scale3_14x14.png         # CAM scale 3
# - 02_cam_scale4_7x7.png           # CAM scale 4
# - 03_cam4_TUM.png                 # Per-class CAMs
# - 03_cam4_STR.png
# - 03_cam4_LYM.png
# - 03_cam4_NEC.png
# - 04_pseudo_seg.png               # Predicted segmentation
# - 05_ground_truth.png             # Ground truth
# - 06_fg_masked.png                # Foreground masked
# - 06_bg_masked.png                # Background masked
# - 07_cam_overlay.png              # CAM overlay on input
# - 08_legend.png                   # Class legend
# - vq_detail/                      # VQ codebook analysis (nếu use_vq=true)

# Xem VQ details:
ls framework_figs/vq_detail/
```

---

## 6. Troubleshooting

### ❌ Lỗi: Command 'python' not found

```bash
# Dùng python3 thay vì python
python3 scripts/train.py ...

# Hoặc tạo alias:
alias python=python3
```

### ❌ Lỗi: ModuleNotFoundError: No module named 'torch'

```bash
# Kiểm tra venv có active không:
# Prompt phải có (venv) ở đầu
# Nếu không, chạy:
source venv/bin/activate

# Nếu vẫn lỗi, cài lại:
bash install.sh
# hoặc cài thủ công theo hướng dẫn ở Bước 1.5
```

### ❌ Lỗi: Could not find a version that satisfies the requirement medclip>=0.0.4

```bash
# MedCLIP chỉ có version 0.0.1, 0.0.2, 0.0.3 (không có 0.0.4)
# Fix: Cài version 0.0.3
pip install medclip==0.0.3

# Hoặc dùng install script đã fix sẵn:
bash install.sh
```

### ❌ Lỗi: CUDA out of memory

**Giải pháp 1: Giảm batch size**

```yaml
# Sửa trong configs/default.yaml:
train:
  samples_per_gpu: 2  # Giảm từ 4 → 2
  accum_steps: 8      # Tăng từ 4 → 8 (giữ effective batch size)
```

**Giải pháp 2: Dùng CPU (chậm hơn nhiều)**

```bash
# Trong code, device sẽ tự động fallback sang CPU nếu không có CUDA
python scripts/train.py --config configs/default.yaml --gpu 0
```

### ❌ Lỗi: HuggingFace token invalid

```bash
# Lấy token mới tại: https://huggingface.co/settings/tokens
# Cần có quyền access model MahmoodLab/uni

# Sửa trong configs/default.yaml:
model:
  hf_token: "hf_YOUR_NEW_TOKEN_HERE"
```

### ❌ Lỗi: FileNotFoundError - data path

```bash
# Kiểm tra đường dẫn trong config:
cat configs/default.yaml | grep "root:"

# Sửa đường dẫn cho đúng với server:
dataset:
  train_root: /path/to/your/data/training
  val_root: /path/to/your/data/validation
  test_root: /path/to/your/data/test
```

### ❌ Training bị đơ/treo

```bash
# Kiểm tra GPU đang làm gì:
nvidia-smi

# Kiểm tra process:
ps aux | grep python

# Kill process nếu cần:
pkill -f "python scripts/train.py"

# Xem log chi tiết hơn:
python scripts/train.py --config configs/default.yaml --gpu 0 2>&1 | tee training.log
```

### ⚠️ Cảnh báo: pydensecrf not installed

```bash
# Không ảnh hưởng training, chỉ ảnh hưởng CRF post-processing
# Nếu muốn dùng CRF, cài thêm:
pip install git+https://github.com/lucasb-eyer/pydensecrf.git

# Enable CRF trong config:
train:
  use_crf: true
```

---

## 📊 Workflow hoàn chỉnh

```bash
# === LẦN ĐẦU TIÊN ===
cd ~/Desktop/WSSS_2026_RESEARCH
python3 -m venv venv
source venv/bin/activate
bash install.sh  # hoặc cài thủ công nếu script lỗi

# === MỖI LẦN SỬ DỤNG ===

# 1. Kích hoạt venv
cd ~/Desktop/WSSS_2026_RESEARCH
source venv/bin/activate

# 2. Training
python scripts/train.py --config configs/default.yaml --gpu 0

# 3. Generate masks
CKPT=$(ls -t checkpoints/*/best_cam.pth | head -1)
python scripts/inference.py --config configs/default.yaml \
    --checkpoint $CKPT --split test --gpu 0

# 4. Visualization
python scripts/visualize.py --config configs/default.yaml \
    --checkpoint $CKPT --output_dir framework_figs --gpu 0

# 5. Tắt venv (optional)
deactivate
```

---

## 🔧 Tips hữu ích

### Chạy training trong background (không bị ngắt khi đóng SSH)

```bash
# Dùng nohup:
nohup python scripts/train.py \
    --config configs/default.yaml \
    --gpu 0 > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Xem log real-time:
tail -f training_*.log

# Kiểm tra process:
ps aux | grep train.py
```

### Chạy nhiều experiments song song

```bash
# GPU 0:
python scripts/train.py --config configs/exp1.yaml --gpu 0 &

# GPU 1:
python scripts/train.py --config configs/exp2.yaml --gpu 1 &

# Theo dõi GPU usage:
watch -n 1 nvidia-smi
```

### Backup checkpoints định kỳ

```bash
# Tạo backup:
tar -czf checkpoints_backup_$(date +%Y%m%d).tar.gz checkpoints/

# Copy về local machine:
scp user@server:~/WSSS_2026_RESEARCH/checkpoints_backup_*.tar.gz ./
```

---

## 📚 Tài liệu thêm

- [README.md](README.md) - Tổng quan dự án
- [USAGE.md](USAGE.md) - Hướng dẫn chi tiết API
- [docs/RESTRUCTURING.md](docs/RESTRUCTURING.md) - Chi tiết kiến trúc
- [docs/TODO.md](docs/TODO.md) - Danh sách tính năng

---

## 💡 Hỏi đáp nhanh

**Q: Training mất bao lâu?**  
A: ~2-4 giờ cho 8 epochs với GPU RTX 3090, ~6-8 giờ với RTX 2080 Ti.

**Q: Cần bao nhiêu VRAM?**  
A: Tối thiểu 11GB (batch_size=4), khuyến nghị 16GB+ cho thoải mái.

**Q: Có thể train trên CPU không?**  
A: Có nhưng cực kỳ chậm (~50-100x), không khuyến nghị.

**Q: Làm sao biết model đang train tốt?**  
A: Xem validation mIoU tăng dần mỗi epoch, loss giảm dần.

**Q: Inference mất bao lâu?**  
A: ~5-10 phút cho 200 ảnh (tập test BCSS).

**Q: File wsss.py có cần thiết không?**  
A: Không, chỉ giữ làm reference. Code hiện tại đã modular hóa.

---

**Chúc bạn training thành công! 🎉**

_Last updated: March 9, 2026_
