# QUICKSTART

## 1. Kích Hoạt Môi Trường

```bash
cd ~/Desktop/WSSS_2026_RESEARCH
source venv/bin/activate
```

Nếu chưa có `venv`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Kiểm tra import cơ bản:

```bash
venv/bin/python -c "import torch; print(torch.__version__)"
venv/bin/python -c "from src.models.network import ClsNetwork; print('imports ok')"
```

## 2. Kiểm Tra Dữ Liệu Và Config

Config full method:

```bash
configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml
```

Các path cần có:

```text
data/BCSS-WSSS/training
data/BCSS-WSSS/val
data/BCSS-WSSS/test
data/prototypes/bcss_label_fea_pro_3333.pkl
```

Kiểm tra nhanh:

```bash
ls data/BCSS-WSSS
ls data/prototypes/bcss_label_fea_pro_3333.pkl
```

Nếu môi trường cần Hugging Face token để load UNI, kiểm tra trường:

```yaml
model:
  hf_token: "..."
```

trong config.

## 3. Chạy Full AB-TAGD

Lệnh chính:

```bash
venv/bin/python scripts/train.py \
  --config configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml \
  --gpu 0 \
  --seed 42 \
  --epoch 8
```

Script sẽ tự làm các bước:

1. Train WSSS model.
2. Evaluate VAL/TEST mỗi epoch.
3. Save `best_cam.pth` theo validation mIoU.
4. Sau training, evaluate lại.
5. Reload best checkpoint.
6. Generate pseudo masks cho toàn bộ training set.

Output theo config:

```text
train_log/bcss_fixed_0p6_feature_region_abdiv_beta2/
├── checkpoints/<timestamp>_seed42/best_cam.pth
└── predictions/*.png
```

Lưu ý: hiện repo đã cleanup, nên thư mục output này có thể chưa tồn tại; script sẽ tạo lại khi chạy.

## 4. Chạy Ablation Paper

Lệnh chạy đủ 3 bảng với 3 seed:

```bash
venv/bin/python scripts/run_wsss_paper_ablation.py \
  --gpu 0 \
  --seeds 42 3407 2026 \
  --epoch 8 \
  --tables 1 2 3
```

Runner sẽ tạo một thư mục mới:

```text
train_log/wsss_paper_ablation_<timestamp>/
├── configs/
├── logs/
├── summary_raw.tsv
├── summary_mean_std.txt
└── work/*/checkpoints/*/best_cam.pth
```

Ablation hiện đã chạy xong và được giữ lại tại:

```text
train_log/wsss_paper_ablation_20260508_065932/
```

Trong thư mục này:

```bash
cat train_log/wsss_paper_ablation_20260508_065932/summary_mean_std.txt
```

Để xem raw từng seed:

```bash
column -t -s $'\t' train_log/wsss_paper_ablation_20260508_065932/summary_raw.tsv | less -S
```

## 5. Nội Dung Các Bảng Ablation

Table 1: component ablation.

```text
Visual baseline
+ Text
+ Hybrid
+ Hybrid + AB-TAGD full
```

Table 2: prototype/source sensitivity.

```text
UNI + none + full AB-TAGD
UNI + CLIP/MedCLIP-text + full AB-TAGD
UNI + CONCH-text + full AB-TAGD
Phikon / CONCH-vision rows: logged as unsupported nếu adapter chưa implement
```

Table 3: robustness/sensitivity của full method.

```text
ABDiv beta sweep
without activation balance
global lambda
weak activation balance
```

Không dùng Table 3 để claim method mới tốt hơn Table 1 nếu chênh lệch nhỏ hơn std. Table 3 nên được viết là sensitivity/robustness analysis.

## 6. Generate Lại Pseudo Masks Từ Checkpoint Ablation

Trong ablation folder hiện tại, các thư mục `predictions/` đã được xóa để tiết kiệm dung lượng. Checkpoint vẫn còn.

Ví dụ lấy checkpoint full method seed 42:

```bash
CKPT=$(find train_log/wsss_paper_ablation_20260508_065932/work/t1_full_abtagd_seed42/checkpoints -name best_cam.pth | head -1)
echo "$CKPT"
```

Nếu cần generate lại pseudo masks, cách an toàn nhất là chạy lại full config hoặc tạo config mới trỏ `work_dir.pred_dir` tới thư mục output mong muốn. Script `scripts/train.py` hiện generate pseudo sau training; `scripts/inference.py` có thể dùng cho inference riêng nếu phù hợp với checkpoint/config đang dùng.

## 7. Visualization

Script visualization được giữ lại:

```bash
venv/bin/python scripts/visualize.py \
  --config configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml \
  --checkpoint <path/to/best_cam.pth> \
  --output_dir framework_figs \
  --gpu 0
```

Nếu script yêu cầu option khác, kiểm tra:

```bash
venv/bin/python scripts/visualize.py --help
```

## 8. File Quan Trọng Cho Paper

Method:

```text
paper/method_abtagd_miccai.tex
paper/method_abtagd_miccai.md
```

Result:

```text
train_log/wsss_paper_ablation_20260508_065932/summary_mean_std.txt
train_log/wsss_paper_ablation_20260508_065932/summary_raw.tsv
```

Code contribution:

```text
src/utils/cam_refine.py
src/engine/wsss.py
src/utils/evaluate.py
src/models/network.py
src/models/encoder.py
```

## 9. Troubleshooting

### CUDA out of memory

Giảm batch/micro-batch trong config:

```yaml
train:
  samples_per_gpu: 2
  accum_steps: 8
```

### Hugging Face token lỗi

Sửa:

```yaml
model:
  hf_token: "hf_xxx"
```

trong config full method.

### Data path lỗi

Kiểm tra:

```bash
grep -n "root\\|label_feature_path" configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml
```

và sửa path cho đúng server.

### Muốn chạy trong tmux

```bash
tmux new -s wsss
venv/bin/python scripts/train.py \
  --config configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml \
  --gpu 0 \
  --seed 42 \
  --epoch 8
```

Detach:

```bash
Ctrl-b d
```

Attach lại:

```bash
tmux attach -t wsss
```

Kill session:

```bash
tmux kill-session -t wsss
```

## 10. Ghi Nhớ

- Method paper hiện tại là WSSS, không phải segmentation stage.
- Full method là `Hybrid + Activation Balance + ABDiv-TAGD`.
- Checkpoint ablation chính đang nằm trong `train_log/wsss_paper_ablation_20260508_065932/work/*/checkpoints`.
- Pseudo predictions trong ablation đã xóa để tiết kiệm dung lượng, checkpoint vẫn còn.
- Hyperparameter values nên để ở phần Experiments/Implementation Details, không nhét vào Method.
