# Quickstart

## 1. Create the environment

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional CONCH install:

```bash
python -m pip install "git+https://github.com/Mahmoodlab/CONCH.git"
```

Check imports:

```bash
venv/bin/python -c "import torch; print(torch.__version__)"
venv/bin/python -c "from src.models.network import ClsNetwork; print('imports ok')"
```

## 2. Prepare data

Expected local layout:

```text
data/BCSS-WSSS/{training,val,test}
data/LUAD-HistoSeg/{train,val,test}
data/prototypes/bcss_label_fea_pro_3333.pkl
data/prototypes/luad_label_fea_proto_3333.pkl
```

Dataset files and prototype banks are not committed to git.

## 3. Train the WSSS model

```bash
venv/bin/python scripts/train.py \
  --config configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml \
  --gpu 0 \
  --seed 42 \
  --epoch 8
```

## 4. Run ablations

```bash
venv/bin/python scripts/run_wsss_paper_ablation.py \
  --gpu 0 \
  --seeds 42 3407 2026 \
  --epoch 8 \
  --tables 1 2 3
```

## 5. Evaluate downstream segmentation checkpoints

```bash
venv/bin/python scripts/eval_bcss_seg_no_tta.py \
  --checkpoint checkpoints/seg_bcss/best_ema.pth \
  --out-dir predictions_bcss_seg_test_no_tta

venv/bin/python scripts/eval_luad_seg_no_tta.py \
  --checkpoint checkpoints/seg_luad/best_ema.pth \
  --out-dir predictions_luad_seg_test_no_tta
```

## 6. Build qualitative BCSS SOTA figures

```bash
venv/bin/python scripts/make_bcss_sota_qual_figure.py \
  --rows 2 \
  --top-k 30 \
  --min-margin 0.0
```

The qualitative script writes selected candidates and figure panels to `qualitative_bcss_sota/`.
