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

## 3. Train the WSSS CAM model

```bash
venv/bin/python scripts/train.py \
  --config configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml \
  --gpu 0 \
  --seed 42 \
  --epoch 8
```

## 4. Generate pseudo masks

```bash
venv/bin/python scripts/inference.py \
  --config configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml \
  --checkpoint <path/to/best_cam.pth> \
  --split test \
  --gpu 0
```
