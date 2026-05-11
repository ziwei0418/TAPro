#!/usr/bin/env python3
"""Run paper-grade WSSS ablations with robust continue-on-error behavior.

Tables:
  1. Main component ablation.
  2. Encoder/prototype-source ablation with the full method.
  3. Hyperparameter/design sensitivity for the full method.

The script writes:
  - configs/*.yaml for each runnable row/seed
  - logs/*.txt for each run
  - summary_raw.tsv with one row per run
  - summary_mean_std.txt grouped by ablation row

Unsupported rows are recorded as SKIPPED_UNSUPPORTED rather than faked.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/experiments/bcss_fixed_0p6_feature_region_abdiv_beta2.yaml"


@dataclass(frozen=True)
class RunSpec:
    table: str
    name: str
    desc: str
    cfg: object | None
    supported: bool = True
    reason: str = ""


def clone_cfg(base):
    return OmegaConf.create(OmegaConf.to_container(base, resolve=True))


def set_dirs(cfg, run_root: Path, name: str, seed: int):
    base = run_root / "work" / f"{name}_seed{seed}"
    cfg.work_dir.ckpt_dir = str(base / "checkpoints")
    cfg.work_dir.pred_dir = str(base / "predictions")
    cfg.work_dir.train_log_dir = str(base)


def disable_abtagd(cfg):
    cfg.train.use_cam_power_calibration = False
    cfg.train.use_feature_region_refine = False
    cfg.train.use_attention_boundary_refine = False


def enable_full_abtagd(cfg):
    cfg.train.use_cam_power_calibration = True
    cfg.train.use_feature_region_refine = True
    cfg.train.use_attention_boundary_refine = True
    cfg.train.hybrid_alpha_mode = "fixed"
    cfg.train.hybrid_alpha_fixed = 0.6


def set_visual_only(cfg):
    cfg.train.use_text_proto = False
    cfg.train.use_hybrid_proto_blend = False


def set_text_only(cfg, text_encoder_mode: str = "conch"):
    cfg.train.use_text_proto = True
    cfg.train.use_hybrid_proto_blend = True
    cfg.train.hybrid_alpha_mode = "fixed"
    cfg.train.hybrid_alpha_fixed = 1.0
    cfg.train.text_encoder_mode = text_encoder_mode


def set_hybrid(cfg, alpha: float = 0.6, text_encoder_mode: str = "conch"):
    cfg.train.use_text_proto = True
    cfg.train.use_hybrid_proto_blend = True
    cfg.train.hybrid_alpha_mode = "fixed"
    cfg.train.hybrid_alpha_fixed = alpha
    cfg.train.text_encoder_mode = text_encoder_mode


def build_specs(tables: set[str]) -> list[RunSpec]:
    base = OmegaConf.load(BASE_CONFIG)
    specs: list[RunSpec] = []

    if "1" in tables:
        cfg = clone_cfg(base)
        set_visual_only(cfg)
        disable_abtagd(cfg)
        specs.append(RunSpec("1", "t1_visual_baseline", "Visual baseline", cfg))

        cfg = clone_cfg(base)
        set_text_only(cfg, "conch")
        disable_abtagd(cfg)
        specs.append(RunSpec("1", "t1_text", "+ Text", cfg))

        cfg = clone_cfg(base)
        set_hybrid(cfg, 0.6, "conch")
        disable_abtagd(cfg)
        specs.append(RunSpec("1", "t1_hybrid", "+ Hybrid", cfg))

        cfg = clone_cfg(base)
        set_hybrid(cfg, 0.6, "conch")
        enable_full_abtagd(cfg)
        specs.append(RunSpec("1", "t1_full_abtagd", "+ Hybrid + AB-TAGD full", cfg))

    if "2" in tables:
        cfg = clone_cfg(base)
        set_visual_only(cfg)
        enable_full_abtagd(cfg)
        specs.append(RunSpec("2", "t2_uni_none_full", "UNI + none + full AB-TAGD", cfg))

        cfg = clone_cfg(base)
        set_hybrid(cfg, 0.6, "medclip")
        enable_full_abtagd(cfg)
        specs.append(RunSpec("2", "t2_uni_clip_full", "UNI + CLIP/MedCLIP-text + full AB-TAGD", cfg))

        cfg = clone_cfg(base)
        set_hybrid(cfg, 0.6, "conch")
        enable_full_abtagd(cfg)
        specs.append(RunSpec("2", "t2_uni_conch_full", "UNI + CONCH-text + full AB-TAGD", cfg))

        specs.append(RunSpec(
            "2",
            "t2_phikon_conch_full",
            "Phikon + CONCH-text + full AB-TAGD",
            None,
            supported=False,
            reason="WSSS visual encoder is currently hard-coded to UNIEncoder; Phikon adapter not implemented.",
        ))
        specs.append(RunSpec(
            "2",
            "t2_conchvision_conch_full",
            "CONCH-vision + CONCH-text + full AB-TAGD",
            None,
            supported=False,
            reason="WSSS visual encoder is currently hard-coded to UNIEncoder; CONCH-vision dense-feature adapter not implemented.",
        ))

    if "3" in tables:
        for beta in (0.0, 1.0, 2.0, 3.0):
            cfg = clone_cfg(base)
            set_hybrid(cfg, 0.6, "conch")
            enable_full_abtagd(cfg)
            cfg.train.attention_boundary_beta = beta
            specs.append(RunSpec("3", f"t3_beta_{str(beta).replace('.', 'p')}", f"ABDiv beta={beta:g}", cfg))

        cfg = clone_cfg(base)
        set_hybrid(cfg, 0.6, "conch")
        enable_full_abtagd(cfg)
        cfg.train.use_cam_power_calibration = False
        specs.append(RunSpec("3", "t3_no_activation_balance", "Full without activation balance", cfg))

        cfg = clone_cfg(base)
        set_hybrid(cfg, 0.6, "conch")
        enable_full_abtagd(cfg)
        cfg.train.feature_region_class_blends = None
        cfg.train.feature_region_blend = 0.35
        specs.append(RunSpec("3", "t3_global_lambda_035", "Full with global lambda=0.35", cfg))

        cfg = clone_cfg(base)
        set_hybrid(cfg, 0.6, "conch")
        enable_full_abtagd(cfg)
        cfg.train.cam_power_values = [1.10, 1.10, 0.90, 1.00]
        cfg.train.cam_power_blend = 0.25
        specs.append(RunSpec("3", "t3_weak_activation_balance", "Full with weak activation balance", cfg))

    return specs


def stream_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        proc.wait()
        log.write(f"\n[returncode] {proc.returncode}\n")
        return proc.returncode


def _parse_per_class(text: str) -> str:
    return ",".join(s.strip().strip("'\"") for s in text.split(","))


def extract_best_metrics(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    val = re.findall(
        r"\[VAL\]\s+mIoU:\s+([0-9.]+)%\s+\|\s+fwIoU:\s+([0-9.]+)%\s+\|\s+BdIoU:\s+([0-9.]+)%\s+\|\s+Dice:\s+([0-9.]+)%.*?per-class IoU:\s+\[([^\]]+)\]",
        text,
        re.S,
    )
    test = re.findall(
        r"\[TEST\]\s+mIoU:\s+([0-9.]+)%\s+\|\s+fwIoU:\s+([0-9.]+)%\s+\|\s+BdIoU:\s+([0-9.]+)%\s+\|\s+Dice:\s+([0-9.]+)%.*?per-class IoU:\s+\[([^\]]+)\]",
        text,
        re.S,
    )
    if not val or not test:
        return {}
    pairs = list(zip(val, test))
    best_idx = max(range(len(pairs)), key=lambda i: float(pairs[i][0][0]))
    best_val, best_test = pairs[best_idx]
    return {
        "best_epoch": best_idx + 1,
        "val_miou": float(best_val[0]),
        "val_fwiou": float(best_val[1]),
        "val_biou": float(best_val[2]),
        "val_dice": float(best_val[3]),
        "val_per_class": _parse_per_class(best_val[4]),
        "test_miou": float(best_test[0]),
        "test_fwiou": float(best_test[1]),
        "test_biou": float(best_test[2]),
        "test_dice": float(best_test[3]),
        "test_per_class": _parse_per_class(best_test[4]),
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def write_summaries(run_root: Path, rows: list[dict[str, object]]):
    raw_path = run_root / "summary_raw.tsv"
    fields = [
        "table", "name", "desc", "seed", "status", "returncode", "best_epoch",
        "val_miou", "val_fwiou", "val_biou", "val_dice", "val_per_class",
        "test_miou", "test_fwiou", "test_biou", "test_dice", "test_per_class",
        "log", "reason",
    ]
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(str(row.get(field, "")) for field in fields))
    raw_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        if row.get("status") == "OK":
            grouped.setdefault((str(row["table"]), str(row["name"])), []).append(row)

    summary_lines = [
        f"WSSS paper ablation summary: {run_root}",
        "Values are BEST-BY-VAL over each run.",
        "",
    ]
    for table, name in sorted(grouped):
        group = grouped[(table, name)]
        desc = str(group[0]["desc"])
        summary_lines.append(f"[Table {table}] {name} | {desc} | n={len(group)}")
        for metric in ("test_miou", "test_fwiou", "test_biou", "test_dice", "val_miou"):
            vals = [float(r[metric]) for r in group if r.get(metric) not in ("", None)]
            mean, std = mean_std(vals)
            summary_lines.append(f"  {metric}: {mean:.2f} ± {std:.2f}")
        per_seed = ", ".join(f"seed{r['seed']}={float(r['test_miou']):.2f}" for r in group)
        summary_lines.append(f"  per-seed TEST mIoU: {per_seed}")
        summary_lines.append("")

    skipped = [r for r in rows if r.get("status", "").startswith("SKIPPED")]
    failed = [r for r in rows if r.get("status") == "FAILED"]
    if skipped:
        summary_lines.append("Skipped rows:")
        for row in skipped:
            summary_lines.append(f"  [{row['table']}] {row['name']} seed{row['seed']}: {row.get('reason', '')}")
        summary_lines.append("")
    if failed:
        summary_lines.append("Failed rows:")
        for row in failed:
            summary_lines.append(f"  [{row['table']}] {row['name']} seed{row['seed']}: rc={row.get('returncode')} log={row.get('log')}")
        summary_lines.append("")

    (run_root / "summary_mean_std.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 3407, 2026])
    parser.add_argument("--epoch", type=int, default=8)
    parser.add_argument("--tables", nargs="+", default=["1", "2", "3"], choices=["1", "2", "3"])
    parser.add_argument("--start-from", type=str, default=None)
    args = parser.parse_args()

    run_root = ROOT / f"train_log/wsss_paper_ablation_{dt.datetime.now():%Y%m%d_%H%M%S}"
    config_dir = run_root / "configs"
    log_dir = run_root / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs(set(args.tables))
    if args.start_from:
        names = [spec.name for spec in specs]
        if args.start_from not in names:
            raise ValueError(f"--start-from must be one of {names}, got {args.start_from}")
        specs = specs[names.index(args.start_from):]

    rows: list[dict[str, object]] = []
    print(f"Run root: {run_root}")
    print(f"Seeds: {args.seeds}")
    print(f"Tables: {args.tables}")

    for spec in specs:
        for seed in args.seeds:
            row_base = {
                "table": spec.table,
                "name": spec.name,
                "desc": spec.desc,
                "seed": seed,
                "reason": spec.reason,
            }
            if not spec.supported:
                rows.append({**row_base, "status": "SKIPPED_UNSUPPORTED", "returncode": ""})
                write_summaries(run_root, rows)
                print(f"SKIP {spec.name} seed{seed}: {spec.reason}")
                continue

            cfg = clone_cfg(spec.cfg)
            cfg.train.epoch = args.epoch
            set_dirs(cfg, run_root, spec.name, seed)
            cfg_path = config_dir / f"{spec.name}_seed{seed}.yaml"
            log_path = log_dir / f"{spec.name}_seed{seed}.txt"
            OmegaConf.save(config=cfg, f=cfg_path)

            print("\n" + "=" * 96)
            print(f"Running Table {spec.table}: {spec.name} | seed={seed}")
            print(f"Desc:   {spec.desc}")
            print(f"Config: {cfg_path}")
            print(f"Log:    {log_path}")
            print("=" * 96)

            cmd = [
                sys.executable,
                "scripts/train.py",
                "--config",
                str(cfg_path.relative_to(ROOT)),
                "--gpu",
                str(args.gpu),
                "--seed",
                str(seed),
                "--epoch",
                str(args.epoch),
            ]
            try:
                rc = stream_command(cmd, log_path)
            except Exception as exc:  # keep the full ablation moving
                log_path.write_text(f"Runner exception:\n{exc}\n", encoding="utf-8")
                rows.append({**row_base, "status": "FAILED", "returncode": "runner_exception", "log": log_path})
                write_summaries(run_root, rows)
                continue

            metrics = extract_best_metrics(log_path)
            status = "OK" if rc == 0 and metrics else "FAILED"
            rows.append({
                **row_base,
                **metrics,
                "status": status,
                "returncode": rc,
                "log": log_path,
            })
            write_summaries(run_root, rows)

    write_summaries(run_root, rows)
    print("\nAblation finished.")
    print(f"Raw summary:      {run_root / 'summary_raw.tsv'}")
    print(f"Mean/std summary: {run_root / 'summary_mean_std.txt'}")


if __name__ == "__main__":
    main()
