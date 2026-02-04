#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import yaml

import numpy as np

from IR_metrics import ir_metrics
from DoA_metrics import doa_metrics


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("doa_algorithm", "NormMUSIC")
    cfg.setdefault("fs", 16000)
    cfg.setdefault("n_fft", 512)
    cfg.setdefault("hop_size", 128)
    cfg.setdefault("window", "hann")
    return cfg


def load_results(npz_path: str):
    data = np.load(npz_path)
    required = ["position_tx", "position_rx", "ir_pred"]
    for k in required:
        if k not in data:
            raise KeyError(f"{npz_path} is missing key: {k}")

    position_tx = data["position_tx"].astype(np.float32)
    position_rx = data["position_rx"].astype(np.float32)
    ir_pred = data["ir_pred"].astype(np.float32)
    ir_gt = data["ir_gt"].astype(np.float32) if "ir_gt" in data else None
    return position_tx, position_rx, ir_pred, ir_gt


def main():
    parser = argparse.ArgumentParser(description="Compute IR/DoA metrics")
    parser.add_argument("--config", required=True, help="Path to metrics_config.yml")
    parser.add_argument("--input", required=True, help="Path to results.npz")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    position_tx, position_rx, ir_pred, ir_gt = load_results(args.input)

    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy2(args.config, os.path.join(args.output_dir, "metrics_config.yml"))

    doa_gt_deg, doa_pred_deg, doa_true_deg = doa_metrics(
        ir_gt=ir_gt,
        ir_pred=ir_pred,
        algorithm=cfg["doa_algorithm"],
        n_fft=int(cfg["n_fft"]),
        fs=int(cfg["fs"]),
        rx_positions=position_rx,
        tx_positions=position_tx,
        hop_size=int(cfg["hop_size"]),
        window=str(cfg["window"]),
    )

    out = {
        "position_tx": position_tx,
        "position_rx": position_rx,
        "ir_pred": ir_pred,
        "doa_true_deg": doa_true_deg.astype(np.float32),
        "doa_pred_deg": doa_pred_deg.astype(np.float32),
    }

    if ir_gt is not None:
        out["ir_gt"] = ir_gt
        out["doa_gt_deg"] = doa_gt_deg.astype(np.float32)

        metrics = ir_metrics(ir_gt, ir_pred, fs=int(cfg["fs"]))
        out["metric_angle"] = metrics["angle_error"].astype(np.float32)
        out["metric_amp"] = metrics["amp_error"].astype(np.float32)
        out["metric_env_pct"] = metrics["env_error_percent"].astype(np.float32)
        out["metric_t60_pct"] = metrics["t60_error_percent"].astype(np.float32)
        out["metric_c50_db"] = metrics["c50_error_db"].astype(np.float32)
        out["metric_edt_ms"] = metrics["edt_error_ms"].astype(np.float32)

    np.savez(os.path.join(args.output_dir, "metrics_results.npz"), **out)


if __name__ == "__main__":
    main()
