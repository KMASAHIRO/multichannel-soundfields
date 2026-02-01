#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pickle

import numpy as np
import torch

from scipy import stats
from scipy.signal import hilbert
import scipy
import auraloss


# =========================
#  T60 / EDT 計算 (元実装と同じ式)
# =========================

def t60_EDT_cal(energys, init_db=-5, end_db=-25, factor=3.0, fs=16000):
    t60_all = []
    edt_all = []

    for energy in energys:
        # EDT
        edt_factor = 6.0
        energy_n10db = energy[np.abs(energy - (-10)).argmin()]
        n10db_sample = np.where(energy == energy_n10db)[0][0]
        edt = n10db_sample / fs * edt_factor  # 秒

        # T60
        energy_init = energy[np.abs(energy - init_db).argmin()]
        energy_end  = energy[np.abs(energy - end_db).argmin()]
        init_sample = np.where(energy == energy_init)[0][0]
        end_sample  = np.where(energy == energy_end)[0][0]

        x = np.arange(init_sample, end_sample + 1) / fs
        y = energy[init_sample:end_sample + 1]

        slope, intercept = stats.linregress(x, y)[0:2]

        db_regress_init = (init_db - intercept) / slope
        db_regress_end  = (end_db  - intercept) / slope

        t60 = factor * (db_regress_end - db_regress_init)

        t60_all.append(t60)
        edt_all.append(edt)

    return np.array(t60_all), np.array(edt_all)


# =========================
#  メトリクス計算本体
# =========================

def metric_cal(ori_ir, pred_ir, fs=16000, window=32):
    """
    返す単位:
      angle: unitless
      amp  : unitless
      env  : [%]
      t60  : [%]
      C50  : [dB]
      EDT  : [ms]
    """

    if ori_ir.ndim == 1:
        ori_ir = ori_ir[np.newaxis, :]
    if pred_ir.ndim == 1:
        pred_ir = pred_ir[np.newaxis, :]

    # === STFT loss（元実装通り計算するが返さない） ===
    multi_stft = auraloss.freq.MultiResolutionSTFTLoss(
        w_lin_mag=1,
        fft_sizes=[512, 256, 128],
        win_lengths=[300, 150, 75],
        hop_sizes=[60, 30, 8],
    )
    _ = multi_stft(torch.tensor(ori_ir).unsqueeze(1),
                   torch.tensor(pred_ir).unsqueeze(1))

    # ========= 1) angle error =========
    fft_ori     = np.fft.fft(ori_ir, axis=-1)
    fft_predict = np.fft.fft(pred_ir, axis=-1)

    cos_ori  = np.cos(np.angle(fft_ori))
    cos_pred = np.cos(np.angle(fft_predict))
    sin_ori  = np.sin(np.angle(fft_ori))
    sin_pred = np.sin(np.angle(fft_predict))

    angle_diff = np.abs(cos_ori - cos_pred) + np.abs(sin_ori - sin_pred)
    angle_error_mean = angle_diff.mean()
    angle_error_std  = angle_diff.std()

    # ========= 2) amplitude error =========
    amp_ori = scipy.ndimage.convolve1d(
        np.abs(fft_ori), np.ones(window), axis=-1
    )
    amp_predict = scipy.ndimage.convolve1d(
        np.abs(fft_predict), np.ones(window), axis=-1
    )

    amp_rel_diff = np.abs(amp_ori - amp_predict) / amp_ori
    amp_error_mean = amp_rel_diff.mean()
    amp_error_std  = amp_rel_diff.std()

    # ========= 3) envelope error (% に変更) =========
    ori_env  = np.abs(hilbert(ori_ir))
    pred_env = np.abs(hilbert(pred_ir))

    env_rel_diff = np.abs(ori_env - pred_env) / np.max(ori_env, axis=1, keepdims=True)

    # ★★★ % に変換 ★★★
    env_error_vals = env_rel_diff * 100.0
    env_error_mean = env_error_vals.mean()
    env_error_std  = env_error_vals.std()

    # ========= 4) energy, T60(%), EDT(ms) =========
    ori_energy = 10.0 * np.log10(
        np.cumsum(ori_ir[:, ::-1]**2 + 1e-9, axis=-1)[:, ::-1]
    )
    pred_energy = 10.0 * np.log10(
        np.cumsum(pred_ir[:, ::-1]**2 + 1e-9, axis=-1)[:, ::-1]
    )

    ori_energy  -= ori_energy[:, 0].reshape(-1, 1)
    pred_energy -= pred_energy[:, 0].reshape(-1, 1)

    ori_t60, ori_edt     = t60_EDT_cal(ori_energy, fs=fs)
    pred_t60, pred_edt   = t60_EDT_cal(pred_energy, fs=fs)

    # T60 → % （相対誤差×100）
    t60_rel_err   = np.abs(ori_t60 - pred_t60) / ori_t60
    t60_error_vals = t60_rel_err * 100.0
    t60_error_mean = t60_error_vals.mean()
    t60_error_std  = t60_error_vals.std()

    # EDT → ms
    edt_abs_err    = np.abs(ori_edt - pred_edt)
    edt_error_vals = edt_abs_err * 1000.0  # → ms
    edt_error_mean = edt_error_vals.mean()
    edt_error_std  = edt_error_vals.std()

    # ========= 5) C50 (dB) =========
    base_sample  = 0
    samples_50ms = int(0.05 * fs)

    energy_ori_early = np.sum(ori_ir[:, base_sample:samples_50ms]**2, axis=-1)
    energy_ori_late  = np.sum(ori_ir[:, samples_50ms:]**2, axis=-1)
    energy_pred_early = np.sum(pred_ir[:, base_sample:samples_50ms]**2, axis=-1)
    energy_pred_late  = np.sum(pred_ir[:, samples_50ms:]**2, axis=-1)

    C50_ori  = 10.0 * np.log10(energy_ori_early / energy_ori_late)
    C50_pred = 10.0 * np.log10(energy_pred_early / energy_pred_late)

    C50_abs_err = np.abs(C50_ori - C50_pred)  # dB
    C50_error_mean = C50_abs_err.mean()
    C50_error_std  = C50_abs_err.std()

    return (
        angle_error_mean,
        amp_error_mean,
        env_error_mean,   # [%]
        t60_error_mean,   # [%]
        C50_error_mean,   # [dB]
        edt_error_mean,   # [ms]
        angle_error_std,
        amp_error_std,
        env_error_std,    # [%]
        t60_error_std,    # [%]
        C50_error_std,
        edt_error_std,    # [ms]
    )


# =========================
#  npz → IR
# =========================

def load_ir_from_npz(npz_path: str):
    data = np.load(npz_path)

    if "pred_sig" in data and "ori_sig" in data:
        pred_spec = data["pred_sig"]
        ori_spec  = data["ori_sig"]
    else:
        raise KeyError(f"{npz_path} に pred_sig / ori_sig がありません。")

    pred_t = torch.fft.irfft(torch.tensor(pred_spec, dtype=torch.cfloat), dim=-1).real
    ori_t  = torch.fft.irfft(torch.tensor(ori_spec, dtype=torch.cfloat), dim=-1).real

    return ori_t.cpu().numpy(), pred_t.cpu().numpy()


# =========================
#  メイン処理
# =========================

def main():
    parser = argparse.ArgumentParser(description="Compute IR metrics from a npz file.")
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--window", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ori_ir, pred_ir = load_ir_from_npz(args.npz_path)

    (
        angle_mean,
        amp_mean,
        env_mean,
        t60_mean,
        c50_mean,
        edt_mean,
        angle_std,
        amp_std,
        env_std,
        t60_std,
        c50_std,
        edt_std,
    ) = metric_cal(ori_ir, pred_ir, fs=args.fs, window=args.window)

    metrics = {
        "npz_path": args.npz_path,
        "fs": args.fs,
        "window": args.window,
        "num_signals": int(ori_ir.shape[0]),
        "angle_mean": float(angle_mean),
        "amp_mean": float(amp_mean),
        "env_mean_percent": float(env_mean),        # ★ [%]
        "t60_mean_percent": float(t60_mean),        # ★ [%]
        "C50_mean_db": float(c50_mean),
        "EDT_mean_ms": float(edt_mean),             # ★ [ms]
        "angle_std": float(angle_std),
        "amp_std": float(amp_std),
        "env_std_percent": float(env_std),          # ★ [%]
        "t60_std_percent": float(t60_std),          # ★ [%]
        "C50_std_db": float(c50_std),
        "EDT_std_ms": float(edt_std),
    }

    base = os.path.splitext(os.path.basename(args.npz_path))[0]
    txt_path = os.path.join(args.outdir, f"{base}_metrics.txt")
    pkl_path = os.path.join(args.outdir, f"{base}_metrics.pkl")

    # --- txt 出力 ---
    lines = []
    lines.append("# IR evaluation metrics\n")
    lines.append(f"npz_path   : {args.npz_path}\n")
    lines.append(f"fs         : {args.fs}\n")
    lines.append(f"window     : {args.window}\n")
    lines.append(f"num_signals: {metrics['num_signals']}\n\n")
    lines.append("Metric order: angle, amp, env(%), t60(%), C50(dB), EDT(ms)\n\n")

    lines.append("=== Mean errors ===\n")
    lines.append(f"angle_mean         : {angle_mean:.6f}\n")
    lines.append(f"amp_mean           : {amp_mean:.6f}\n")
    lines.append(f"env_mean_percent   : {env_mean:.6f}\n")
    lines.append(f"t60_mean_percent   : {t60_mean:.6f}\n")
    lines.append(f"C50_mean_db        : {c50_mean:.6f}\n")
    lines.append(f"EDT_mean_ms        : {edt_mean:.6f}\n\n")

    lines.append("=== Std of errors ===\n")
    lines.append(f"angle_std          : {angle_std:.6f}\n")
    lines.append(f"amp_std            : {amp_std:.6f}\n")
    lines.append(f"env_std_percent    : {env_std:.6f}\n")
    lines.append(f"t60_std_percent    : {t60_std:.6f}\n")
    lines.append(f"C50_std_db         : {c50_std:.6f}\n")
    lines.append(f"EDT_std_ms         : {edt_std:.6f}\n")

    with open(txt_path, "w") as f:
        f.writelines(lines)

    # --- pkl 出力 ---
    with open(pkl_path, "wb") as f:
        pickle.dump(metrics, f)

    print(f"[INFO] Saved txt  -> {txt_path}")
    print(f"[INFO] Saved pkl  -> {pkl_path}")


if __name__ == "__main__":
    main()
