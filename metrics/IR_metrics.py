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


def ir_metrics(ir_gt, ir_pred, fs=16000, window=32):
    """
    Compute IR metrics without reduction.

    Parameters
    ----------
    ir_gt : np.ndarray
        Ground-truth IR. Shape (..., T).
    ir_pred : np.ndarray
        Predicted IR. Shape (..., T).
    fs : int
        Sampling rate.
    window : int
        Window size for amplitude smoothing.

    Returns
    -------
    metrics : dict
        Each value has shape ir_gt.shape[:-1].
    """
    if ir_gt is None or ir_pred is None:
        raise ValueError("ir_gt and ir_pred must be provided.")

    ir_gt = np.asarray(ir_gt)
    ir_pred = np.asarray(ir_pred)
    if ir_gt.shape != ir_pred.shape:
        raise ValueError(f"Shape mismatch: ir_gt {ir_gt.shape} vs ir_pred {ir_pred.shape}")
    if ir_gt.ndim < 1:
        raise ValueError("ir_gt must have at least 1 dimension.")

    lead_shape = ir_gt.shape[:-1]
    t_len = ir_gt.shape[-1]
    flat_gt = ir_gt.reshape(-1, t_len)
    flat_pred = ir_pred.reshape(-1, t_len)

    # STFT loss (computed but not returned, matches original behavior)
    multi_stft = auraloss.freq.MultiResolutionSTFTLoss(
        w_lin_mag=1,
        fft_sizes=[512, 256, 128],
        win_lengths=[300, 150, 75],
        hop_sizes=[60, 30, 8],
    )
    _ = multi_stft(
        torch.tensor(flat_gt).unsqueeze(1),
        torch.tensor(flat_pred).unsqueeze(1),
    )

    # 1) angle error
    fft_gt = np.fft.fft(flat_gt, axis=-1)
    fft_pred = np.fft.fft(flat_pred, axis=-1)
    cos_gt = np.cos(np.angle(fft_gt))
    cos_pred = np.cos(np.angle(fft_pred))
    sin_gt = np.sin(np.angle(fft_gt))
    sin_pred = np.sin(np.angle(fft_pred))
    angle_diff = np.abs(cos_gt - cos_pred) + np.abs(sin_gt - sin_pred)
    angle_error = angle_diff.mean(axis=-1)

    # 2) amplitude error
    amp_gt = scipy.ndimage.convolve1d(np.abs(fft_gt), np.ones(window), axis=-1)
    amp_pred = scipy.ndimage.convolve1d(np.abs(fft_pred), np.ones(window), axis=-1)
    amp_rel_diff = np.abs(amp_gt - amp_pred) / amp_gt
    amp_error = amp_rel_diff.mean(axis=-1)

    # 3) envelope error (%)
    gt_env = np.abs(hilbert(flat_gt))
    pred_env = np.abs(hilbert(flat_pred))
    env_rel_diff = np.abs(gt_env - pred_env) / np.max(gt_env, axis=1, keepdims=True)
    env_error = (env_rel_diff * 100.0).mean(axis=-1)

    # 4) energy, T60(%), EDT(ms)
    gt_energy = 10.0 * np.log10(
        np.cumsum(flat_gt[:, ::-1] ** 2 + 1e-9, axis=-1)[:, ::-1]
    )
    pred_energy = 10.0 * np.log10(
        np.cumsum(flat_pred[:, ::-1] ** 2 + 1e-9, axis=-1)[:, ::-1]
    )
    gt_energy -= gt_energy[:, 0].reshape(-1, 1)
    pred_energy -= pred_energy[:, 0].reshape(-1, 1)

    gt_t60, gt_edt = t60_EDT_cal(gt_energy, fs=fs)
    pred_t60, pred_edt = t60_EDT_cal(pred_energy, fs=fs)

    t60_error = (np.abs(gt_t60 - pred_t60) / gt_t60) * 100.0
    edt_error = np.abs(gt_edt - pred_edt) * 1000.0

    # 5) C50 (dB)
    base_sample = 0
    samples_50ms = int(0.05 * fs)
    energy_gt_early = np.sum(flat_gt[:, base_sample:samples_50ms] ** 2, axis=-1)
    energy_gt_late = np.sum(flat_gt[:, samples_50ms:] ** 2, axis=-1)
    energy_pred_early = np.sum(flat_pred[:, base_sample:samples_50ms] ** 2, axis=-1)
    energy_pred_late = np.sum(flat_pred[:, samples_50ms:] ** 2, axis=-1)
    c50_gt = 10.0 * np.log10(energy_gt_early / energy_gt_late)
    c50_pred = 10.0 * np.log10(energy_pred_early / energy_pred_late)
    c50_error = np.abs(c50_gt - c50_pred)

    return {
        "angle_error": angle_error.reshape(lead_shape),
        "amp_error": amp_error.reshape(lead_shape),
        "env_error_percent": env_error.reshape(lead_shape),
        "t60_error_percent": t60_error.reshape(lead_shape),
        "c50_error_db": c50_error.reshape(lead_shape),
        "edt_error_ms": edt_error.reshape(lead_shape),
    }
