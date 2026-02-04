import argparse
import os
import shutil
import yaml

import numpy as np
import librosa
import pyroomacoustics as pra


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("doa_algorithm", "NormMUSIC")
    cfg.setdefault("fs", 16000)
    cfg.setdefault("n_fft", 1024)
    cfg.setdefault("hop_size", 512)
    cfg.setdefault("window", "hann")
    cfg.setdefault("noise_seconds", 100)
    cfg.setdefault("random_seed", 0)
    cfg.setdefault("split_time_frame", 64)
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


def true_angle_deg(tx_xy: np.ndarray, rx_center_xy: np.ndarray) -> float:
    dx, dy = tx_xy[0] - rx_center_xy[0], tx_xy[1] - rx_center_xy[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def white_noise(seconds: float, fs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(round(seconds * fs))).astype(np.float32)


def stft_multi(y: np.ndarray, n_fft: int, hop_size: int, window: str) -> np.ndarray:
    win = "boxcar" if window in ("none", "rect", "") else window
    specs = []
    for ch in range(y.shape[0]):
        X = librosa.stft(
            y[ch], n_fft=n_fft, hop_length=hop_size, window=win, center=True
        )
        specs.append(X.astype(np.complex64))
    return np.stack(specs, axis=0)


def doa_sliding(X: np.ndarray, mic: np.ndarray, algorithm: str, fs: int, n_fft: int, split_time_frame: int):
    T = X.shape[-1]
    if T < split_time_frame:
        return np.array([], dtype=np.float32)
    doa = pra.doa.algorithms[algorithm](mic, fs=fs, nfft=n_fft)

    angs = []
    for t0 in range(0, T - split_time_frame + 1, split_time_frame):
        Xseg = X[:, :, t0:t0 + split_time_frame]
        doa.locate_sources(Xseg)
        if algorithm.upper() == "FRIDA":
            ang = float(np.argmax(np.abs(doa._gen_dirty_img())))
        else:
            ang = float(np.argmax(doa.grid.values))
        angs.append(ang)
    return np.asarray(angs, dtype=np.float32)


def convolve_ir(ir: np.ndarray, noise: np.ndarray) -> np.ndarray:
    ys = [np.convolve(noise, ir[ch], mode="full") for ch in range(ir.shape[0])]
    return np.stack(ys, axis=0).astype(np.float32)


def compute_doa_sequences(ir: np.ndarray, rx_pos: np.ndarray, tx_pos: np.ndarray, cfg: dict):
    rx_center = rx_pos.mean(axis=0)
    rx_center_xy = rx_center[:2]
    tx_xy = tx_pos[:2]
    true_deg = true_angle_deg(tx_xy, rx_center_xy)

    noise = white_noise(cfg["noise_seconds"], cfg["fs"], cfg["random_seed"])
    y = convolve_ir(ir, noise)
    X = stft_multi(y, int(cfg["n_fft"]), int(cfg["hop_size"]), str(cfg["window"]))

    mic = rx_pos[:, :2].T
    angs = doa_sliding(
        X,
        mic,
        cfg["doa_algorithm"],
        int(cfg["fs"]),
        int(cfg["n_fft"]),
        int(cfg["split_time_frame"]),
    )

    if len(angs) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    true_series = np.full_like(angs, fill_value=true_deg, dtype=np.float32)
    return angs, true_series


def main():
    parser = argparse.ArgumentParser(description="Compute whitenoise DoA metrics")
    parser.add_argument("--config", required=True, help="Path to whitenoise_metrics_config.yml")
    parser.add_argument("--input", required=True, help="Path to results.npz")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    position_tx, position_rx, ir_pred, ir_gt = load_results(args.input)

    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy2(args.config, os.path.join(args.output_dir, "whitenoise_metrics_config.yml"))

    doa_pred_list = []
    doa_true_list = []
    doa_gt_list = [] if ir_gt is not None else None

    for i in range(ir_pred.shape[0]):
        pred_deg, true_deg = compute_doa_sequences(
            ir_pred[i], position_rx[i], position_tx[i], cfg
        )
        doa_pred_list.append(pred_deg)
        doa_true_list.append(true_deg)

        if ir_gt is not None:
            gt_deg, _ = compute_doa_sequences(
                ir_gt[i], position_rx[i], position_tx[i], cfg
            )
            doa_gt_list.append(gt_deg)

    max_len = max((len(x) for x in doa_pred_list), default=0)

    def pad_to(arr_list):
        out = np.full((len(arr_list), max_len), np.nan, dtype=np.float32)
        for i, arr in enumerate(arr_list):
            if len(arr) == 0:
                continue
            out[i, : len(arr)] = arr
        return out

    doa_pred_deg = pad_to(doa_pred_list)
    doa_true_deg = pad_to(doa_true_list)

    out = {
        "position_tx": position_tx,
        "position_rx": position_rx,
        "ir_pred": ir_pred,
        "doa_true_deg": doa_true_deg,
        "doa_pred_deg": doa_pred_deg,
    }

    if ir_gt is not None:
        out["ir_gt"] = ir_gt
        out["doa_gt_deg"] = pad_to(doa_gt_list)

    np.savez(os.path.join(args.output_dir, "whitenoise_metrics_results.npz"), **out)


if __name__ == "__main__":
    main()
