from typing import Optional, Tuple

import numpy as np
import pyroomacoustics as pra
from scipy.signal import get_window


def _stft(
    signal: np.ndarray,
    n_fft: int,
    hop: Optional[int] = None,
    window: str = "hann",
) -> np.ndarray:
    if hop is None:
        hop = n_fft // 2
    win = get_window(window, n_fft, fftbins=True)
    pad = max(0, n_fft - signal.shape[-1])
    if pad > 0:
        signal = np.pad(signal, (0, pad))
    frames = 1 + (signal.shape[-1] - n_fft) // hop if signal.shape[-1] >= n_fft else 1
    stft = np.zeros((n_fft // 2 + 1, frames), dtype=np.complex64)
    for i in range(frames):
        s = i * hop
        e = s + n_fft
        frame = signal[s:e]
        if frame.shape[0] < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.shape[0]))
        spectrum = np.fft.rfft(frame * win, n=n_fft)
        stft[:, i] = spectrum
    return stft


def _resolve_algorithm(name: str):
    if name in pra.doa.algorithms:
        return pra.doa.algorithms[name]
    lower_map = {k.lower(): v for k, v in pra.doa.algorithms.items()}
    key = name.lower()
    if key in lower_map:
        return lower_map[key]
    raise ValueError(f"Unknown DoA algorithm: {name}")


def _estimate_doa_deg(
    ir: np.ndarray,
    rx_pos: np.ndarray,
    algorithm: str,
    n_fft: int,
    fs: int,
    hop_size: Optional[int] = None,
    window: str = "hann",
    speed: float = 343.8,
) -> float:
    dim = rx_pos.shape[1]
    mic = rx_pos.T  # (dim, M)
    stfts = [_stft(ch, n_fft, hop=hop_size, window=window) for ch in ir]
    X = np.stack(stfts, axis=0)  # (M, F, T)
    if not np.isfinite(X).all():
        return float("nan")

    algo_cls = _resolve_algorithm(algorithm)
    doa = algo_cls(mic, fs=fs, nfft=n_fft, c=speed, num_src=1)
    try:
        doa.locate_sources(X)
    except np.linalg.LinAlgError:
        return float("nan")

    if algorithm.upper() == "FRIDA":
        deg = float(np.argmax(np.abs(doa._gen_dirty_img())))
    else:
        deg = float(np.argmax(doa.grid.values))
    return deg


def _ensure_shapes(
    ir: np.ndarray,
    rx_positions: np.ndarray,
    tx_positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rx_positions = np.asarray(rx_positions, dtype=np.float32)
    tx_positions = np.asarray(tx_positions, dtype=np.float32)

    if rx_positions.ndim == 2:
        rx_positions = rx_positions[None, ...]
    if tx_positions.ndim == 1:
        tx_positions = tx_positions[None, ...]

    if ir.ndim == 2:
        ir = ir[None, ...]

    if rx_positions.ndim != 3:
        raise ValueError("rx_positions must be (N, N_ch, 2/3)")
    if tx_positions.ndim != 2:
        raise ValueError("tx_positions must be (N, 2/3)")
    if ir.ndim != 3:
        raise ValueError("ir must be (N, N_ch, T)")

    if rx_positions.shape[0] != tx_positions.shape[0]:
        raise ValueError("rx_positions and tx_positions must have same N")
    if ir.shape[0] != rx_positions.shape[0]:
        raise ValueError("ir N must match rx_positions N")
    if ir.shape[1] != rx_positions.shape[1]:
        raise ValueError("ir N_ch must match rx_positions N_ch")

    return ir, rx_positions, tx_positions


def _true_doa_deg(rx_centers: np.ndarray, tx_positions: np.ndarray) -> np.ndarray:
    vec = tx_positions[:, :2] - rx_centers[:, :2]
    ang = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))
    return ang.astype(np.float32)


def doa_metrics(
    ir_gt: Optional[np.ndarray],
    ir_pred: np.ndarray,
    algorithm: str,
    n_fft: int,
    fs: int,
    rx_positions: np.ndarray,
    tx_positions: np.ndarray,
    hop_size: Optional[int] = None,
    window: str = "hann",
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Estimate DoA angles from ir_gt/ir_pred and return true directions.

    Returns
    -------
    doa_gt_deg : np.ndarray or None
    doa_pred_deg : np.ndarray
    doa_true_deg : np.ndarray
    """
    ir_pred = np.asarray(ir_pred)
    ir_pred, rx_positions, tx_positions = _ensure_shapes(ir_pred, rx_positions, tx_positions)

    rx_centers = rx_positions.mean(axis=1)
    doa_true_deg = _true_doa_deg(rx_centers, tx_positions)

    doa_pred = []
    for n in range(ir_pred.shape[0]):
        doa_pred.append(
            _estimate_doa_deg(
                ir_pred[n],
                rx_positions[n],
                algorithm,
                n_fft,
                fs,
                hop_size=hop_size,
                window=window,
            )
        )
    doa_pred_deg = np.array(doa_pred, dtype=np.float32)

    doa_gt_deg = None
    if ir_gt is not None:
        ir_gt = np.asarray(ir_gt)
        ir_gt, _, _ = _ensure_shapes(ir_gt, rx_positions, tx_positions)
        doa_gt = []
        for n in range(ir_gt.shape[0]):
            doa_gt.append(
                _estimate_doa_deg(
                    ir_gt[n],
                    rx_positions[n],
                    algorithm,
                    n_fft,
                    fs,
                    hop_size=hop_size,
                    window=window,
                )
            )
        doa_gt_deg = np.array(doa_gt, dtype=np.float32)

    return doa_gt_deg, doa_pred_deg, doa_true_deg
