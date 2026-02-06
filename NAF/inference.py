#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import yaml
import librosa

from model.modules import embedding_module_log
from model.networks import kernel_residual_fc_embeds


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_spectrograms(input_stft, input_if, log_eps):
    padded_input_stft = np.concatenate((input_stft, input_stft[:, -1:]), axis=1)
    padded_input_if = np.concatenate((input_if, input_if[:, -1:]), axis=1)
    unwrapped = np.cumsum(padded_input_if, axis=-1) * np.pi
    phase_val = np.cos(unwrapped) + 1j * np.sin(unwrapped)
    return (np.exp(padded_input_stft) - log_eps) * phase_val


def istft_ir(spec: np.ndarray, nfft=512, hop=128, win="hann") -> np.ndarray:
    y = librosa.istft(
        spec,
        n_fft=nfft,
        hop_length=hop,
        win_length=nfft,
        window=win,
        center=True,
    )
    return y.astype(np.float32)


def build_model(cfg, min_xy, max_xy, device):
    xyz_embedder = embedding_module_log(num_freqs=cfg["embed_xyz_num_freqs"], ch_dim=2, max_freq=cfg["embed_xyz_max_freq"]).to(device)
    time_embedder = embedding_module_log(num_freqs=cfg["embed_time_num_freqs"], ch_dim=2, max_freq=cfg["embed_time_max_freq"]).to(device)
    freq_embedder = embedding_module_log(num_freqs=cfg["embed_freq_num_freqs"], ch_dim=2, max_freq=cfg["embed_freq_max_freq"]).to(device)

    input_ch = (
        2 * (2 * cfg["embed_xyz_num_freqs"] + 1)
        + 2 * (2 * cfg["embed_time_num_freqs"] + 1)
        + 2 * (2 * cfg["embed_freq_num_freqs"] + 1)
    )

    net = kernel_residual_fc_embeds(
        input_ch=input_ch,
        dir_ch=cfg["dir_ch"],
        output_ch=2,
        intermediate_ch=cfg["features"],
        grid_ch=cfg["grid_features"],
        num_block=cfg["layers"],
        num_block_residual=cfg["layers_residual"],
        grid_gap=cfg["grid_gap"],
        grid_bandwidth=cfg["bandwith_init"],
        bandwidth_min=cfg["min_bandwidth"],
        bandwidth_max=cfg["max_bandwidth"],
        float_amt=cfg["position_float"],
        min_xy=min_xy,
        max_xy=max_xy,
        batch_norm="none",
        batch_norm_features=cfg["pixel_count"],
        activation_func_name=cfg["activation_func_name"],
    ).to(device)

    return net, xyz_embedder, time_embedder, freq_embedder


def normalize_xy(xy: np.ndarray, min_xy: np.ndarray, max_xy: np.ndarray) -> np.ndarray:
    return np.clip(((xy - min_xy) / (max_xy - min_xy) - 0.5) * 2.0, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="NAF inference")
    parser.add_argument("--config", required=True, help="Path to inference_config.yml")
    parser.add_argument("--chkpt", required=True, help="Checkpoint path")
    parser.add_argument("--speaker", required=True, help="Speaker positions json")
    parser.add_argument("--receiver", required=True, help="Receiver positions json")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    full_cfg = load_yaml(Path(args.config))
    cfg = full_cfg.get("model_param", {})
    model_type = cfg.get("model_type", "NAF+")
    dir_ch = 1 if model_type == "NAF" else int(cfg.get("dir_ch", 1))
    cfg["dir_ch"] = dir_ch

    with open(args.speaker, "r", encoding="utf-8") as f:
        speaker = json.load(f)
    with open(args.receiver, "r", encoding="utf-8") as f:
        receiver = json.load(f)

    tx_positions = np.asarray(speaker["positions"], dtype=np.float32)
    rx_positions = np.asarray(receiver["positions"], dtype=np.float32)

    if rx_positions.ndim != 3:
        raise ValueError("receiver positions must be (N_rx, N_ch, 2/3)")

    rx_original = rx_positions
    if model_type == "NAF":
        # NAF runs per-channel without center reduction
        pass
    else:
        # NAF+ uses rx_center (keep full array for output)
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PyTorch 2.6 defaults to weights_only=True; use full load for trusted checkpoints.
    chkpt = torch.load(args.chkpt, map_location=device, weights_only=False)
    mean = chkpt.get("mean")
    std = chkpt.get("std")
    phase_std = float(chkpt.get("phase_std"))
    sound_size = chkpt.get("sound_size")
    max_len = int(chkpt.get("max_len"))
    n_fft = int(chkpt.get("n_fft", 512))
    hop_size = int(chkpt.get("hop_size", 128))
    window = chkpt.get("window", "hann")
    log_eps = float(chkpt.get("log_eps", 1e-3))

    if sound_size is None or mean is None or std is None:
        raise ValueError("Checkpoint missing normalization stats or sound_size")

    mean = np.asarray(mean)
    std = np.asarray(std)

    if "xy_min" in cfg and "xy_max" in cfg:
        min_xy = np.asarray(cfg.get("xy_min"), dtype=np.float32)
        max_xy = np.asarray(cfg.get("xy_max"), dtype=np.float32)
    else:
        all_xy = np.concatenate([tx_positions[:, :2], rx_positions.reshape(-1, rx_positions.shape[-1])[:, :2]], axis=0)
        min_xy = all_xy.min(axis=0)
        max_xy = all_xy.max(axis=0)

    net, xyz_embedder, time_embedder, freq_embedder = build_model(cfg, min_xy, max_xy, device)
    net.load_state_dict(chkpt["network"])
    net.eval()

    F = sound_size[1]
    T = sound_size[2]
    freq_idx = np.arange(0, F)
    time_idx = np.arange(0, T)
    grid_t, grid_f = np.meshgrid(time_idx, freq_idx)
    grid_t = grid_t.reshape(-1)
    grid_f = grid_f.reshape(-1)

    freq_norm = 2.0 * torch.from_numpy(grid_f).float() / 255.0 - 1.0
    time_norm = 2.0 * torch.from_numpy(grid_t).float() / float(max_len - 1) - 1.0
    freq_norm = freq_norm[None, :].to(device)
    time_norm = time_norm[None, :].to(device)

    results_tx = []
    results_rx = []
    results_ir = []

    with torch.no_grad():
        for tx in tx_positions:
            for rx in rx_positions:
                if model_type == "NAF":
                    # rx is (N_ch, D); predict each channel and stack
                    ir_ch_list = []
                    for ch in range(rx.shape[0]):
                        rx_point = rx[ch]
                        rx_xy = rx_point[:2]
                        tx_xy = tx[:2]
                        start_pos = torch.from_numpy(normalize_xy(tx_xy, min_xy, max_xy))[None]
                        end_pos = torch.from_numpy(normalize_xy(rx_xy, min_xy, max_xy))[None]
                        total_position = torch.cat((start_pos, end_pos), dim=1).float().to(device)
                        total_non_norm = torch.cat(
                            (torch.from_numpy(tx_xy)[None], torch.from_numpy(rx_xy)[None]), dim=1
                        ).float().to(device)

                        position_embed = xyz_embedder(total_position).expand(1, grid_f.shape[0], -1)
                        freq_embed = freq_embedder(freq_norm.unsqueeze(2) * 2.0 * math.pi)
                        time_embed = time_embedder(time_norm.unsqueeze(2) * 2.0 * math.pi)
                        total_in = torch.cat((position_embed, freq_embed, time_embed), dim=2)

                        out_list = []
                        pixel_count = int(cfg.get("pixel_count", 2000))
                        for split_id in range(-(-total_in.shape[1] // pixel_count)):
                            chunk = total_in[:, split_id * pixel_count : (split_id + 1) * pixel_count]
                            if chunk.shape[1] < pixel_count:
                                pad = torch.zeros(chunk.shape[0], pixel_count - chunk.shape[1], chunk.shape[2], device=device)
                                chunk = torch.cat((chunk, pad), dim=1)
                                out_chunk = net(chunk, total_non_norm).transpose(1, 2)
                                out_chunk = out_chunk[:, :, : total_in.shape[1] - split_id * pixel_count, :]
                            else:
                                out_chunk = net(chunk, total_non_norm).transpose(1, 2)
                            out_list.append(out_chunk)

                        output = torch.cat(out_list, dim=2)
                        myout = output.cpu().numpy()
                        myout_mag = myout[..., 0].reshape(1, dir_ch, F, T)
                        myout_phase = myout[..., 1].reshape(1, dir_ch, F, T)

                        net_mag = (myout_mag * std + mean)[0]
                        net_phase = myout_phase[0] * phase_std
                        net_spec = get_spectrograms(net_mag, net_phase, log_eps)
                        ir_pred = istft_ir(net_spec, nfft=n_fft, hop=hop_size, win=window)
                        ir_ch_list.append(ir_pred.astype(np.float32))

                    results_tx.append(tx.astype(np.float32))
                    results_rx.append(rx.astype(np.float32))
                    results_ir.append(np.stack(ir_ch_list, axis=0))
                else:
                    rx_center = rx.mean(axis=0)
                    rx_xy = rx_center[:2]
                    tx_xy = tx[:2]
                    start_pos = torch.from_numpy(normalize_xy(tx_xy, min_xy, max_xy))[None]
                    end_pos = torch.from_numpy(normalize_xy(rx_xy, min_xy, max_xy))[None]
                    total_position = torch.cat((start_pos, end_pos), dim=1).float().to(device)
                    total_non_norm = torch.cat(
                        (torch.from_numpy(tx_xy)[None], torch.from_numpy(rx_xy)[None]), dim=1
                    ).float().to(device)

                    position_embed = xyz_embedder(total_position).expand(1, grid_f.shape[0], -1)
                    freq_embed = freq_embedder(freq_norm.unsqueeze(2) * 2.0 * math.pi)
                    time_embed = time_embedder(time_norm.unsqueeze(2) * 2.0 * math.pi)
                    total_in = torch.cat((position_embed, freq_embed, time_embed), dim=2)

                    out_list = []
                    pixel_count = int(cfg.get("pixel_count", 2000))
                    for split_id in range(-(-total_in.shape[1] // pixel_count)):
                        chunk = total_in[:, split_id * pixel_count : (split_id + 1) * pixel_count]
                        if chunk.shape[1] < pixel_count:
                            pad = torch.zeros(chunk.shape[0], pixel_count - chunk.shape[1], chunk.shape[2], device=device)
                            chunk = torch.cat((chunk, pad), dim=1)
                            out_chunk = net(chunk, total_non_norm).transpose(1, 2)
                            out_chunk = out_chunk[:, :, : total_in.shape[1] - split_id * pixel_count, :]
                        else:
                            out_chunk = net(chunk, total_non_norm).transpose(1, 2)
                        out_list.append(out_chunk)

                    output = torch.cat(out_list, dim=2)
                    myout = output.cpu().numpy()
                    myout_mag = myout[..., 0].reshape(1, dir_ch, F, T)
                    myout_phase = myout[..., 1].reshape(1, dir_ch, F, T)

                    net_mag = (myout_mag * std + mean)[0]
                    net_phase = myout_phase[0] * phase_std
                    net_spec = get_spectrograms(net_mag, net_phase, log_eps)
                    ir_pred = istft_ir(net_spec, nfft=n_fft, hop=hop_size, win=window)

                    results_tx.append(tx.astype(np.float32))
                    results_rx.append(rx.astype(np.float32))
                    results_ir.append(ir_pred.astype(np.float32))

    results_tx = np.stack(results_tx, axis=0)
    results_rx = np.stack(results_rx, axis=0)
    results_ir = np.stack(results_ir, axis=0)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "inference_config.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(full_cfg, f, allow_unicode=True, sort_keys=False)

    np.savez(
        os.path.join(args.output_dir, "results.npz"),
        position_tx=results_tx,
        position_rx=results_rx,
        ir_pred=results_ir,
    )


if __name__ == "__main__":
    main()
