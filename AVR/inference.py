import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml

from model import AVRModel
from renderer import AVRRender


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_model_and_renderer(cfg: dict) -> Tuple[AVRModel, AVRRender]:
    setting = cfg["setting"]
    param = cfg["param"]
    model_cfg = dict(cfg["model"])

    model_type = setting["model_type"]
    channel_embed = dict(model_cfg.get("channel_embed", {}))
    if model_type == "AVR":
        channel_embed["is_embed"] = False
    else:
        channel_embed["is_embed"] = True
        channel_embed["ch_num"] = setting["dir_ch"]
    model_cfg["channel_embed"] = channel_embed

    model = AVRModel(model_cfg)
    render = AVRRender(
        networks_fn=model,
        n_samples=param["n_samples"],
        near=param["near"],
        far=param["far"],
        n_azi=param["n_azi"],
        n_ele=param["n_ele"],
        speed=setting["speed"],
        fs=setting["fs"],
        pathloss=param["pathloss"],
        xyz_min=setting["xyz_min"],
        xyz_max=setting["xyz_max"],
    )
    return model, render


def load_positions(path: Path, key: str) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if key not in data:
        raise KeyError(f"Missing '{key}' in {path}")
    return np.asarray(data[key], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="AVR inference")
    parser.add_argument("--config", required=True, help="Path to inference_config.yml")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint tar")
    parser.add_argument("--speaker", required=True, help="Speaker positions JSON")
    parser.add_argument("--receiver", required=True, help="Receiver positions JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    setting = cfg["setting"]
    model_type = setting["model_type"]
    dir_ch = setting["dir_ch"]
    seq_len = cfg["model"]["signal_output_dim"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "inference_config.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, renderer = build_model_and_renderer(cfg)
    renderer = renderer.to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    renderer.load_state_dict(ckpt["model_state_dict"])
    renderer.eval()

    tx_positions = load_positions(Path(args.speaker), "positions")
    rx_positions = load_positions(Path(args.receiver), "positions")

    if rx_positions.ndim != 3:
        raise ValueError("receiver positions must be (N_rx, N_ch, 3)")
    if tx_positions.ndim != 2:
        raise ValueError("speaker positions must be (N_tx, 3)")
    if rx_positions.shape[1] != dir_ch:
        raise ValueError(f"Expected N_ch={dir_ch}, got {rx_positions.shape[1]}")

    position_tx_list = []
    position_rx_list = []
    ir_pred_list = []

    with torch.no_grad():
        for tx in tx_positions:
            for rx in rx_positions:
                if model_type == "AVR":
                    rays_o = torch.tensor(rx, dtype=torch.float32, device=device)
                    ch_idx = None
                else:
                    rx_center = rx.mean(axis=0)
                    rays_o = torch.tensor(
                        np.repeat(rx_center[None, :], dir_ch, axis=0),
                        dtype=torch.float32,
                        device=device,
                    )
                    ch_idx = torch.arange(dir_ch, device=device, dtype=torch.long)

                position_tx_batch = torch.tensor(
                    np.repeat(tx[None, :], dir_ch, axis=0),
                    dtype=torch.float32,
                    device=device,
                )

                pred_sig = renderer(rays_o, position_tx_batch, ch_idx=ch_idx)
                pred_sig = pred_sig[..., 0] + 1j * pred_sig[..., 1]
                ir_pred = torch.fft.irfft(pred_sig, n=seq_len, dim=-1).cpu().numpy()

                position_tx_list.append(tx.astype(np.float32))
                position_rx_list.append(rx.astype(np.float32))
                ir_pred_list.append(ir_pred.astype(np.float32))

    results = {
        "position_tx": np.stack(position_tx_list, axis=0),
        "position_rx": np.stack(position_rx_list, axis=0),
        "ir_pred": np.stack(ir_pred_list, axis=0),
    }
    np.savez_compressed(output_dir / "results.npz", **results)


if __name__ == "__main__":
    main()
