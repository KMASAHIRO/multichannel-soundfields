import argparse
import json
import os
import shutil

import numpy as np
import pyroomacoustics as pra
from tqdm import tqdm
import yaml


def load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_speaker_positions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    positions = data["positions"]
    return np.array(positions, dtype=np.float32)


def load_receiver_positions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    positions = data["positions"]
    positions = np.array(positions, dtype=np.float32)
    return positions


def build_ir(room, mic_index, ir_len):
    ir = room.rir[mic_index][0]
    if len(ir) >= ir_len:
        return np.array(ir[:ir_len], dtype=np.float32)
    padded = np.zeros(ir_len, dtype=np.float32)
    padded[: len(ir)] = ir
    return padded


def simulate_pyroomacoustics_ir(config_path, speaker_path, receiver_path, output_dir):
    config = load_yaml_config(config_path)
    room_cfg = config.get("room", {})
    signal_cfg = config.get("signal", {})

    room_dim = room_cfg.get("room_dim", [6.110, 8.807, 2.7])
    max_order = room_cfg.get("max_order", 10)
    e_absorption = room_cfg.get("e_absorption", 0.0055)
    sampling_rate = signal_cfg.get("sampling_rate", 16000)
    ir_len = signal_cfg.get("ir_len", 1600)

    tx_positions = load_speaker_positions(speaker_path)
    rx_positions = load_receiver_positions(receiver_path)
    num_channels = rx_positions.shape[1]
    rx_centers = rx_positions.mean(axis=1)

    os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(config_path, os.path.join(output_dir, "config.yml"))
    shutil.copy2(speaker_path, os.path.join(output_dir, "speaker_data.json"))
    shutil.copy2(receiver_path, os.path.join(output_dir, "receiver_data.json"))

    for tx_index, tx_pos in tqdm(
        enumerate(tx_positions),
        total=len(tx_positions),
        desc="Pyroom IR Simulation",
    ):
        tx_output_path = os.path.join(output_dir, f"tx_{tx_index}")
        os.makedirs(tx_output_path, exist_ok=True)

        valid_rx_indices = [
            i for i, center in enumerate(rx_centers) if not np.allclose(center, tx_pos)
        ]
        rx_pos_valid = rx_positions[valid_rx_indices]
        mic_positions = rx_pos_valid.reshape(-1, 3).T  # shape: (3, N_rx*ch_num)

        room = pra.ShoeBox(
            room_dim,
            fs=sampling_rate,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )
        room.add_source(tx_pos.tolist())
        room.add_microphone_array(mic_positions)
        room.compute_rir()

        for rx_out_idx, _rx_idx in enumerate(valid_rx_indices):
            ir_channels = []
            for ch in range(num_channels):
                mic_idx = rx_out_idx * num_channels + ch
                ir_channels.append(build_ir(room, mic_idx, ir_len))
            ir = np.stack(ir_channels, axis=0)

            out_path = os.path.join(tx_output_path, f"rx_{rx_out_idx}.npz")
            np.savez(
                out_path,
                ir=ir,
                position_rx=rx_pos_valid[rx_out_idx],
                position_tx=tx_pos,
            )


def main():
    parser = argparse.ArgumentParser(description="Pyroomacoustics simulation")
    parser.add_argument("--config", required=True, help="Path to config.yml")
    parser.add_argument("--speaker", required=True, help="Path to speaker_data.json")
    parser.add_argument("--receiver", required=True, help="Path to receiver_data.json")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    simulate_pyroomacoustics_ir(
        config_path=args.config,
        speaker_path=args.speaker,
        receiver_path=args.receiver,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
