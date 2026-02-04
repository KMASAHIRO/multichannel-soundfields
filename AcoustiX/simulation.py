import argparse
import json
import os
import sys
from shutil import copyfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from simu_utils import ir_simulation, load_cfg

tf.get_logger().setLevel("ERROR")
tf.random.set_seed(1)


def load_speaker_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    positions = np.array(data["positions"], dtype=np.float32)
    orientations = np.array(
        data.get("orientations", [[1.0, 0.0, 0.0]] * len(positions)),
        dtype=np.float32,
    )
    patterns = data.get("patterns", ["uniform"] * len(positions))
    return positions, orientations, patterns


def load_receiver_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    positions = np.array(data["positions"], dtype=np.float32)
    if "orientations" in data:
        orientations = np.array(data["orientations"], dtype=np.float32)
    else:
        orientations = np.tile(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            (positions.shape[0], positions.shape[1], 1),
        )
    patterns = data.get("patterns", None)
    if patterns is None:
        patterns = [["uniform"] * positions.shape[1] for _ in range(positions.shape[0])]
    return positions, orientations, patterns


def main():
    parser = argparse.ArgumentParser(description="AcoustiX simulation")
    parser.add_argument("--config", required=True, help="Path to config.yml")
    parser.add_argument("--scene", required=True, help="Path to scene xml")
    parser.add_argument("--speaker", required=True, help="Path to speaker_data.json")
    parser.add_argument("--receiver", required=True, help="Path to receiver_data.json")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    tx_positions, tx_orientations, tx_patterns = load_speaker_data(args.speaker)
    rx_positions, rx_orientations, rx_patterns = load_receiver_data(args.receiver)

    rx_centers = rx_positions.mean(axis=1)
    num_channels = rx_positions.shape[1]

    os.makedirs(args.output_dir, exist_ok=True)
    copyfile(args.config, os.path.join(args.output_dir, "config.yml"))
    copyfile(args.speaker, os.path.join(args.output_dir, "speaker_data.json"))
    copyfile(args.receiver, os.path.join(args.output_dir, "receiver_data.json"))

    simu_config = load_cfg(config_file=args.config)

    for tx_index, tx_pos in tqdm(
        enumerate(tx_positions),
        total=len(tx_positions),
        desc="Simulating IR",
    ):
        tx_output_path = os.path.join(args.output_dir, f"tx_{tx_index}")
        os.makedirs(tx_output_path, exist_ok=True)

        tx_ori = tx_orientations[tx_index]
        tx_ori = tx_ori / np.linalg.norm(tx_ori)
        tx_pattern = tx_patterns[tx_index]

        valid_rx_indices = [
            i for i, center in enumerate(rx_centers) if not np.allclose(center, tx_pos)
        ]
        rx_pos_valid = rx_positions[valid_rx_indices]
        rx_ori_valid = rx_orientations[valid_rx_indices]
        rx_patterns_valid = [rx_patterns[i] for i in valid_rx_indices]

        rx_pos_flat = rx_pos_valid.reshape(-1, 3)
        rx_ori_flat = rx_ori_valid.reshape(-1, 3)
        rx_pattern_flat = [p for row in rx_patterns_valid for p in row]

        ir_time_all, rx_pos_out, rx_ori_out = ir_simulation(
            scene_path=args.scene,
            rx_pos=rx_pos_flat,
            tx_pos=tx_pos,
            rx_ori=rx_ori_flat,
            tx_ori=tx_ori,
            simu_config=simu_config,
            rx_pattern_types=rx_pattern_flat,
            tx_pattern_type=tx_pattern,
        )

        num_rx = rx_pos_valid.shape[0]
        ir_reshaped = ir_time_all.reshape(num_rx, num_channels, -1)
        rx_pos_reshaped = rx_pos_out.reshape(num_rx, num_channels, 3)
        rx_ori_reshaped = rx_ori_out.reshape(num_rx, num_channels, 3)

        for rx_index in range(num_rx):
            out_path = os.path.join(tx_output_path, f"rx_{rx_index}.npz")
            np.savez(
                out_path,
                ir=ir_reshaped[rx_index],
                position_rx=rx_pos_reshaped[rx_index],
                position_tx=tx_pos,
                orientation_rx=rx_ori_reshaped[rx_index],
                orientation_tx=tx_ori,
            )


if __name__ == "__main__":
    main()
