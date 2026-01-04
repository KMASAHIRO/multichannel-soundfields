import os
import numpy as np
import pyroomacoustics as pra
from tqdm import tqdm
import json

def generate_positions_real_env():
    # 部屋隅が原点
    x_offset = 1.0
    y_offset = 1.5
    z_pos = 1.5  # 床から1.5m

    all_centers = []
    for i in range(6):  # Y方向
        for j in range(4):  # X方向
            x = x_offset + j * 1.0
            y = y_offset + i * 1.0
            all_centers.append([x, y, z_pos])
    all_centers = np.array(all_centers)

    spk_indices = [0, 3, 20, 23, 9, 10, 13, 14]
    tx_pos = all_centers[spk_indices]

    num_channels = 8
    radius = 0.0365
    num_speakers = len(spk_indices)
    num_total = len(all_centers)
    num_mics_per_spk = num_total - 1

    rx_pos = np.zeros((num_speakers, num_mics_per_spk, num_channels, 3))
    mic_centers_per_spk = []

    for s_idx, spk_idx in enumerate(spk_indices):
        mic_indices = [i for i in range(num_total) if i != spk_idx]
        mic_centers = all_centers[mic_indices]
        mic_centers_per_spk.append(mic_centers)

        for m_idx, (cx, cy, cz) in enumerate(mic_centers):
            for ch in range(num_channels):
                theta = np.pi / 2 + ch * (2 * np.pi / num_channels)
                x = cx + radius * np.cos(theta)
                y = cy + radius * np.sin(theta)
                rx_pos[s_idx, m_idx, ch] = [x, y, cz]

    return tx_pos, mic_centers_per_spk, rx_pos

def simulate_pyroomacoustics_ir(
    output_path,
    room_dim=(6.110, 8.807, 2.7),
    sampling_rate=16000,
    max_order=10,
    e_absorption=0.0055,
    mic_num=8,
    ir_len=1600
):
    tx_all, mic_centers_all, rx_all = generate_positions_real_env()

    speaker_data = {
        "speaker": {
            "positions": tx_all.tolist()
        }
    }
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'speaker_data.json'), 'w') as f:
        json.dump(speaker_data, f, indent=4)

    for tx_index, tx_pos in tqdm(enumerate(tx_all), total=len(tx_all), desc="Pyroom IR Sim"):
        tx_output_path = os.path.join(output_path, f"tx_{tx_index}")
        os.makedirs(tx_output_path, exist_ok=True)

        rx_pos = rx_all[tx_index].reshape(-1, 3).T  # shape: (3, 23×8)
        room = pra.ShoeBox(
            room_dim,
            fs=sampling_rate,
            materials=pra.Material(e_absorption),
            max_order=max_order
        )
        room.add_source(tx_pos.tolist())
        room.add_microphone_array(rx_pos)
        room.compute_rir()

        # 各マイク中心ごとに保存（rx_0, rx_1, ..., rx_22）
        num_mics_per_spk = rx_all.shape[1]
        for mic_idx in range(num_mics_per_spk):
            rx_folder = os.path.join(tx_output_path, f"rx_{mic_idx}")
            os.makedirs(rx_folder, exist_ok=True)

            for ch in range(mic_num):
                idx = mic_idx * mic_num + ch
                ir = room.rir[idx][0][:ir_len]
                np.savez(
                    os.path.join(rx_folder, f'ir_{str(ch).zfill(6)}.npz'),
                    ir=np.array(ir),
                    position_rx=rx_pos[:, idx],
                    position_tx=np.array(tx_pos)
                )

if __name__ == "__main__":
    simulate_pyroomacoustics_ir(output_path="./outputs/real_env_avr_16kHz")
