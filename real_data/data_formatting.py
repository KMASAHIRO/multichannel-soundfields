import os
import numpy as np
import soundfile as sf
from pathlib import Path

def generate_positions_real_env():
    x_offset = 1.0
    y_offset = 1.5
    z_pos = 1.5
    all_centers = []
    for i in range(6):  # Y方向（0〜5）
        for j in range(4):  # X方向（0〜3）
            x = x_offset + j * 1.0
            y = y_offset + i * 1.0
            all_centers.append([x, y, z_pos])
    all_centers = np.array(all_centers)

    spk_indices = [0, 3, 20, 23, 9, 10, 13, 14]  # all_centersのindex
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

    return tx_pos, mic_centers_per_spk, rx_pos, spk_indices

# 数式による index 変換
def wav_index_to_all_centers_index(wav_idx: int) -> int:
    y = (wav_idx - 1) % 6
    x = (wav_idx - 1) // 6
    return y * 4 + x

def all_centers_index_to_wav_index(center_idx: int) -> int:
    y = center_idx // 4
    x = center_idx % 4
    return x * 6 + y + 1

# メイン処理
def convert_ir_to_npz(ir_dir, output_dir, ir_start=9600, ir_len=1600):
    tx_pos, mic_centers_per_spk, rx_pos, spk_indices = generate_positions_real_env()
    ir_dir = Path(ir_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_speakers = len(spk_indices)
    num_channels = 8
    mic_per_spk = 23  # 24 - 1

    for s_idx, spk_idx in enumerate(spk_indices):
        tx_output_path = output_dir / f"tx_{s_idx}"
        tx_output_path.mkdir(parents=True, exist_ok=True)

        for mic_idx in range(mic_per_spk):
            rx_folder = tx_output_path / f"rx_{mic_idx}"
            rx_folder.mkdir(exist_ok=True)

            for ch in range(num_channels):
                rx_index = [i for i in range(24) if i != spk_idx][mic_idx]

                spk_file_id = all_centers_index_to_wav_index(spk_idx)
                rx_file_id = all_centers_index_to_wav_index(rx_index)
                wav_name = f"{spk_file_id:02d}_{rx_file_id:02d}_{ch+1}.wav"
                wav_path = ir_dir / wav_name

                if not wav_path.exists():
                    print(f"Missing: {wav_name}")
                    continue

                ir, sr = sf.read(wav_path)
                ir = ir[ir_start:ir_start+ir_len]

                np.savez(
                    rx_folder / f"ir_{str(ch).zfill(6)}.npz",
                    ir=np.array(ir),
                    position_rx=rx_pos[s_idx, mic_idx, ch],
                    position_tx=tx_pos[s_idx]
                )

    print(f"sampling_rate: {sr}")
    print(f"ir_len: {ir_len}")

if __name__ == "__main__":
    convert_ir_to_npz("../ir_peak_same", "./outputs/real_exp_8720", ir_start=8720, ir_len=1600)
