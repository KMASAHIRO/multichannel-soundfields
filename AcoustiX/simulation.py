import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(1)

import json
import numpy as np
from tqdm import tqdm
from simu_utils import ir_simulation, save_ir, load_cfg
from shutil import copyfile

def generate_positions_real_env():
    # 部屋中心が原点
    x_offset = -6.110 / 2 + 1.0  # X=1.0m から開始
    y_offset = -8.807 / 2 + 1.5  # Y=1.5m から開始
    z_pos = -2.7 / 2 + 1.5       # 床から1.5m上 = 0.15

    # グリッド全体の24箇所をリストアップ
    all_centers = []
    for i in range(6):  # Y方向（奥行）
        for j in range(4):  # X方向（幅）
            x = x_offset + j * 1.0
            y = y_offset + i * 1.0
            all_centers.append([x, y, z_pos])
    all_centers = np.array(all_centers)  # shape: (24, 3)

    # スピーカー配置：四隅 + 中央4点のインデックス
    spk_indices = [0*4+0, 0*4+3, 5*4+0, 5*4+3, 2*4+1, 2*4+2, 3*4+1, 3*4+2]
    tx_pos = all_centers[spk_indices]  # shape: (8, 3)

    # 定数定義
    num_channels = 8
    radius = 0.0365  # マイク円半径
    num_speakers = len(spk_indices)
    num_total = len(all_centers)
    num_mics_per_spk = num_total - 1  # 1つをスピーカーに使うので残りがマイク

    # 出力配列初期化
    rx_pos = np.zeros((num_speakers, num_mics_per_spk, num_channels, 3))
    mic_centers_per_spk = []

    for s_idx, spk_idx in enumerate(spk_indices):
        # スピーカー以外の23個のマイク中心位置を取得
        mic_indices = [i for i in range(num_total) if i != spk_idx]
        mic_centers = all_centers[mic_indices]  # shape: (23, 3)
        mic_centers_per_spk.append(mic_centers)

        for m_idx, (cx, cy, cz) in enumerate(mic_centers):
            for ch in range(num_channels):
                # Y+方向を1ch目として反時計回りに配置
                theta = np.pi / 2 + ch * (2 * np.pi / num_channels)
                x = cx + radius * np.cos(theta)
                y = cy + radius * np.sin(theta)
                rx_pos[s_idx, m_idx, ch] = [x, y, cz]

    return tx_pos, mic_centers_per_spk, rx_pos

if __name__ == '__main__':
    config_file = "./simu_config/basic_config.yml"
    dataset_name = "real_env_Smooth_concrete_painted_16kHz"
    scene_path = "./custom_scene/real_env_Smooth_concrete_painted/real_env_Smooth_concrete_painted.xml"

    # 送受信機の配置生成
    tx_all, mic_centers_all, rx_all = generate_positions_real_env()

    # 出力ディレクトリ作成・設定ファイルコピー
    scene_folder = os.path.dirname(os.path.abspath(scene_path))
    output_path = os.path.join(scene_folder, dataset_name)
    os.makedirs(output_path, exist_ok=True)
    copyfile(config_file, os.path.join(output_path, "config.yml"))

    # ✅ 送信機データをまとめて保存（ループ外）
    speaker_data = {
        "speaker": {
            "positions": tx_all.tolist(),
            "orientations": [[1.0, 0.0, 0.0]] * len(tx_all)  # X+方向
        }
    }
    with open(os.path.join(output_path, 'speaker_data.json'), 'w') as json_file:
        json.dump(speaker_data, json_file, indent=4)

    # シミュレーション設定読み込み
    simu_config = load_cfg(config_file=config_file)

    for tx_index, tx_pos in tqdm(enumerate(tx_all), total=len(tx_all), desc="Simulating IR"):
        tx_output_path = os.path.join(output_path, f"tx_{tx_index}")
        os.makedirs(tx_output_path, exist_ok=True)

        # 送信機の向き（固定）
        tx_ori = np.array([1.0, 0.0, 0.0])
        tx_ori = tx_ori / np.linalg.norm(tx_ori)

        # 受信機の位置・向き
        rx_pos = rx_all[tx_index].reshape(-1, 3)  # (23×8, 3)
        rx_ori = np.tile(np.array([[1.0, 0.0, 0.0]]), (rx_pos.shape[0], 1))
        rx_ori /= np.linalg.norm(rx_ori, axis=1, keepdims=True)

        # IRシミュレーション
        ir_time_all, rx_pos_out, rx_ori_out = ir_simulation(
            scene_path=scene_path,
            rx_pos=rx_pos,
            tx_pos=tx_pos,
            rx_ori=rx_ori,
            tx_ori=tx_ori,
            simu_config=simu_config
        )

        # IR保存
        num_speakers, num_mics, num_channels, _ = rx_all.shape

        rx_pos_reshaped = rx_pos.reshape(num_mics, num_channels, 3)
        rx_ori_reshaped = rx_ori.reshape(num_mics, num_channels, 3)
        ir_reshaped = ir_time_all.reshape(num_mics, num_channels, -1)

        for mic_index in range(num_mics):
            rx_dir = os.path.join(tx_output_path, f"rx_{mic_index}")
            os.makedirs(rx_dir, exist_ok=True)

            save_ir(
                ir_samples=ir_reshaped[mic_index],
                rx_pos=rx_pos_reshaped[mic_index],
                rx_ori=rx_ori_reshaped[mic_index],
                tx_pos=tx_pos,
                tx_ori=tx_ori,
                save_path=rx_dir,
                prefix=0
            )

