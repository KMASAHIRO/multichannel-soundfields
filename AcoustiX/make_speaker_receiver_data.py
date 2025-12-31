import os
import json
import numpy as np


# ============================================================
# 0) パラメータ・変数
# ============================================================

# --- 出力先 ---
OUTPUT_DIR = "./simu_input"
SPEAKER_JSON_PATH  = os.path.join(OUTPUT_DIR, "speaker_data.json")
RECEIVER_JSON_PATH = os.path.join(OUTPUT_DIR, "receiver_data.json")

# --- マイクロフォンアレイ・スピーカーを配置するグリッドの設定 ---
X_OFFSET = 1.0   # X=1.0m から開始
Y_OFFSET = 1.5   # Y=1.5m から開始
Z_POS    = 1.5   # 床から1.5m上
NX = 4           # X方向（幅）の点数
NY = 6           # Y方向（奥行）の点数
GRID_STEP_X = 1.0
GRID_STEP_Y = 1.0

# --- スピーカー配置（グリッド上のインデックス） ---
SPK_INDICES = [0*4+0, 0*4+3, 5*4+0, 5*4+3, 2*4+1, 2*4+2, 3*4+1, 3*4+2]  # length=8

# --- マイクアレイ（8chの円形アレイ） ---
NUM_CHANNELS = 8
MIC_RADIUS = 0.0365  # [m]
# 1ch目をY+方向、反時計回り
THETA0 = np.pi / 2

# --- 向き（X+方向、指向性なし（uniform）の場合向きの影響はなし） ---
TX_ORIENTATION = np.array([1.0, 0.0, 0.0], dtype=float)
RX_ORIENTATION = np.array([1.0, 0.0, 0.0], dtype=float)

# --- 指向性パターン ---
TX_PATTERN_DEFAULT = "uniform"  # "heart" / "donut" / "uniform"
RX_PATTERN_DEFAULT = "uniform"  # "heart" / "donut" / "uniform"

# ============================================================
# 1) 位置生成
# ============================================================

# --- 全中心点 (NY*NX, 3) ---
all_centers = []
for i in range(NY):       # Y方向
    for j in range(NX):   # X方向
        x = X_OFFSET + j * GRID_STEP_X
        y = Y_OFFSET + i * GRID_STEP_Y
        all_centers.append([x, y, Z_POS])
all_centers = np.array(all_centers, dtype=float)  # (24, 3)

# --- スピーカー位置 ---
tx_positions = all_centers[SPK_INDICES]  # (N_tx=8, 3)
N_TX = tx_positions.shape[0]

# --- マイクロフォンアレイの中心位置 ---
spk_set = set(SPK_INDICES)
rx_center_indices = [idx for idx in range(all_centers.shape[0])]
rx_centers = all_centers[rx_center_indices]  # (N_center=24, 3)
N_CENTER = rx_centers.shape[0]

# --- マイクロフォンアレイの各チャンネルの位置 ---
rx_positions = np.zeros((N_CENTER, NUM_CHANNELS, 3), dtype=float)
for c_idx, (cx, cy, cz) in enumerate(rx_centers):
    for ch in range(NUM_CHANNELS):
        theta = THETA0 + ch * (2 * np.pi / NUM_CHANNELS)
        x = cx + MIC_RADIUS * np.cos(theta)
        y = cy + MIC_RADIUS * np.sin(theta)
        rx_positions[c_idx, ch] = [x, y, cz]

rx_positions_flat = rx_positions.reshape(-1, 3)
N_RX = rx_positions_flat.shape[0]


# ============================================================
# 2) 向き・指向性パターン配列
# ============================================================

# --- 送信機向き（N_tx, 3） ---
tx_ori = np.tile(TX_ORIENTATION / np.linalg.norm(TX_ORIENTATION), (N_TX, 1))

# --- 受信機向き（N_rx, 3） ---
rx_ori = np.tile(RX_ORIENTATION / np.linalg.norm(RX_ORIENTATION), (N_RX, 1))

# --- 指向性パターン（N_tx,), (N_rx,) ---
tx_patterns = [TX_PATTERN_DEFAULT] * N_TX
rx_patterns = [RX_PATTERN_DEFAULT] * N_RX


# ============================================================
# 3) JSON 出力
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

speaker_data = {
    "speaker": {
        "positions": tx_positions.tolist(),
        "orientations": tx_ori.tolist(),
        "patterns": tx_patterns
    }
}

receiver_data = {
    "receiver": {
        "positions": rx_positions_flat.tolist(),
        "orientations": rx_ori.tolist(),
        "patterns": rx_patterns
    }
}

with open(SPEAKER_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(speaker_data, f, indent=4)

with open(RECEIVER_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(receiver_data, f, indent=4)

print(f"[OK] speaker_data.json  -> {SPEAKER_JSON_PATH}  (N_tx={N_TX})")
print(f"[OK] receiver_data.json -> {RECEIVER_JSON_PATH} (N_rx={N_RX})")
