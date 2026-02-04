import argparse
from pathlib import Path
import numpy as np
import soundfile as sf


def load_points(points_file: Path) -> dict:
    points = {}
    with points_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(f"Invalid line in points file: {line}")
            idx_str, x, y, z = parts
            idx = int(idx_str)
            points[idx] = np.array([float(x), float(y), float(z)], dtype=np.float32)
    return points


def mic_positions(
    center: np.ndarray,
    num_channels: int = 8,
    radius: float = 0.0365,
    phase_offset: float = np.pi / 2,
) -> np.ndarray:
    positions = np.zeros((num_channels, 3), dtype=np.float32)
    for ch in range(num_channels):
        theta = phase_offset + ch * (2 * np.pi / num_channels)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        positions[ch] = [x, y, center[2]]
    return positions


def parse_wav_name(path: Path):
    parts = path.stem.split("_")
    if len(parts) != 3:
        return None
    try:
        tx_idx = int(parts[0])
        rx_idx = int(parts[1])
        ch_idx = int(parts[2])
    except ValueError:
        return None
    return tx_idx, rx_idx, ch_idx


def find_wavs(data_dir: Path):
    wavs = list(data_dir.glob("*.wav"))
    if wavs:
        return wavs
    return list(data_dir.rglob("*.wav"))


def convert_ir_to_npz(
    data_dir: Path,
    output_dir: Path,
    ir_start: int = 8720,
    ir_len: int = 1600,
) -> None:
    points_file = data_dir / "points.txt"
    points = load_points(points_file)
    wav_files = find_wavs(data_dir)
    if not wav_files:
        raise FileNotFoundError(f"No wav files found in {data_dir}")

    grouped = {}
    for wav_path in wav_files:
        parsed = parse_wav_name(wav_path)
        if parsed is None:
            print(f"Skipping (unrecognized name): {wav_path.name}")
            continue
        tx_idx, rx_idx, ch_idx = parsed
        grouped.setdefault((tx_idx, rx_idx), {})[ch_idx] = wav_path

    num_channels = 8
    sample_rate = 16000
    total_written = 0

    tx_indices = sorted({tx for (tx, _rx) in grouped.keys()})
    tx_map = {tx: i for i, tx in enumerate(tx_indices)}

    rx_map_per_tx = {}
    for tx in tx_indices:
        rx_indices = sorted({rx for (t, rx) in grouped.keys() if t == tx})
        rx_map_per_tx[tx] = {rx: i for i, rx in enumerate(rx_indices)}

    for (tx_idx, rx_idx), ch_map in sorted(grouped.items()):
        missing = [ch for ch in range(1, num_channels + 1) if ch not in ch_map]
        if missing:
            print(f"Skipping tx {tx_idx} rx {rx_idx}: missing channels {missing}")
            continue
        if tx_idx not in points or rx_idx not in points:
            print(f"Skipping tx {tx_idx} rx {rx_idx}: missing point definition")
            continue

        ir_list = []
        lengths = []
        for ch in range(1, num_channels + 1):
            wav_path = ch_map[ch]
            audio, sr = sf.read(wav_path, always_2d=False)
            if audio.ndim > 1:
                audio = audio[:, 0]
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                print(f"Warning: sample rate mismatch in {wav_path.name} ({sr} != {sample_rate})")
            if len(audio) < ir_start + ir_len:
                print(f"Skipping {wav_path.name}: too short for slice")
                ir_list = []
                break
            sliced = audio[ir_start:ir_start + ir_len]
            ir_list.append(sliced.astype(np.float32))
            lengths.append(len(sliced))

        if not ir_list:
            continue

        ir = np.stack(ir_list, axis=0)
        position_tx = points[tx_idx]
        position_rx = mic_positions(points[rx_idx], num_channels=num_channels)

        tx_dir = output_dir / f"tx_{tx_map[tx_idx]}"
        tx_dir.mkdir(parents=True, exist_ok=True)
        out_path = tx_dir / f"rx_{rx_map_per_tx[tx_idx][rx_idx]}.npz"
        np.savez(
            out_path,
            ir=ir,
            position_rx=position_rx,
            position_tx=position_tx,
        )
        total_written += 1
    
    print(f"written: {total_written} files")


def main():
    parser = argparse.ArgumentParser(description="Format real WAV data into npz files.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing WAV files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    convert_ir_to_npz(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
