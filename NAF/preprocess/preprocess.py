import argparse
from pathlib import Path

import h5py
import librosa
import numpy as np
import yaml


def if_compute(arg):
    unwrapped_angle = np.unwrap(arg).astype(np.single)
    return np.concatenate([unwrapped_angle[:, :, 0:1], np.diff(unwrapped_angle, n=1)], axis=-1)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("fs", 16000)
    cfg.setdefault("n_fft", 512)
    cfg.setdefault("hop_size", 128)
    cfg.setdefault("window", "hann")
    cfg.setdefault("log_eps", 1e-3)
    cfg.setdefault("mag_std_eps", 0.1)
    return cfg


def list_npz(dataset_dir: Path):
    npz_files = []
    for tx_dir in sorted(dataset_dir.glob("tx_*"), key=lambda p: int(p.name.split("_")[1])):
        for rx_file in sorted(tx_dir.glob("rx_*.npz"), key=lambda p: int(p.stem.split("_")[1])):
            npz_files.append(rx_file)
    if not npz_files:
        raise FileNotFoundError(f"No rx_*.npz found under {dataset_dir}")
    return npz_files


def make_key(tx_dir: Path, rx_file: Path) -> str:
    return f"{tx_dir.name}_{rx_file.stem}"


def compute_spec(wav_data_prepad: np.ndarray, n_fft: int, hop_size: int, log_eps: float):
    wav_data = librosa.util.fix_length(
        wav_data_prepad, size=wav_data_prepad.shape[-1] + n_fft // 2
    )
    transformed_data = np.array(librosa.stft(wav_data, n_fft=n_fft, hop_length=hop_size))[:, :-1]
    real_component = np.abs(transformed_data)
    img_component = np.angle(transformed_data)
    gen_if = if_compute(img_component) / np.pi
    return np.log(real_component + log_eps), gen_if


def pad(input_arr, max_len_in, constant):
    return np.pad(input_arr, [[0, 0], [0, 0], [0, max_len_in - input_arr.shape[2]]], constant_values=constant)


def main():
    parser = argparse.ArgumentParser(description="Preprocess NAF dataset")
    parser.add_argument("--config", required=True, help="Path to preprocess_config.yml")
    parser.add_argument("--dataset_dir", required=True, help="Dataset directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy config
    with open(output_dir / "preprocess_config.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    n_fft = int(cfg["n_fft"])
    hop_size = int(cfg["hop_size"])
    log_eps = float(cfg["log_eps"])
    mag_std_eps = float(cfg["mag_std_eps"])

    magnitudes_path = output_dir / "magnitudes.h5"
    phases_path = output_dir / "phases.h5"
    positions_path = output_dir / "positions.h5"

    npz_files = list_npz(dataset_dir)

    f_mag = h5py.File(magnitudes_path, "w")
    f_phase = h5py.File(phases_path, "w")
    f_pos = h5py.File(positions_path, "w")

    length_tracker = []

    for rx_file in npz_files:
        tx_dir = rx_file.parent
        key = make_key(tx_dir, rx_file)
        data = np.load(rx_file)
        ir = data["ir"].astype(np.float32)
        if ir.ndim == 1:
            ir = ir[None, :]

        # Compute specs
        real_spec, img_spec = compute_spec(ir, n_fft=n_fft, hop_size=hop_size, log_eps=log_eps)
        length_tracker.append(real_spec.shape[2])

        f_mag.create_dataset(key, data=real_spec.astype(np.half))
        f_phase.create_dataset(key, data=img_spec.astype(np.half))

        pos_grp = f_pos.create_group(key)
        pos_grp.create_dataset("position_tx", data=data["position_tx"].astype(np.float32))
        pos_grp.create_dataset("position_rx", data=data["position_rx"].astype(np.float32))

    max_len = int(np.max(length_tracker))
    f_mag.attrs["max_len"] = np.int32(max_len)

    # Compute mean/std for magnitudes
    keys = list(f_mag.keys())
    all_arrs = []
    for idx in np.random.choice(len(keys), size=len(keys), replace=False):
        all_arrs.append(pad(f_mag[keys[idx]], max_len, constant=np.log(log_eps)).astype(np.single))
    mean_val = np.mean(all_arrs, axis=(0, 1))
    std_val = np.std(all_arrs, axis=(0, 1)) + mag_std_eps
    f_mag.attrs["magnitude_mean"] = mean_val.astype(np.float32)
    f_mag.attrs["magnitude_std"] = std_val.astype(np.float32)
    # Compute phase std
    keys = list(f_phase.keys())
    all_arrs = []
    for idx in np.random.choice(len(keys), size=len(keys), replace=False):
        all_arrs.append(pad(f_phase[keys[idx]], max_len, constant=0.0).astype(np.single))
    std_val = np.std(all_arrs)
    f_phase.attrs["phase_std"] = np.float32(std_val)

    f_mag.close()
    f_phase.close()
    f_pos.close()

    print(f"[OK] magnitudes: {magnitudes_path}")
    print(f"[OK] phases    : {phases_path}")
    print(f"[OK] positions : {positions_path}")


if __name__ == "__main__":
    main()
