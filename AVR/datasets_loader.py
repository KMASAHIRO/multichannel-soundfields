import numpy as np
import math
import pickle
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Dict, Any


class AVRDataset(Dataset):
    def __init__(self, dataset_dir: str, split: str, seq_len: int, model_type: str, ch_num: int):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.seq_len = seq_len
        self.model_type = model_type
        self.ch_num = ch_num

        split_path = self.dataset_dir / "train_test_split.pkl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        with split_path.open("rb") as f:
            split_dict = pickle.load(f)

        key = "train" if split == "train" else "test"
        rel_paths = split_dict.get(key, [])
        if not rel_paths:
            raise ValueError(f"No files found for split '{key}' in {split_path}")

        self.files: List[Path] = [self.dataset_dir / Path(p) for p in rel_paths]
        self.samples: List[Tuple[int, int]] = []
        for file_idx, _ in enumerate(self.files):
            for ch_idx in range(self.ch_num):
                self.samples.append((file_idx, ch_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_npz(self, file_idx: int) -> Dict[str, Any]:
        data = np.load(self.files[file_idx])
        return {
            "ir": data["ir"],
            "position_rx": data["position_rx"],
            "position_tx": data["position_tx"],
        }

    def __getitem__(self, idx: int):
        file_idx, ch_idx = self.samples[idx]
        data = self._load_npz(file_idx)

        ir = data["ir"]
        position_rx = data["position_rx"]
        position_tx = data["position_tx"]

        if ir.shape[0] != self.ch_num:
            raise ValueError(f"Expected ir shape (ch_num, ir_len) with ch_num={self.ch_num}, got {ir.shape}")

        ir_ch = ir[ch_idx][: self.seq_len]
        wave_signal = np.fft.rfft(ir_ch)

        if self.model_type == "AVR":
            rx_point = position_rx[ch_idx]
        else:
            rx_point = position_rx.mean(axis=0)

        wave_signal = torch.tensor(wave_signal, dtype=torch.complex64)
        rx_point = torch.tensor(rx_point, dtype=torch.float32)
        position_tx = torch.tensor(position_tx, dtype=torch.float32)
        ch_idx_tensor = torch.tensor(ch_idx, dtype=torch.long)

        return wave_signal, rx_point, position_tx, ch_idx_tensor


def quaternion_to_direction_vector(q):
    """Convert a quaternion to direction vectors in Cartesian coordinates

    Parameters
    ----------
    q : Quaternion, given as a Tensor [x, y, z, w].

    Returnsdata
    -------
    Direction vectors as pts_x, pts_y, pts_z
    """

    x, y, z, w = q

    # Convert quaternion to forward direction vector
    fwd_x = 2 * (x*z + w*y)
    fwd_y = 2 * (y*z - w*x)
    fwd_z = 1 - 2 * (x*x + y*y)

    # Normalize the vector (in case it's not exactly 1 due to numerical precision)
    norm = math.sqrt(fwd_x**2 + 0**2 + fwd_z**2)
    
    return np.array([-fwd_x / norm, -fwd_z / norm, 0])
