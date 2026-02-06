import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import h5py
import numpy as np
import torch


def _normalize_xy(xy: np.ndarray, min_xy: np.ndarray, max_xy: np.ndarray) -> np.ndarray:
    return np.clip(((xy - min_xy) / (max_xy - min_xy) - 0.5) * 2.0, -1.0, 1.0)


@dataclass
class DatasetConfig:
    data_dir: str
    pixel_count: int
    reg_eps: float
    model_type: str
    ch_num: int
    xy_min: np.ndarray = None
    xy_max: np.ndarray = None


class soundsamples(torch.utils.data.Dataset):
    def __init__(self, cfg: DatasetConfig):
        self.data_dir = cfg.data_dir
        self.pixel_count = cfg.pixel_count
        self.pos_reg_amt = cfg.reg_eps
        self.model_type = cfg.model_type
        self.ch_num = 1 if self.model_type == "NAF" else cfg.ch_num

        self.magnitudes_path = os.path.join(self.data_dir, "magnitudes.h5")
        self.phases_path = os.path.join(self.data_dir, "phases.h5")
        self.positions_path = os.path.join(self.data_dir, "positions.h5")
        self.split_path = os.path.join(self.data_dir, "train_val_split.pkl")

        with open(self.split_path, "rb") as f:
            train_val_split = pickle.load(f)
        self.sound_files = train_val_split[0]
        self.sound_files_val = train_val_split[1]

        with h5py.File(self.magnitudes_path, "r") as f:
            self.keys = list(f.keys())
            self.mean = torch.from_numpy(f.attrs["magnitude_mean"]).float()[None]
            self.std = torch.from_numpy(f.attrs["magnitude_std"]).float()[None]
            self.max_len = int(f.attrs["max_len"])

        with h5py.File(self.phases_path, "r") as f:
            self.phase_std = float(f.attrs["phase_std"])

        with h5py.File(self.positions_path, "r") as f:
            tx_list = []
            rx_list = []
            for key in f.keys():
                tx_list.append(f[key]["position_tx"][...])
                rx_pos = f[key]["position_rx"][...]
                if self.model_type == "NAF":
                    rx_list.append(rx_pos.reshape(-1, rx_pos.shape[-1]))
                else:
                    rx_list.append(rx_pos.mean(axis=0)[None, :])
            all_tx = np.array(tx_list)
            all_rx = np.concatenate(rx_list, axis=0)
            all_xy = np.concatenate([all_tx[:, :2], all_rx[:, :2]], axis=0)
            self.min_pos = all_xy.min(axis=0)
            self.max_pos = all_xy.max(axis=0)

        if cfg.xy_min is not None and cfg.xy_max is not None:
            self.min_pos = np.asarray(cfg.xy_min, dtype=np.float32)
            self.max_pos = np.asarray(cfg.xy_max, dtype=np.float32)

        self.sound_data = None
        self.phase_data = None
        self.pos_data = None
        if self.model_type == "NAF":
            self.sound_files = self._expand_channels(self.sound_files)
            self.sound_files_val = self._expand_channels(self.sound_files_val)

    def _normalize_key(self, key: str) -> str:
        if "/" in key or key.endswith(".npz"):
            key = key.replace("\\", "/")
            key = key.replace(".npz", "")
            key = key.replace("/", "_")
        return key

    def _expand_channels(self, keys):
        expanded = []
        with h5py.File(self.magnitudes_path, "r") as f:
            for key in keys:
                k = self._normalize_key(key)
                ch = f[k].shape[0]
                for c in range(ch):
                    expanded.append((key, c))
        return expanded

    def __len__(self):
        return len(self.sound_files)

    def _open_files(self):
        if self.sound_data is None:
            self.sound_data = h5py.File(self.magnitudes_path, "r")
        if self.phase_data is None:
            self.phase_data = h5py.File(self.phases_path, "r")
        if self.pos_data is None:
            self.pos_data = h5py.File(self.positions_path, "r")

    def _load_sample(self, key):
        self._open_files()
        ch_idx = None
        if isinstance(key, tuple):
            key, ch_idx = key
        key = self._normalize_key(key)
        mag = torch.from_numpy(self.sound_data[key][:]).float()
        phase = torch.from_numpy(self.phase_data[key][:]).float()
        if self.model_type == "NAF":
            if ch_idx is None:
                ch_idx = 0
            mag = mag[ch_idx : ch_idx + 1]
            phase = phase[ch_idx : ch_idx + 1]
        position_tx = self.pos_data[key]["position_tx"][...].astype(np.float32)
        position_rx = self.pos_data[key]["position_rx"][...].astype(np.float32)
        if self.model_type == "NAF":
            rx_point = position_rx[ch_idx]
        else:
            rx_point = position_rx.mean(axis=0)
        return mag, phase, position_tx, position_rx, rx_point, ch_idx, key

    def __getitem__(self, idx):
        loaded = False
        while not loaded:
            try:
                key = self.sound_files[idx]
                mag, phase, position_tx, position_rx, rx_point, ch_idx, base_key = self._load_sample(key)

                mag = mag[:, :, : self.max_len]
                phase = phase[:, :, : self.max_len]
                actual_spec_len = mag.shape[2]

                mag = (mag - self.mean[:, :, :actual_spec_len]) / self.std[:, :, :actual_spec_len]
                phase = phase / self.phase_std

                sound_size = mag.shape
                selected_time = np.random.randint(0, sound_size[2], self.pixel_count)
                selected_freq = np.random.randint(0, sound_size[1], self.pixel_count)

                tx_xy = position_tx[:2] + np.random.normal(0, 1, 2) * self.pos_reg_amt
                rx_xy = rx_point[:2] + np.random.normal(0, 1, 2) * self.pos_reg_amt

                start_position = torch.from_numpy(_normalize_xy(tx_xy, self.min_pos, self.max_pos))[None]
                end_position = torch.from_numpy(_normalize_xy(rx_xy, self.min_pos, self.max_pos))[None]
                total_position = torch.cat((start_position, end_position), dim=1).float()

                total_non_norm_position = torch.cat(
                    (torch.from_numpy(tx_xy)[None], torch.from_numpy(rx_xy)[None]), dim=1
                ).float()

                selected_mag = mag[:, selected_freq, selected_time]
                selected_phase = phase[:, selected_freq, selected_time]
                selected_total = torch.cat((selected_mag, selected_phase), dim=0)
                loaded = True
            except Exception as e:
                print(key)
                print(e)
                print("Failed to load sound sample")

        return (
            selected_total,
            total_position,
            total_non_norm_position,
            2.0 * torch.from_numpy(selected_freq).float() / 255.0 - 1.0,
            2.0 * torch.from_numpy(selected_time).float() / float(self.max_len - 1) - 1.0,
        )

    def get_item_val(self, idx):
        key = self.sound_files_val[idx]
        mag, phase, position_tx, position_rx, rx_point, _ch_idx, _base_key = self._load_sample(key)

        mag = mag[:, :, : self.max_len]
        phase = phase[:, :, : self.max_len]
        actual_spec_len = mag.shape[2]

        mag = (mag - self.mean[:, :, :actual_spec_len]) / self.std[:, :, :actual_spec_len]
        phase = phase / self.phase_std

        sound_size = mag.shape
        self.sound_size = sound_size

        selected_time = np.arange(0, sound_size[2])
        selected_freq = np.arange(0, sound_size[1])
        selected_time, selected_freq = np.meshgrid(selected_time, selected_freq)
        selected_time = selected_time.reshape(-1)
        selected_freq = selected_freq.reshape(-1)

        tx_xy = position_tx[:2]
        rx_xy = rx_point[:2]
        start_position = torch.from_numpy(_normalize_xy(tx_xy, self.min_pos, self.max_pos))[None]
        end_position = torch.from_numpy(_normalize_xy(rx_xy, self.min_pos, self.max_pos))[None]
        total_position = torch.cat((start_position, end_position), dim=1).float()
        total_non_norm_position = torch.cat(
            (torch.from_numpy(tx_xy)[None], torch.from_numpy(rx_xy)[None]), dim=1
        ).float()

        selected_mag = mag[:, selected_freq, selected_time]
        selected_phase = phase[:, selected_freq, selected_time]
        selected_total = torch.cat((selected_mag, selected_phase), dim=0)

        return (
            selected_total,
            total_position,
            total_non_norm_position,
            2.0 * torch.from_numpy(selected_freq).float() / 255.0 - 1.0,
            2.0 * torch.from_numpy(selected_time).float() / float(self.max_len - 1) - 1.0,
            position_tx.astype(np.float32),
            (rx_point.astype(np.float32) if self.model_type == "NAF" else position_rx.astype(np.float32)),
        )
