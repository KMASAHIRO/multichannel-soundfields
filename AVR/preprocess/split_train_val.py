#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import random
from pathlib import Path


def collect_npz_files(dataset_dir: Path):
    files = []
    for tx_dir in sorted(dataset_dir.glob("tx_*"), key=lambda p: int(p.name.split("_")[1])):
        for rx_file in sorted(tx_dir.glob("rx_*.npz"), key=lambda p: int(p.stem.split("_")[1])):
            files.append(rx_file.relative_to(dataset_dir).as_posix())
    if not files:
        raise FileNotFoundError(f"No rx_*.npz found under {dataset_dir}")
    return files


def main():
    parser = argparse.ArgumentParser(description="Split AVR dataset into train/val")
    parser.add_argument("--dataset_dir", required=True, help="Dataset directory")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    files = collect_npz_files(dataset_dir)

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")

    rng = random.Random(args.seed)
    rng.shuffle(files)
    train_len = int(len(files) * args.train_ratio)

    train_files = files[:train_len]
    val_files = files[train_len:]

    split = {"train": train_files, "test": val_files}
    out_path = dataset_dir / "train_test_split.pkl"
    with out_path.open("wb") as f:
        pickle.dump(split, f)

    print(f"[Split] train={len(train_files)}, val={len(val_files)}")
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
