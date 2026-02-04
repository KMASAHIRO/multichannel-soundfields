import argparse
import os
import pickle
import random
from pathlib import Path


def collect_npz_files(dataset_dir: Path):
    files = []
    for tx_dir in sorted(dataset_dir.glob("tx_*"), key=lambda p: int(p.name.split("_")[1])):
        for rx_file in sorted(tx_dir.glob("rx_*.npz"), key=lambda p: int(p.stem.split("_")[1])):
            rel_path = rx_file.relative_to(dataset_dir).as_posix()
            files.append(rel_path)
    if not files:
        raise FileNotFoundError(f"No rx_*.npz found under {dataset_dir}")
    return files


def split_train_val(files, train_ratio, seed):
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    rng = random.Random(seed)
    files_shuffled = files[:]
    rng.shuffle(files_shuffled)
    train_len = int(len(files_shuffled) * train_ratio)
    train = files_shuffled[:train_len]
    val = files_shuffled[train_len:]
    return train, val


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val lists")
    parser.add_argument("--dataset_dir", required=True, help="Dataset directory (tx_*/rx_*.npz)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_npz_files(dataset_dir)
    train, val = split_train_val(files, args.train_ratio, args.seed)

    out_path = output_dir / "train_val_split.pkl"
    with out_path.open("wb") as f:
        pickle.dump([train, val], f)

    print(f"[Split] train={len(train)}, val={len(val)}")
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
