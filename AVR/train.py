import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import yaml

from datasets_loader import AVRDataset
from model import AVRModel
from renderer import AVRRender
from utils.criterion import Criterion

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from metrics.IR_metrics import ir_metrics
from metrics.DoA_metrics import doa_metrics


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def build_model_and_renderer(cfg: dict) -> Tuple[AVRModel, AVRRender]:
    setting = cfg["setting"]
    param = cfg["param"]
    model_cfg = dict(cfg["model"])

    model_type = setting["model_type"]
    channel_embed = dict(model_cfg.get("channel_embed", {}))
    if model_type == "AVR":
        channel_embed["is_embed"] = False
    else:
        channel_embed["is_embed"] = True
        channel_embed["ch_num"] = setting["ch_num"]
    model_cfg["channel_embed"] = channel_embed

    model = AVRModel(model_cfg)
    render = AVRRender(
        networks_fn=model,
        n_samples=param["n_samples"],
        near=param["near"],
        far=param["far"],
        n_azi=param["n_azi"],
        n_ele=param["n_ele"],
        speed=setting["speed"],
        fs=setting["fs"],
        pathloss=param["pathloss"],
        xyz_min=setting["xyz_min"],
        xyz_max=setting["xyz_max"],
    )
    return model, render


def build_criterion(cfg: dict) -> Criterion:
    setting = cfg["setting"]
    param = cfg["param"]
    model_type = setting["model_type"]
    das_weight = param.get("das_loss_weight", 0.0) if model_type == "AVR++" else 0.0

    criterion_cfg = {
        "spec_loss_weight": param["spec_loss_weight"],
        "amplitude_loss_weight": param["amplitude_loss_weight"],
        "angle_loss_weight": param["angle_loss_weight"],
        "time_loss_weight": param["time_loss_weight"],
        "energy_loss_weight": param["energy_loss_weight"],
        "multistft_loss_weight": param["multistft_loss_weight"],
        "das_loss_weight": das_weight,
        "beta": param.get("softargmax_beta", 100.0),
        "das_n_fft": int(cfg.get("doa_metric", {}).get("n_fft", 512)),
        "ch_num": int(setting["ch_num"]),
    }

    render_cfg = {"fs": setting["fs"], "speed": setting["speed"]}
    return Criterion(criterion_cfg, render_cfg)


def load_latest_checkpoint(ckpt_dir: Path) -> Path:
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("*.tar"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def angular_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a) % 360.0
    b = np.asarray(b) % 360.0
    d = np.abs(a - b)
    return np.minimum(d, 360.0 - d)


def run_validation(
    renderer: AVRRender,
    criterion: Criterion,
    dataset_val: AVRDataset,
    cfg: dict,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], float]:
    setting = cfg["setting"]
    param = cfg["param"]
    doa_cfg = cfg.get("doa_metric", {})
    model_type = setting["model_type"]
    seq_len = cfg["model"]["signal_output_dim"]
    ch_num = setting["ch_num"]

    val_losses = {
        "spec": 0.0,
        "amplitude": 0.0,
        "angle": 0.0,
        "time": 0.0,
        "energy": 0.0,
        "multistft": 0.0,
        "das": 0.0,
        "total": 0.0,
    }
    count = 0

    position_tx_list: List[np.ndarray] = []
    position_rx_list: List[np.ndarray] = []
    ir_gt_list: List[np.ndarray] = []
    ir_pred_list: List[np.ndarray] = []

    renderer.eval()
    with torch.no_grad():
        for file_path in dataset_val.files:
            data = np.load(file_path)
            ir = data["ir"]
            position_rx = data["position_rx"]
            position_tx = data["position_tx"]

            if ir.shape[0] != ch_num:
                raise ValueError(f"Expected ir shape (ch_num, ir_len) with ch_num={ch_num}, got {ir.shape}")

            if ir.shape[-1] < seq_len:
                pad = seq_len - ir.shape[-1]
                ir = np.pad(ir, ((0, 0), (0, pad)), mode="constant")
            ir = ir[:, :seq_len]

            if model_type == "AVR":
                rays_o = torch.tensor(position_rx, dtype=torch.float32, device=device)
                ch_idx = None
            else:
                rx_center = position_rx.mean(axis=0)
                rays_o = torch.tensor(
                    np.repeat(rx_center[None, :], ch_num, axis=0),
                    dtype=torch.float32,
                    device=device,
                )
                ch_idx = torch.arange(ch_num, device=device, dtype=torch.long)

            position_tx_batch = torch.tensor(
                np.repeat(position_tx[None, :], ch_num, axis=0),
                dtype=torch.float32,
                device=device,
            )

            pred_sig = renderer(rays_o, position_tx_batch, ch_idx=ch_idx)
            pred_sig = pred_sig[..., 0] + 1j * pred_sig[..., 1]

            ori_sig = np.fft.rfft(ir, axis=-1)
            ori_sig = torch.tensor(ori_sig, dtype=pred_sig.dtype, device=device)

            das_rx = torch.tensor(position_rx, dtype=torch.float32, device=device) if model_type == "AVR++" else None
            spec_loss, amp_loss, angle_loss, time_loss, energy_loss, multistft_loss, das_loss, _, _ = criterion(
                pred_sig, ori_sig, rx_positions=das_rx
            )
            total_loss = spec_loss + amp_loss + angle_loss + time_loss + energy_loss + multistft_loss + das_loss

            val_losses["spec"] += float(spec_loss.detach())
            val_losses["amplitude"] += float(amp_loss.detach())
            val_losses["angle"] += float(angle_loss.detach())
            val_losses["time"] += float(time_loss.detach())
            val_losses["energy"] += float(energy_loss.detach())
            val_losses["multistft"] += float(multistft_loss.detach())
            val_losses["das"] += float(das_loss.detach())
            val_losses["total"] += float(total_loss.detach())
            count += 1

            ir_pred = torch.fft.irfft(pred_sig, n=seq_len, dim=-1).detach().cpu().numpy()
            ir_gt = ir.astype(np.float32)

            position_tx_list.append(position_tx.astype(np.float32))
            position_rx_list.append(position_rx.astype(np.float32))
            ir_gt_list.append(ir_gt)
            ir_pred_list.append(ir_pred.astype(np.float32))

    if count == 0:
        raise RuntimeError("No validation samples found.")

    for key in val_losses:
        val_losses[key] /= count

    position_tx_arr = np.stack(position_tx_list, axis=0)
    position_rx_arr = np.stack(position_rx_list, axis=0)
    ir_gt_arr = np.stack(ir_gt_list, axis=0)
    ir_pred_arr = np.stack(ir_pred_list, axis=0)

    metrics = ir_metrics(ir_gt_arr, ir_pred_arr, fs=setting["fs"])
    doa_gt_deg, doa_pred_deg, doa_true_deg = doa_metrics(
        ir_gt_arr,
        ir_pred_arr,
        algorithm=doa_cfg.get("algorithm", "NormMUSIC"),
        n_fft=int(doa_cfg.get("n_fft", 512)),
        fs=int(setting["fs"]),
        rx_positions=position_rx_arr,
        tx_positions=position_tx_arr,
    )

    val_results = {
        "position_tx": position_tx_arr,
        "position_rx": position_rx_arr,
        "ir_gt": ir_gt_arr,
        "ir_pred": ir_pred_arr,
        "doa_true_deg": doa_true_deg,
        "doa_gt_deg": doa_gt_deg,
        "doa_pred_deg": doa_pred_deg,
        "metric_angle": metrics["angle_error"],
        "metric_amp": metrics["amp_error"],
        "metric_env_pct": metrics["env_error_percent"],
        "metric_t60_pct": metrics["t60_error_percent"],
        "metric_c50_db": metrics["c50_error_db"],
        "metric_edt_ms": metrics["edt_error_ms"],
    }

    fallback = float(doa_cfg.get("fallback_value", 999.0))
    if doa_pred_deg is None:
        score = fallback
    else:
        valid = ~np.isnan(doa_pred_deg)
        if not np.any(valid):
            score = fallback
        else:
            err = angular_error_deg(doa_pred_deg[valid], doa_true_deg[valid])
            if np.isnan(err).any():
                score = fallback
            else:
                score = float(np.mean(err))

    return val_losses, val_results, score


def run_training(cfg: dict, data_dir: str, output_dir: str) -> None:
    setting = cfg["setting"]
    param = cfg["param"]
    model_type = setting["model_type"]

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    save_yaml(output_root / "train_config.yml", cfg)

    ckpt_dir = output_root / "ckpt"
    loss_dir = output_root / "loss"
    val_dir = output_root / "val_results"
    ckpt_dir.mkdir(exist_ok=True)
    loss_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, renderer = build_model_and_renderer(cfg)
    renderer = renderer.to(device)
    criterion = build_criterion(cfg).to(device)

    optimizer = torch.optim.Adam(
        renderer.parameters(),
        lr=float(param["lr"]),
        weight_decay=float(param["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(setting["T_max"]),
        eta_min=float(param["eta_min"]),
    )

    global_step = 0
    eval_count = 0
    best_scores: List[Tuple[float, Path]] = []

    if setting.get("resume", False):
        latest_ckpt = load_latest_checkpoint(ckpt_dir)
        if latest_ckpt is not None:
            ckpt = torch.load(latest_ckpt, map_location=device)
            renderer.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            global_step = int(ckpt.get("global_step", 0))
            eval_count = int(ckpt.get("eval_count", 0))

    train_dataset = AVRDataset(
        dataset_dir=data_dir,
        split="train",
        seq_len=cfg["model"]["signal_output_dim"],
        model_type=model_type,
        ch_num=setting["ch_num"],
    )
    val_dataset = AVRDataset(
        dataset_dir=data_dir,
        split="test",
        seq_len=cfg["model"]["signal_output_dim"],
        model_type=model_type,
        ch_num=setting["ch_num"],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(setting["batch_size"]),
        shuffle=True,
        num_workers=int(setting.get("load_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )

    max_steps = int(setting["T_max"])
    val_freq = int(setting["val_freq"])
    grad_clip = float(setting.get("grad_clip_norm", 1.0))
    save_best_k = int(setting.get("save_best_k", 10))

    train_acc = {
        "spec": 0.0,
        "amplitude": 0.0,
        "angle": 0.0,
        "time": 0.0,
        "energy": 0.0,
        "multistft": 0.0,
        "das": 0.0,
        "total": 0.0,
        "count": 0,
    }

    while global_step < max_steps:
        renderer.train()
        for batch in train_loader:
            if global_step >= max_steps:
                break
            ori_sig, position_rx, position_tx, ch_idx = batch

            ori_sig = ori_sig.to(device)
            position_rx = position_rx.to(device)
            position_tx = position_tx.to(device)
            ch_idx = ch_idx.to(device)

            if model_type == "AVR":
                pred_sig = renderer(position_rx, position_tx)
            else:
                pred_sig = renderer(position_rx, position_tx, ch_idx=ch_idx)

            pred_sig = pred_sig[..., 0] + 1j * pred_sig[..., 1]
            das_rx = position_rx if model_type == "AVR++" else None
            spec_loss, amp_loss, angle_loss, time_loss, energy_loss, multistft_loss, das_loss, _, _ = criterion(
                pred_sig, ori_sig, rx_positions=das_rx
            )
            total_loss = spec_loss + amp_loss + angle_loss + time_loss + energy_loss + multistft_loss + das_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(renderer.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()

            train_acc["spec"] += float(spec_loss.detach())
            train_acc["amplitude"] += float(amp_loss.detach())
            train_acc["angle"] += float(angle_loss.detach())
            train_acc["time"] += float(time_loss.detach())
            train_acc["energy"] += float(energy_loss.detach())
            train_acc["multistft"] += float(multistft_loss.detach())
            train_acc["das"] += float(das_loss.detach())
            train_acc["total"] += float(total_loss.detach())
            train_acc["count"] += 1

            global_step += 1

            if global_step % val_freq == 0 or global_step == max_steps:
                eval_count += 1
                val_losses, val_results, score = run_validation(renderer, criterion, val_dataset, cfg, device)

                if train_acc["count"] == 0:
                    train_acc["count"] = 1
                train_avg = {k: train_acc[k] / train_acc["count"] for k in train_acc if k != "count"}

                loss_payload = {
                    "epoch": np.int32(eval_count),
                    "loss_train": np.float32(train_avg["total"]),
                    "spec_train": np.float32(train_avg["spec"]),
                    "amplitude_train": np.float32(train_avg["amplitude"]),
                    "angle_train": np.float32(train_avg["angle"]),
                    "time_train": np.float32(train_avg["time"]),
                    "energy_train": np.float32(train_avg["energy"]),
                    "multistft_train": np.float32(train_avg["multistft"]),
                    "loss_val": np.float32(val_losses["total"]),
                    "spec_val": np.float32(val_losses["spec"]),
                    "amplitude_val": np.float32(val_losses["amplitude"]),
                    "angle_val": np.float32(val_losses["angle"]),
                    "time_val": np.float32(val_losses["time"]),
                    "energy_val": np.float32(val_losses["energy"]),
                    "multistft_val": np.float32(val_losses["multistft"]),
                }
                if model_type == "AVR++":
                    loss_payload["das_train"] = np.float32(train_avg["das"])
                    loss_payload["das_val"] = np.float32(val_losses["das"])

                np.savez_compressed(loss_dir / f"epoch{eval_count:04d}.npz", **loss_payload)
                np.savez_compressed(val_dir / f"epoch{eval_count:04d}.npz", **val_results)

                ckpt_path = ckpt_dir / f"eval{eval_count:04d}.tar"
                torch.save(
                    {
                        "model_state_dict": renderer.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "global_step": global_step,
                        "eval_count": eval_count,
                        "config": cfg,
                    },
                    ckpt_path,
                )

                best_scores.append((score, ckpt_path))
                best_scores = sorted(best_scores, key=lambda x: x[0])[:save_best_k]

                keep_eval = {p for _, p in best_scores}
                for p in ckpt_dir.glob("eval*.tar"):
                    if p not in keep_eval:
                        p.unlink()

                for order_idx in range(len(best_scores), 0, -1):
                    _, src = best_scores[order_idx - 1]
                    final_path = ckpt_dir / f"best{order_idx:04d}.tar"
                    if src == final_path:
                        continue
                    if final_path.exists():
                        final_path.unlink()
                    if src.exists():
                        src.rename(final_path)

                best_scores = [
                    (score, ckpt_dir / f"best{order_idx:04d}.tar")
                    for order_idx, (score, _) in enumerate(best_scores, start=1)
                ]

                train_acc = {k: 0.0 for k in train_acc}
                train_acc["count"] = 0

    # eval checkpoints are cleaned during training


def main():
    parser = argparse.ArgumentParser(description="AVR training")
    parser.add_argument("--config", required=True, help="Path to train_config.yml")
    parser.add_argument("--data_dir", required=True, help="Dataset directory")
    parser.add_argument("--output_dir", required=True, help="Training output directory")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    run_training(cfg, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
