import argparse
import math
import os
import socket
import shutil
from contextlib import closing
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from model.modules import embedding_module_log
from model.networks import kernel_residual_fc_embeds
from sound_loader import DatasetConfig, soundsamples

# metrics
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from metrics.IR_metrics import ir_metrics
from metrics.DoA_metrics import doa_metrics

import librosa
import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def get_spectrograms(input_stft, input_if, log_eps):
    padded_input_stft = np.concatenate((input_stft, input_stft[:, -1:]), axis=1)
    padded_input_if = np.concatenate((input_if, input_if[:, -1:]), axis=1)
    unwrapped = np.cumsum(padded_input_if, axis=-1) * np.pi
    phase_val = np.cos(unwrapped) + 1j * np.sin(unwrapped)
    return (np.exp(padded_input_stft) - log_eps) * phase_val


def istft_ir(spec: np.ndarray, nfft=512, hop=128, win="hann") -> np.ndarray:
    y = librosa.istft(
        spec,
        n_fft=nfft,
        hop_length=hop,
        win_length=nfft,
        window=win,
        center=True,
    )
    return y.astype(np.float32)


def angular_error_deg(a, b):
    a = np.asarray(a) % 360.0
    b = np.asarray(b) % 360.0
    d = np.abs(a - b)
    return np.minimum(d, 360.0 - d)


def build_model(cfg, dataset, device):
    xyz_embedder = embedding_module_log(num_freqs=cfg["embed_xyz_num_freqs"], ch_dim=2, max_freq=cfg["embed_xyz_max_freq"]).to(device)
    time_embedder = embedding_module_log(num_freqs=cfg["embed_time_num_freqs"], ch_dim=2, max_freq=cfg["embed_time_max_freq"]).to(device)
    freq_embedder = embedding_module_log(num_freqs=cfg["embed_freq_num_freqs"], ch_dim=2, max_freq=cfg["embed_freq_max_freq"]).to(device)

    input_ch = (
        2 * (2 * cfg["embed_xyz_num_freqs"] + 1)
        + 2 * (2 * cfg["embed_time_num_freqs"] + 1)
        + 2 * (2 * cfg["embed_freq_num_freqs"] + 1)
    )

    if "xy_min" in cfg and "xy_max" in cfg:
        min_xy = np.asarray(cfg["xy_min"], dtype=np.float32)
        max_xy = np.asarray(cfg["xy_max"], dtype=np.float32)
    else:
        min_xy = dataset.min_pos
        max_xy = dataset.max_pos

    net = kernel_residual_fc_embeds(
        input_ch=input_ch,
        ch_num=cfg["ch_num"],
        output_ch=2,
        intermediate_ch=cfg["features"],
        grid_ch=cfg["grid_features"],
        num_block=cfg["layers"],
        num_block_residual=cfg["layers_residual"],
        grid_gap=cfg["grid_gap"],
        grid_bandwidth=cfg["bandwith_init"],
        bandwidth_min=cfg["min_bandwidth"],
        bandwidth_max=cfg["max_bandwidth"],
        float_amt=cfg["position_float"],
        min_xy=min_xy,
        max_xy=max_xy,
        activation_func_name=cfg["activation_func_name"],
    ).to(device)

    return net, xyz_embedder, time_embedder, freq_embedder


def train_worker(rank, world_size, freeport, cfg):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = freeport
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    dataset = soundsamples(cfg["dataset_cfg"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["batch_size"] // world_size,
        shuffle=False,
        num_workers=3,
        sampler=train_sampler,
        drop_last=False,
        persistent_workers=True,
    )

    net, xyz_embedder, time_embedder, freq_embedder = build_model(cfg, dataset, device)
    ddp_net = DDP(net, find_unused_parameters=True, device_ids=[rank])

    criterion = torch.nn.MSELoss()

    grid_params = []
    main_params = []
    for name, p in ddp_net.named_parameters():
        if "grid" in name:
            grid_params.append(p)
        else:
            main_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": grid_params, "lr": cfg["lr_init"], "weight_decay": cfg["weight_decay_grid"]},
            {"params": main_params, "lr": cfg["lr_init"], "weight_decay": cfg["weight_decay_main"]},
        ]
    )

    exp_dir = cfg["output_dir"]
    loss_dir = os.path.join(exp_dir, "loss")
    val_dir = os.path.join(exp_dir, "val_results")
    ckpt_dir = os.path.join(exp_dir, "ckpt")
    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_scores = []

    if rank == 0:
        start_time = time()

    for epoch in range(1, cfg["epochs"] + 1):
        ddp_net.train()
        total_loss = 0.0
        total_mag = 0.0
        total_phase = 0.0
        iter_count = 0

        for data_stuff in data_loader:
            gt = data_stuff[0].to(device, non_blocking=True)
            position = data_stuff[1].to(device, non_blocking=True)
            non_norm_position = data_stuff[2].to(device, non_blocking=True)
            freqs = data_stuff[3].to(device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi
            times = data_stuff[4].to(device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi

            with torch.no_grad():
                position_embed = xyz_embedder(position).expand(-1, cfg["pixel_count"], -1)
                freq_embed = freq_embedder(freqs)
                time_embed = time_embedder(times)

            total_in = torch.cat((position_embed, freq_embed, time_embed), dim=2)
            optimizer.zero_grad(set_to_none=False)

            output = ddp_net(total_in, non_norm_position.squeeze(1)).transpose(1, 2)
            mag_loss = criterion(output[..., 0], gt[:, : cfg["ch_num"]]) * cfg["mag_alpha"]
            phase_loss = criterion(output[..., 1], gt[:, cfg["ch_num"] :]) * cfg["phase_alpha"]
            loss = mag_loss + phase_loss
            loss.backward()
            optimizer.step()

            if rank == 0:
                total_loss += float(loss.detach())
                total_mag += float(mag_loss.detach())
                total_phase += float(phase_loss.detach())
                iter_count += 1

        # lr decay
        decay_rate = cfg["lr_decay"]
        new_lrate = cfg["lr_init"] * (decay_rate ** (epoch / cfg["epochs"]))
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = new_lrate

        # validation
        ddp_net.eval()
        ir_pred_list = []
        ir_gt_list = []
        position_tx_list = []
        position_rx_list = []
        total_loss_val = 0.0
        total_mag_val = 0.0
        total_phase_val = 0.0
        val_iter = 0

        with torch.no_grad():
            for val_id in range(len(dataset.sound_files_val)):
                data_val = dataset.get_item_val(val_id)
                gt_val = data_val[0][None].to(device, non_blocking=True)
                position_val = data_val[1][None].to(device, non_blocking=True)
                non_norm_position_val = data_val[2][None].to(device, non_blocking=True)
                freqs_val = data_val[3][None].to(device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi
                times_val = data_val[4][None].to(device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi
                pos_tx = data_val[5]
                pos_rx = data_val[6]
                base_key = dataset.sound_files_val[val_id]
                ch_idx = -1
                if isinstance(base_key, tuple):
                    base_key, ch_idx = base_key

                pix_count_val = gt_val.shape[-1]
                position_embed_val = xyz_embedder(position_val).expand(-1, pix_count_val, -1)
                freq_embed_val = freq_embedder(freqs_val)
                time_embed_val = time_embedder(times_val)
                total_in_val = torch.cat((position_embed_val, freq_embed_val, time_embed_val), dim=2)

                output_val_list = []
                for split_id in range(-(-pix_count_val // cfg["pixel_count"])):
                    total_in_val_split = total_in_val[:, split_id * cfg["pixel_count"] : (split_id + 1) * cfg["pixel_count"], :]
                    if total_in_val_split.shape[1] < cfg["pixel_count"]:
                        pad = torch.zeros(
                            total_in_val_split.shape[0],
                            cfg["pixel_count"] - total_in_val_split.shape[1],
                            total_in_val_split.shape[2],
                            device=device,
                        )
                        total_in_val_split = torch.cat((total_in_val_split, pad), dim=1)
                        output_val_split = ddp_net(total_in_val_split, non_norm_position_val.squeeze(1)).transpose(1, 2)
                        output_val_split = output_val_split[:, :, : pix_count_val - split_id * cfg["pixel_count"], :]
                    else:
                        output_val_split = ddp_net(total_in_val_split, non_norm_position_val.squeeze(1)).transpose(1, 2)

                    mag_loss_val = criterion(
                        output_val_split[..., 0],
                        gt_val[:, : cfg["ch_num"], split_id * cfg["pixel_count"] : (split_id + 1) * cfg["pixel_count"]],
                    ) * cfg["mag_alpha"]
                    phase_loss_val = criterion(
                        output_val_split[..., 1],
                        gt_val[:, cfg["ch_num"] :, split_id * cfg["pixel_count"] : (split_id + 1) * cfg["pixel_count"]],
                    ) * cfg["phase_alpha"]
                    loss_val = mag_loss_val + phase_loss_val
                    total_mag_val += float(mag_loss_val.detach())
                    total_phase_val += float(phase_loss_val.detach())
                    total_loss_val += float(loss_val.detach())
                    val_iter += 1
                    output_val_list.append(output_val_split)

                output_val = torch.cat(output_val_list, dim=2)

                # reconstruct specs
                myout = output_val.cpu().numpy()
                myout_mag = myout[..., 0].reshape(1, cfg["ch_num"], dataset.sound_size[1], dataset.sound_size[2])
                myout_phase = myout[..., 1].reshape(1, cfg["ch_num"], dataset.sound_size[1], dataset.sound_size[2])
                mygt = gt_val.cpu().numpy()
                mygt_mag = mygt[:, : cfg["ch_num"]].reshape(1, cfg["ch_num"], dataset.sound_size[1], dataset.sound_size[2])
                mygt_phase = mygt[:, cfg["ch_num"] :].reshape(1, cfg["ch_num"], dataset.sound_size[1], dataset.sound_size[2])

                net_mag = (myout_mag * dataset.std.numpy() + dataset.mean.numpy())[0]
                gt_mag = (mygt_mag * dataset.std.numpy() + dataset.mean.numpy())[0]
                net_phase = myout_phase[0] * dataset.phase_std
                gt_phase = mygt_phase[0] * dataset.phase_std

                net_spec = get_spectrograms(net_mag, net_phase, cfg["log_eps"])
                gt_spec = get_spectrograms(gt_mag, gt_phase, cfg["log_eps"])

                ir_pred = istft_ir(net_spec, nfft=cfg["n_fft"], hop=cfg["hop_size"], win=cfg["window"])
                ir_gt = istft_ir(gt_spec, nfft=cfg["n_fft"], hop=cfg["hop_size"], win=cfg["window"])

                ir_pred_list.append((base_key, ch_idx, ir_pred))
                ir_gt_list.append((base_key, ch_idx, ir_gt))

                position_tx_list.append((base_key, ch_idx, pos_tx.astype(np.float32)))
                position_rx_list.append((base_key, ch_idx, pos_rx.astype(np.float32)))

        if rank == 0:
            if cfg["model_type"] == "NAF":
                grouped = {}
                for base_key, ch_idx, arr in ir_pred_list:
                    grouped.setdefault(base_key, {})[ch_idx] = arr
                grouped_gt = {}
                for base_key, ch_idx, arr in ir_gt_list:
                    grouped_gt.setdefault(base_key, {})[ch_idx] = arr
                grouped_tx = {}
                for base_key, ch_idx, arr in position_tx_list:
                    grouped_tx.setdefault(base_key, arr)
                grouped_rx = {}
                for base_key, ch_idx, arr in position_rx_list:
                    grouped_rx.setdefault(base_key, {})[ch_idx] = arr

                base_keys = sorted(grouped.keys())
                ir_pred_arr = []
                ir_gt_arr = []
                pos_tx_arr = []
                pos_rx_arr = []
                for bk in base_keys:
                    ch_dict = grouped[bk]
                    ch_order = sorted(ch_dict.keys())
                    ir_pred_arr.append(np.stack([ch_dict[c] for c in ch_order], axis=0))
                    ir_gt_arr.append(np.stack([grouped_gt[bk][c] for c in ch_order], axis=0))
                    pos_tx_arr.append(grouped_tx[bk])
                    pos_rx_arr.append(np.stack([grouped_rx[bk][c] for c in ch_order], axis=0))

                ir_pred_arr = np.stack(ir_pred_arr, axis=0)
                ir_gt_arr = np.stack(ir_gt_arr, axis=0)
                pos_tx_arr = np.stack(pos_tx_arr, axis=0)
                pos_rx_arr = np.stack(pos_rx_arr, axis=0)
            else:
                ir_pred_arr = np.stack([x[2] for x in ir_pred_list], axis=0)
                ir_gt_arr = np.stack([x[2] for x in ir_gt_list], axis=0)
                pos_tx_arr = np.stack([x[2] for x in position_tx_list], axis=0)
                pos_rx_arr = np.stack([x[2] for x in position_rx_list], axis=0)

            metrics = ir_metrics(ir_gt_arr, ir_pred_arr, fs=cfg["fs"])
            doa_gt_deg, doa_pred_deg, doa_true_deg = doa_metrics(
                ir_gt_arr,
                ir_pred_arr,
                algorithm=cfg["doa_algorithm"],
                n_fft=cfg["n_fft"],
                fs=cfg["fs"],
                rx_positions=pos_rx_arr,
                tx_positions=pos_tx_arr,
                hop_size=cfg["hop_size"],
                window=cfg["window"],
            )

            doa_err = float(np.mean(angular_error_deg(doa_pred_deg, doa_gt_deg)))

            # save val results
            val_path = os.path.join(val_dir, f"epoch{epoch:04d}.npz")
            np.savez(
                val_path,
                position_tx=pos_tx_arr,
                position_rx=pos_rx_arr,
                ir_gt=ir_gt_arr,
                ir_pred=ir_pred_arr,
                doa_true_deg=doa_true_deg,
                doa_gt_deg=doa_gt_deg,
                doa_pred_deg=doa_pred_deg,
                metric_angle=metrics["angle_error"],
                metric_amp=metrics["amp_error"],
                metric_env_pct=metrics["env_error_percent"],
                metric_t60_pct=metrics["t60_error_percent"],
                metric_c50_db=metrics["c50_error_db"],
                metric_edt_ms=metrics["edt_error_ms"],
            )

            # loss log
            loss_path = os.path.join(loss_dir, f"epoch{epoch:04d}.npz")
            np.savez(
                loss_path,
                epoch=np.int32(epoch),
                loss_train=np.float32(total_loss / max(iter_count, 1)),
                mag_train=np.float32(total_mag / max(iter_count, 1)),
                phase_train=np.float32(total_phase / max(iter_count, 1)),
                loss_val=np.float32(total_loss_val / max(val_iter, 1)),
                mag_val=np.float32(total_mag_val / max(val_iter, 1)),
                phase_val=np.float32(total_phase_val / max(val_iter, 1)),
                doa_err_val=np.float32(doa_err),
                doa_err=np.float32(doa_err),
            )

            print(
                f"Epoch {epoch}: loss={total_loss/max(iter_count,1):.5f}, "
                f"DoA_err={doa_err:.5f}, time={time()-start_time:.1f}s"
            )

            # Save candidate checkpoint, then reorder/move by score.
            candidate_path = Path(ckpt_dir) / f"eval{epoch:04d}.ckpt"
            torch.save(
                {
                    "network": ddp_net.module.state_dict(),
                    "mean": dataset.mean.numpy(),
                    "std": dataset.std.numpy(),
                    "phase_std": dataset.phase_std,
                    "max_len": dataset.max_len,
                    "sound_size": dataset.sound_size,
                    "n_fft": cfg["n_fft"],
                    "hop_size": cfg["hop_size"],
                    "window": cfg["window"],
                    "log_eps": cfg["log_eps"],
                    "model_type": cfg["model_type"],
                    "ch_num": cfg["ch_num"],
                    "epoch": epoch,
                },
                candidate_path,
            )

            best_scores.append((doa_err, candidate_path))
            best_scores = sorted(best_scores, key=lambda x: x[0])[: cfg["save_best_k"]]

            keep_eval = {p for _, p in best_scores}
            for p in Path(ckpt_dir).glob("eval*.ckpt"):
                if p not in keep_eval:
                    p.unlink()

            for order_idx in range(len(best_scores), 0, -1):
                _, src = best_scores[order_idx - 1]
                final_path = Path(ckpt_dir) / f"best{order_idx:04d}.ckpt"
                if src == final_path:
                    continue
                if final_path.exists():
                    final_path.unlink()
                if src.exists():
                    src.rename(final_path)

            best_scores = [
                (score, Path(ckpt_dir) / f"best{order_idx:04d}.ckpt")
                for order_idx, (score, _) in enumerate(best_scores, start=1)
            ]

        dist.barrier()

    dist.destroy_process_group()


def run_training(config_path: str, data_dir: str, output_dir: str):
    train_cfg = load_yaml(Path(config_path))
    preprocess_cfg = load_yaml(Path(data_dir) / "preprocess_config.yml")

    setting = train_cfg.get("setting", {})
    param = train_cfg.get("param", {})
    doa_metric = train_cfg.get("doa_metric", {})

    model_type = setting.get("model_type", "NAF+")
    ch_num = 1 if model_type == "NAF" else int(setting.get("ch_num", 1))

    cfg = {
        "output_dir": output_dir,
        "gpus": int(setting.get("gpus", 1)),
        "epochs": int(setting.get("epochs", 200)),
        "resume": bool(setting.get("resume", False)),
        "batch_size": int(setting.get("batch_size", 20)),
        "save_best_k": int(setting.get("save_best_k", 10)),
        "model_type": model_type,
        "ch_num": ch_num,
        "mag_alpha": float(param.get("mag_alpha", 1.0)),
        "phase_alpha": float(param.get("phase_alpha", 1.0)),
        "lr_init": float(param.get("lr_init", 1.0e-3)),
        "lr_decay": float(param.get("lr_decay", 1.0e-1)),
        "weight_decay_grid": float(param.get("weight_decay_grid", 1.0e-2)),
        "weight_decay_main": float(param.get("weight_decay_main", 0.0)),
        "reg_eps": float(param.get("reg_eps", 0.05)),
        "pixel_count": int(param.get("pixel_count", 2000)),
        "layers": int(param.get("layers", 8)),
        "layers_residual": int(param.get("layers_residual", 1)),
        "features": int(param.get("features", 256)),
        "grid_features": int(param.get("grid_features", 64)),
        "activation_func_name": str(param.get("activation_func_name", "default")),
        "grid_gap": float(param.get("grid_gap", 0.25)),
        "bandwith_init": float(param.get("bandwith_init", 0.25)),
        "position_float": float(param.get("position_float", 0.1)),
        "min_bandwidth": float(param.get("min_bandwidth", 0.1)),
        "max_bandwidth": float(param.get("max_bandwidth", 0.5)),
        "embed_xyz_num_freqs": int(param.get("embed_xyz_num_freqs", 10)),
        "embed_xyz_max_freq": float(param.get("embed_xyz_max_freq", 7)),
        "embed_time_num_freqs": int(param.get("embed_time_num_freqs", 10)),
        "embed_time_max_freq": float(param.get("embed_time_max_freq", 10)),
        "embed_freq_num_freqs": int(param.get("embed_freq_num_freqs", 10)),
        "embed_freq_max_freq": float(param.get("embed_freq_max_freq", 10)),
        "xy_min": param.get("xy_min"),
        "xy_max": param.get("xy_max"),
        "doa_algorithm": str(doa_metric.get("algorithm", "NormMUSIC")),
        "fs": int(preprocess_cfg.get("fs", 16000)),
        "n_fft": int(preprocess_cfg.get("n_fft", 512)),
        "hop_size": int(preprocess_cfg.get("hop_size", 128)),
        "window": str(preprocess_cfg.get("window", "hann")),
        "log_eps": float(preprocess_cfg.get("log_eps", 1e-3)),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_config.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(train_cfg, f, allow_unicode=True, sort_keys=False)

    dataset_cfg = DatasetConfig(
        data_dir=data_dir,
        pixel_count=cfg["pixel_count"],
        reg_eps=cfg["reg_eps"],
        model_type=cfg["model_type"],
        ch_num=cfg["ch_num"],
        xy_min=cfg.get("xy_min"),
        xy_max=cfg.get("xy_max"),
    )
    cfg["dataset_cfg"] = dataset_cfg

    gpus = cfg["gpus"]
    freeport = str(find_free_port())
    mp.spawn(train_worker, args=(gpus, freeport, cfg), nprocs=gpus, join=True)


def main():
    parser = argparse.ArgumentParser(description="NAF training")
    parser.add_argument("--config", required=True, help="Path to train_config.yml")
    parser.add_argument("--data_dir", required=True, help="Preprocessed data directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()
    run_training(args.config, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
