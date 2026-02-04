import argparse
import os
from pathlib import Path
import yaml
import optuna
import numpy as np


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def build_trial_config(base_cfg: dict, trial: optuna.trial.Trial) -> dict:
    setting = dict(base_cfg.get("setting", {}))
    fixed = dict(base_cfg.get("fixed", {}))
    search = base_cfg.get("search_space", {})

    param = dict(fixed)

    for key, spec in search.items():
        stype = spec.get("type")
        if stype == "int":
            val = trial.suggest_int(key, int(spec["low"]), int(spec["high"]))
        elif stype == "float":
            val = trial.suggest_float(
                key,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif stype == "categorical":
            val = trial.suggest_categorical(key, spec["choices"])
        else:
            raise ValueError(f"Unknown search space type: {stype}")
        param[key] = val

    train_cfg = {
        "setting": setting,
        "param": param,
        "doa_metric": base_cfg.get("doa_metric", {}),
    }

    return train_cfg


def run_training(train_cfg_path: Path, data_dir: Path, output_dir: Path) -> int:
    from train import run_training as run_training_impl

    run_training_impl(str(train_cfg_path), str(data_dir), str(output_dir))
    return 0


def load_best_doa(val_dir: Path, fallback: float) -> float:
    if not val_dir.exists():
        return fallback
    npzs = sorted(val_dir.glob("epoch*.npz"))
    if not npzs:
        return fallback
    last = npzs[-1]
    data = np.load(last)
    if "doa_pred_deg" not in data or "doa_gt_deg" not in data:
        return fallback
    pred = data["doa_pred_deg"]
    gt = data["doa_gt_deg"]
    err = np.abs(pred - gt)
    err = np.minimum(err, 360.0 - err)
    return float(np.mean(err))


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for NAF")
    parser.add_argument("--config", required=True, help="Path to optuna_config.yml")
    parser.add_argument("--data_dir", required=True, help="Preprocessed data dir")
    parser.add_argument("--output_dir", required=True, help="Optuna output dir")
    args = parser.parse_args()

    base_cfg = load_yaml(Path(args.config))
    study_cfg = base_cfg.get("study", {})
    study_name = study_cfg.get("study_name", "naf_optuna")
    direction = study_cfg.get("direction", "minimize")
    n_trials = int(study_cfg.get("n_trials", 50))

    fallback = float(base_cfg.get("doa_metric", {}).get("fallback_value", 999.0))

    out_root = Path(args.output_dir) / study_name
    out_root.mkdir(parents=True, exist_ok=True)
    write_yaml(out_root / "optuna_config.yml", base_cfg)

    storage_path = f"sqlite:///{out_root / (study_name + '.db')}"
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage_path,
        load_if_exists=True,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        trial_cfg = build_trial_config(base_cfg, trial)
        trial_dir = out_root / "trials" / f"trial{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        train_cfg_path = trial_dir / "train_config.yml"
        write_yaml(train_cfg_path, trial_cfg)

        ret = run_training(train_cfg_path, Path(args.data_dir), trial_dir)
        if ret != 0:
            return fallback

        val_dir = trial_dir / "val_results"
        return load_best_doa(val_dir, fallback)

    study.optimize(objective, n_trials=n_trials)

    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
    print("Study name:", study.study_name)


if __name__ == "__main__":
    main()
