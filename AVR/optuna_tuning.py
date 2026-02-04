import argparse
from pathlib import Path
import yaml
import optuna
import numpy as np


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def set_nested(cfg: dict, key: str, value):
    if key.startswith("param."):
        target = cfg.setdefault("param", {})
        path = key.split(".")[1:]
    elif key.startswith("model."):
        target = cfg.setdefault("model", {})
        path = key.split(".")[1:]
    elif key.startswith("setting."):
        target = cfg.setdefault("setting", {})
        path = key.split(".")[1:]
    else:
        target = cfg.setdefault("param", {})
        path = [key]

    cur = target
    for p in path[:-1]:
        cur = cur.setdefault(p, {})
    cur[path[-1]] = value


def build_trial_config(base_cfg: dict, trial: optuna.trial.Trial) -> dict:
    trial_cfg = {
        "setting": dict(base_cfg.get("setting", {})),
        "param": dict(base_cfg.get("param", {})),
        "model": dict(base_cfg.get("model", {})),
        "doa_metric": dict(base_cfg.get("doa_metric", {})),
    }

    fixed = base_cfg.get("fixed", {})
    for key, val in fixed.items():
        set_nested(trial_cfg, key, val)

    search = base_cfg.get("search_space", {})
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
        set_nested(trial_cfg, key, val)

    return trial_cfg


def load_best_doa(val_dir: Path, fallback: float) -> float:
    if not val_dir.exists():
        return fallback
    npzs = sorted(val_dir.glob("epoch*.npz"))
    if not npzs:
        return fallback
    last = npzs[-1]
    data = np.load(last)
    if "doa_pred_deg" not in data or "doa_true_deg" not in data:
        return fallback
    pred = data["doa_pred_deg"]
    true = data["doa_true_deg"]
    err = np.abs(pred - true)
    err = np.minimum(err, 360.0 - err)
    if np.isnan(err).any():
        return fallback
    return float(np.mean(err))


def run_training(train_cfg: dict, data_dir: Path, output_dir: Path):
    from train import run_training as run_training_impl

    run_training_impl(train_cfg, str(data_dir), str(output_dir))


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for AVR")
    parser.add_argument("--config", required=True, help="Path to optuna_config.yml")
    parser.add_argument("--data_dir", required=True, help="Dataset directory")
    parser.add_argument("--output_dir", required=True, help="Optuna output directory")
    args = parser.parse_args()

    base_cfg = load_yaml(Path(args.config))
    study_cfg = base_cfg.get("study", {})
    study_name = study_cfg.get("study_name", "avr_optuna")
    direction = study_cfg.get("direction", "minimize")
    n_trials = int(study_cfg.get("n_trials", 50))

    fallback = float(base_cfg.get("doa_metric", {}).get("fallback_value", 999.0))

    out_root = Path(args.output_dir) / study_name
    out_root.mkdir(parents=True, exist_ok=True)
    save_yaml(out_root / "optuna_config.yml", base_cfg)

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
        save_yaml(train_cfg_path, trial_cfg)

        run_training(trial_cfg, Path(args.data_dir), trial_dir)
        return load_best_doa(trial_dir / "val_results", fallback)

    study.optimize(objective, n_trials=n_trials)

    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
    print("Study name:", study.study_name)


if __name__ == "__main__":
    main()
