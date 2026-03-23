import json
import os
from dataclasses import asdict, dataclass, fields, replace

import yaml


@dataclass
class TrainConfig:
    target: str
    out_dir: str = "./runs/default"
    n_strokes: int = 72
    width: int = 224
    steps: int = 260
    iters: int = 650
    lr: float = 0.05
    seed: int = 0
    cpu: bool = False
    batch: int = 1
    w_clip: float = 1.0
    w_sparse: float = 0.08
    w_ink: float = 0.10
    w_l2: float = 0.15
    w_edge: float = 0.22
    save_every: int = 50
    tau_active: float = 0.5
    tau_white: float = 0.95


def _load_raw_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def load_train_config(config_path: str, **overrides) -> TrainConfig:
    cfg_data = _load_raw_config(config_path) or {}
    for key, value in overrides.items():
        if value is not None:
            cfg_data[key] = value
    if "target" in cfg_data and cfg_data["target"] is not None:
        cfg_data["target"] = str(cfg_data["target"])
    if "out_dir" in cfg_data and cfg_data["out_dir"] is not None:
        cfg_data["out_dir"] = str(cfg_data["out_dir"])
    return TrainConfig(**cfg_data)


def apply_overrides(cfg: TrainConfig, **overrides) -> TrainConfig:
    valid_keys = {f.name for f in fields(TrainConfig)}
    filtered = {k: v for k, v in overrides.items() if k in valid_keys and v is not None}
    if len(filtered) == 0:
        return cfg
    return replace(cfg, **filtered)


def config_to_dict(cfg: TrainConfig):
    return asdict(cfg)


def default_config_path(project_root: str) -> str:
    return os.path.join(project_root, "configs", "main_weighted.yaml")
