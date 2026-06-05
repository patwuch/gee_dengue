from __future__ import annotations

from pathlib import Path

import pickle
import yaml
import torch
from sklearn.preprocessing import StandardScaler
import json
import numpy as np

def _output_dir(cfg: dict) -> Path:
    return Path("data/processed/STGNN") / cfg["name"]


def save_scaler(scalers: dict, cfg: dict) -> None:
    path = _output_dir(cfg) / "scaler.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scalers, f)

def load_scaler(cfg: dict) -> StandardScaler:
    path = _output_dir(cfg) / "scalers" / "metadata.json"
    with open(path) as f:
        metadata = json.load(f)
    scaler        = StandardScaler()
    scaler.mean_  = np.array(metadata["scaler"]["mean"])
    scaler.scale_ = np.array(metadata["scaler"]["std"])
    return scaler


def load_config(path: str) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_tensors(snapshots: dict, cfg: dict, window_size: int) -> None:
    out_dir = _output_dir(cfg) / f"window_{window_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split, windows in snapshots.items():
        splits[f"{split}_x"]    = torch.stack([x    for x, y, mask in windows])
        splits[f"{split}_y"]    = torch.stack([y    for x, y, mask in windows])
        splits[f"{split}_mask"] = torch.stack([mask for x, y, mask in windows])

    torch.save(splits, out_dir / "tensors.pt")

def save_edge_index(edge_index: torch.Tensor, cfg: dict) -> None:
    """Persist edge_index to disk."""
    out_dir = _output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(edge_index, out_dir / "edge_index.pt")


def get_window_sizes(cfg: dict) -> list[int]:
    """Extract window size choices from tuning search space config."""
    return (
        cfg.get("tune", {})
           .get("search_space", {})
           .get("stgnn", {})
           .get("window_size", {})
           .get("choices", [0])
    )