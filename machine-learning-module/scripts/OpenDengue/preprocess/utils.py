from __future__ import annotations

from pathlib import Path

import yaml
from sklearn.preprocessing import StandardScaler
import json
import numpy as np

def save_scaler(scaler, cfg: dict) -> None:
    path = Path(cfg["output_path"]) / "artifacts" / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "scaler": {
            "mean":  scaler.mean_.tolist(),    # .tolist() converts np array → plain python
            "std":   scaler.scale_.tolist()
        }
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

def load_scaler(cfg: dict) -> StandardScaler:
    path = Path(cfg["output_path"]) / "artifacts" / "metadata.json"
    with open(path) as f:
        metadata = json.load(f)
    
    scaler = StandardScaler()
    scaler.mean_   = np.array(metadata["scaler"]["mean"])
    scaler.scale_  = np.array(metadata["scaler"]["std"])
    return scaler


def load_config(path: str) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_tensors(train, val, test, cfg: dict) -> None:
    """Persist processed splits to disk."""
    # TODO: save train/val/test tensors to cfg["output_path"]
    pass