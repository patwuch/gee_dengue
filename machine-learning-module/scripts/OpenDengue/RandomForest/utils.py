"""
scripts/OpenDengue/RandomForest/utils.py
------------------------------------------
Re-exports all shared helpers from the XGBoost utils module.

The Random Forest and XGBoost experiments share the same data loading,
feature-building, split logic, and path resolution — no duplication needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the XGBoost utils importable from this directory
sys.path.insert(0, str(Path(__file__).parent.parent / "XGBoost"))

from utils import (  # noqa: F401  (re-export)
    build_feature_columns,
    calculate_sample_weights,
    load_config,
    load_data,
    resolve_paths,
    split_train_test,
    validation_strategy,
)
