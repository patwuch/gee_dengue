"""
Run STGNN inference on delta-corrected zonal statistics.

Loads the production model (trained on full 2011-2018 data) and runs a single
forward pass for the current month. Predictions are expressed as a relative
risk index (exp(predicted_log_IR) / exp(training_period_mean_log_IR)) rather
than raw case counts, since the training/deployment gap makes absolute
predictions unreliable.

Inputs:
    site/data/delta_corrected_stats.json
    machine-learning-module/results/STGNN/production_logIR/best_model.pt
    machine-learning-module/results/STGNN/baseline_logIR/best_params.json
    machine-learning-module/data/processed/STGNN/production_logIR/scaler.pkl
    machine-learning-module/data/processed/STGNN/production_logIR/edge_index.pt

Outputs:
    site/data/latest_predictions.json
"""

import json
import pickle
import pathlib
import datetime
import numpy as np
import torch
import sys

ROOT = pathlib.Path(__file__).parent.parent.parent
ML_ROOT = ROOT / "machine-learning-module"

# Allow importing STGNN model definition
sys.path.insert(0, str(ML_ROOT / "src/OpenDengue/STGNN"))
from model import STGATGRU  # noqa: E402

CORRECTED_STATS  = pathlib.Path(__file__).parent.parent / "data/delta_corrected_stats.json"
BEST_PARAMS      = ML_ROOT / "results/STGNN/baseline_logIR/best_params.json"
MODEL_PATH       = ML_ROOT / "results/STGNN/production_logIR/best_model.pt"
SCALER_PATH      = ML_ROOT / "data/processed/STGNN/production_logIR/scaler.pkl"
EDGE_INDEX_PATH  = ML_ROOT / "data/processed/STGNN/production_logIR/edge_index.pt"
TRAINING_CSV     = ML_ROOT / "data/interim/SEA_dengue_env_monthly_2011-2018.csv"
DATA_DIR         = pathlib.Path(__file__).parent.parent / "data"
OUTPUT_PATH      = DATA_DIR / "latest_predictions.json"
ARCHIVE_DIR      = DATA_DIR / "predictions"

ENV_FEATURES = [
    "precipitation_sum",
    "temperature_2m_mean",
    "temperature_2m_max_mean",
    "temperature_2m_min_mean",
    "potential_evaporation_sum_mean",
    "total_evaporation_sum_mean",
    "evaporation_from_bare_soil_sum_mean",
    "evaporation_from_open_water_surfaces_excluding_oceans_sum_mean",
    "evaporation_from_the_top_of_canopy_sum_mean",
    "evaporation_from_vegetation_transpiration_sum_mean",
    "LST_Day_1km_mean",
    "LST_Night_1km_mean",
    "NDVI_mean",
    "EVI_mean",
]

LULC_FEATURES = [f"LC_Type1_pct_class{i}" for i in range(1, 18)]
ALL_FEATURES  = ENV_FEATURES + LULC_FEATURES


def load_model(params: dict, n_features: int, device: torch.device) -> STGATGRU:
    model = STGATGRU(
        in_channels  = n_features,
        gat1_hidden  = params["gat1_hidden"],
        gat1_heads   = params["gat1_heads"],
        mlp_hidden   = params["mlp_hidden"],
        mlp_layers   = params["mlp_layers"],
        gat2_hidden  = params["gat2_hidden"],
        gat2_heads   = params["gat2_heads"],
        gru_hidden   = params.get("gru_hidden", params["gat2_hidden"]),
        pred_horizon = 1,
        dropout      = params.get("dropout", 0.0),
    )
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def build_feature_tensor(
    regions: dict,
    region_order: list[str],
    scalers: dict,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build (1, window_size, N, F) feature tensor and (1, N) mask from region stats.
    For window_size > 1 we repeat the single current month (conservative — avoids
    needing historical GEE accumulation for v1).
    """
    N = len(region_order)
    F = len(ALL_FEATURES)
    x = np.zeros((N, F), dtype=np.float32)
    mask = np.zeros(N, dtype=np.float32)

    env_scaler  = scalers.get("env")
    lulc_scaler = scalers.get("lulc")

    for i, region in enumerate(region_order):
        vals = regions.get(region, {})
        if not vals:
            continue
        mask[i] = 1.0

        env_raw = np.array(
            [vals.get(f, 0.0) or 0.0 for f in ENV_FEATURES],
            dtype=np.float32,
        )
        lulc_raw = np.array(
            [vals.get(f, 0.0) or 0.0 for f in LULC_FEATURES],
            dtype=np.float32,
        )

        if env_scaler is not None:
            env_scaled = env_scaler.transform(env_raw.reshape(1, -1))[0]
        else:
            env_scaled = env_raw

        if lulc_scaler is not None:
            lulc_scaled = lulc_scaler.transform(lulc_raw.reshape(1, -1))[0]
        else:
            lulc_scaled = lulc_raw

        x[i] = np.concatenate([env_scaled, lulc_scaled])

    # Repeat across window dimension — conservative fallback for v1
    x_seq  = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,N,F)
    x_seq  = x_seq.expand(1, window_size, N, F).clone()
    mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)             # (1, N)

    return x_seq, mask_t


def training_mean_log_ir() -> float:
    """Compute the grand-mean log IR over the training set for risk index normalisation."""
    import pandas as pd
    df = pd.read_csv(TRAINING_CSV)
    ir = df["IR"].dropna()
    ir = ir[ir > 0]
    return float(np.log1p(ir).mean())


def main() -> None:
    for p in (CORRECTED_STATS, BEST_PARAMS, MODEL_PATH, SCALER_PATH, EDGE_INDEX_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Ensure the production model has been trained and zonal stats fetched."
            )

    device = torch.device("cpu")  # inference on CI runner, no GPU needed

    stats  = json.loads(CORRECTED_STATS.read_text())
    params = json.loads(BEST_PARAMS.read_text())

    with open(SCALER_PATH, "rb") as f:
        scalers = pickle.load(f)

    edge_index = torch.load(EDGE_INDEX_PATH, weights_only=True).to(device)

    regions      = stats.get("regions", {})
    region_order = sorted(regions.keys())
    n_features   = len(ALL_FEATURES)
    window_size  = int(params.get("window_size", 1))

    print(f"Loading production model from {MODEL_PATH} ...")
    model = load_model(params, n_features, device)

    print(f"Building feature tensor for {len(region_order)} regions (window={window_size}) ...")
    x, mask = build_feature_tensor(regions, region_order, scalers, window_size)
    x, mask = x.to(device), mask.to(device)

    print("Running forward pass ...")
    with torch.no_grad():
        pred = model(x, edge_index, mask=mask)  # (1, N, 1)
    pred_log_ir = pred.squeeze().cpu().numpy()   # (N,)

    baseline_mean = training_mean_log_ir()
    risk_index    = np.exp(pred_log_ir - baseline_mean)  # >1 = elevated risk

    target_month = stats.get("target_month", datetime.datetime.now().strftime("%Y-%m"))

    output = {
        "generated_at":  datetime.datetime.utcnow().isoformat() + "Z",
        "target_month":  target_month,
        "bias_corrected": stats.get("bias_corrected", False),
        "baseline_mean_log_ir": round(baseline_mean, 4),
        "regions": {
            region: {
                "country":     regions[region].get("country", ""),
                "risk_index":  round(float(risk_index[i]), 4),
                "pred_log_ir": round(float(pred_log_ir[i]), 4),
                "has_data":    bool(mask[0, i].item()),
            }
            for i, region in enumerate(region_order)
        },
    }

    payload = json.dumps(output, indent=2)

    # Always update the "latest" pointer
    OUTPUT_PATH.write_text(payload)

    # Archive a permanent per-month copy — never overwritten
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = ARCHIVE_DIR / f"{target_month}.json"
    archive_path.write_text(payload)

    print(f"\nPredictions saved → {OUTPUT_PATH}")
    print(f"Archived         → {archive_path}")
    print(f"Regions above baseline (risk_index > 1): "
          f"{sum(1 for r in output['regions'].values() if r['risk_index'] > 1)}/{len(region_order)}")


if __name__ == "__main__":
    main()
