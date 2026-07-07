# test.py
"""
Evaluate a trained STGATGRU model on the test split (or any split).

Inputs:
    config (YAML)        — defines name, device, data paths, log_scale, ...
    model checkpoint     — the trained .pt to evaluate
    best_params.json     — hyperparameters used to *build* the model
                           (required to reconstruct the architecture before
                           loading the state dict)

Outputs (results/STGNN/<name>/):
    test_predictions.npz — stacked predictions + targets (IR scale)
    predictions.png      — per-node pred vs true over windows
    scatter.png          — pred vs true scatter
    metrics.json         — test (huber/mae/mse/rmse) [+ best_val_loss if available]

Run via Snakemake (script mode) or directly:
    python test.py --config config.yaml --model best_model.pt [--params best_params.json] [--split test]
"""

import json
import yaml
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from model import STGATGRU
from loss  import masked_huber_loss
from torch.utils.data import DataLoader
from tune  import STGNNDataset, load_tensors, load_edge_index
from preprocess.utils import load_preprocessing_params


# ── Paths / params ────────────────────────────────────────────────────────────

def _results_dir(cfg: dict) -> Path:
    return Path("results/STGNN") / cfg["name"]


def _best_params_path(cfg: dict) -> Path:
    return _results_dir(cfg) / "best_params.json"


def load_best_params(cfg: dict, override_path: str = None) -> dict:
    path = Path(override_path) if override_path else _best_params_path(cfg)
    with open(path) as f:
        return json.load(f)


# ── Inverse transform ─────────────────────────────────────────────────────────
def inverse_transform(
    preds:   np.ndarray,        # (W, N) in model output space
    targets: np.ndarray,        # (W, N) in model output space
    params:  dict,              # from load_preprocessing_params, or {} if unavailable
    months:  np.ndarray | None, # (W,) month integers 1-12, or None
    cfg:     dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert preprocessing transforms in reverse order: scaler → seasonal mean → log."""
    prep = cfg.get("preprocessing", {})
    p, t = preds.copy(), targets.copy()

    if prep.get("scale_target", True):
        si = params.get("scaler_inc")
        if si:
            p = p * si["std"] + si["mean"]
            t = t * si["std"] + si["mean"]

    if prep.get("deseasonalise") and months is not None:
        sm = {int(k): v for k, v in params.get("seasonal_means", {}).items()}
        if sm:
            offsets = np.array([sm.get(int(m), 0.0) for m in months])  # (W,)
            p = p + offsets[:, None]   # broadcast over nodes
            t = t + offsets[:, None]

    if prep.get("log_transform"):
        p = np.expm1(p)
        t = np.expm1(t)

    return p, t


# ── Evaluation pass ───────────────────────────────────────────────────────────
def eval_with_metrics(
    model:      STGATGRU,
    dataloader: DataLoader,
    edge_index: torch.Tensor,
    device:     torch.device,
) -> tuple[float, list, list, list]:
    """Collect predictions and targets in model output space (no inverse transform)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_masks = [], [], []

    with torch.no_grad():
        for x, y, mask in dataloader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            pred = model(x, edge_index, mask=mask)
            if pred.dim() == 3:
                pred = pred.squeeze(-1)

            target_mask = mask[:, -1, :] if mask.dim() == 3 else mask

            loss = masked_huber_loss(pred, y, target_mask)
            total_loss += loss.item()

            if y.dim() == 3:
                y = y.squeeze(-1)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_masks.append(target_mask.cpu().numpy())

    mean_loss = total_loss / max(len(dataloader), 1)
    return mean_loss, all_preds, all_targets, all_masks

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_predictions(
    all_preds:   list,
    all_targets: list,
    out_dir:     Path,
    n_nodes:     int = 6,
) -> Path:
    """Plot pred vs target IR for the first n_nodes nodes across all windows."""
    preds   = np.concatenate(all_preds,   axis=0)  # (W, N)
    targets = np.concatenate(all_targets, axis=0)  # (W, N)
    n_show  = min(n_nodes, preds.shape[1])

    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.5 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(targets[:, i], label="True IR",  linewidth=1.2, alpha=0.8)
        ax.plot(preds[:, i],   label="Pred IR",  linewidth=1.2, alpha=0.8, linestyle="--")
        ax.set_ylabel(f"Node {i}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Window")
    fig.suptitle("Predictions vs targets (IR scale)", y=1.01)
    fig.tight_layout()
    path = out_dir / "predictions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scatter(
    all_preds:   list,
    all_targets: list,
    out_dir:     Path,
) -> Path:
    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(targets, preds, alpha=0.3, s=8, linewidths=0)
    lim = max(targets.max(), preds.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="Perfect")
    ax.set_xlabel("True IR")
    ax.set_ylabel("Predicted IR")
    ax.set_title("Pred vs True (test set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Test / evaluation ─────────────────────────────────────────────────────────

def test(cfg: dict, params: dict, model_path, split: str = "test"):
    """Load a trained model and evaluate it on the given split."""
    model_path = Path(model_path)
    out_dir    = _results_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    device      = torch.device(cfg.get("device", "cpu"))
    window_size = params["window_size"]

    # ── Data ─────────────────────────────────────────────────────────────────
    tensors = load_tensors(cfg, window_size)

    # Prefer the inference split when available — it covers all test months
    # by borrowing context from the train/val tail (see preprocess/temporal.py).
    # Fall back to the regular test split for backwards compatibility.
    if split == "test" and "inference_x" in tensors:
        split = "inference"
        print("INFO: using inference split — predictions cover all test months.")

    dataset     = STGNNDataset(tensors, split)
    dataloader  = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)
    edge_index  = load_edge_index(cfg, device)
    in_channels = dataset.x.shape[-1]

    # ── Preprocessing params for inverse transform ────────────────────────────
    try:
        preproc_params = load_preprocessing_params(cfg)
    except FileNotFoundError:
        print("WARNING: preprocessing_params.json not found — metrics will be in model output space.")
        preproc_params = {}
    months    = tensors.get(f"{split}_months")
    months_np = months.numpy() if months is not None else None

    # ── Rebuild model & load weights ─────────────────────────────────────────
    model = STGATGRU(
        in_channels  = in_channels,
        gat1_hidden  = params["gat1_hidden"],
        gat1_heads   = params["gat1_heads"],
        mlp_hidden   = params.get("mlp_hidden", params["gat1_hidden"] * params["gat1_heads"]),
        mlp_layers   = params["mlp_layers"],
        gat2_hidden  = params["gat2_hidden"],
        gat2_heads   = params["gat2_heads"],
        gru_hidden   = params["gru_hidden"],
        pred_horizon = 1,
        dropout      = params["dropout"],
    ).to(device)

    print(f"Loading model from {model_path} ...")
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # ── Evaluate ──────────────────────────────────────────────────────────────
    loss, raw_preds, raw_targets, all_masks = eval_with_metrics(
        model, dataloader, edge_index, device
    )

    # ── Inverse transform to IR scale ─────────────────────────────────────────
    preds   = np.concatenate(raw_preds,   axis=0)  # (W, N)
    targets = np.concatenate(raw_targets, axis=0)  # (W, N)
    masks   = np.concatenate(all_masks,   axis=0).astype(bool)

    preds, targets = inverse_transform(preds, targets, preproc_params, months_np, cfg)

    # ── Metrics on valid positions only ───────────────────────────────────────
    valid_preds   = preds[masks]
    valid_targets = targets[masks]
    mae     = float(np.abs(valid_preds - valid_targets).mean())
    mse     = float(((valid_preds - valid_targets) ** 2).mean())
    metrics = {"mae": mae, "mse": mse, "rmse": mse ** 0.5}

    print(
        f"\n{split} results | Huber {loss:.4f} | "
        f"MAE {metrics['mae']:.4f} | RMSE {metrics['rmse']:.4f}"
    )

    # ── Save predictions ──────────────────────────────────────────────────────
    # Always written as test_predictions.npz so Snakemake output paths are stable
    # regardless of whether the inference or test split was used internally.
    preds_path = out_dir / "test_predictions.npz"
    np.savez(preds_path, predictions=preds, targets=targets)
    print(f"Predictions saved to {preds_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    pred_plot    = plot_predictions([preds],   [targets], out_dir)
    scatter_plot = plot_scatter(    [preds],   [targets], out_dir)

    # ── Metrics summary ───────────────────────────────────────────────────────
    # Keys are always prefixed "test_" so metrics.json is stable across splits.
    summary = {
        "test_huber": loss,
        "test_mae":   metrics["mae"],
        "test_mse":   metrics["mse"],
        "test_rmse":  metrics["rmse"],
    }
    # Fold in best_val_loss from training if it's available.
    losses_path = out_dir / "train_val_losses.json"
    if losses_path.exists():
        try:
            with open(losses_path) as f:
                summary["best_val_loss"] = json.load(f).get("best_val_loss")
        except (json.JSONDecodeError, OSError):
            pass

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Optional wandb logging ────────────────────────────────────────────────
    if cfg.get("use_wandb", True):
        run = wandb.init(
            project = cfg.get("wandb_project", "stgnn-dengue"),
            name    = f"{cfg['name']}_{split}",
            config  = params,
        )
        wandb.run.summary.update(summary)
        wandb.log({
            "plots/predictions": wandb.Image(str(pred_plot)),
            "plots/scatter":     wandb.Image(str(scatter_plot)),
        })
        run.finish()

    print(f"\nAll outputs saved to {out_dir}")


# ── Entry point (Snakemake script mode or CLI) ────────────────────────────────

def _main_cli():
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate a trained STGATGRU model.")
    ap.add_argument("--config", required=True, help="Path to config YAML.")
    ap.add_argument("--model", default=None,
                    help="Checkpoint .pt to evaluate (default: results dir best_model.pt).")
    ap.add_argument("--params", default=None,
                    help="best_params.json used to build the model (default: results dir).")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="Which split to evaluate on (default: test).")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    params     = load_best_params(cfg, override_path=args.params)
    model_path = Path(args.model) if args.model else _results_dir(cfg) / "best_model.pt"
    test(cfg, params, model_path, split=args.split)


if __name__ == "__main__":
    if "snakemake" in globals():
        with open(snakemake.params.cfg) as f:                       # noqa: F821
            cfg = yaml.safe_load(f)
        override   = getattr(snakemake.params, "best_params", None)  # noqa: F821
        params     = load_best_params(cfg, override_path=override)
        model_path = getattr(snakemake.input, "model", None)         # noqa: F821
        model_path = Path(model_path) if model_path else _results_dir(cfg) / "best_model.pt"
        test(cfg, params, model_path, split="test")
    else:
        _main_cli()