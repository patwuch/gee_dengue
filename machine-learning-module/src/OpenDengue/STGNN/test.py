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
from tune  import load_tensors, load_edge_index, make_windows


# ── Paths / params ────────────────────────────────────────────────────────────

def _results_dir(cfg: dict) -> Path:
    return Path("results/STGNN") / cfg["name"]


def _best_params_path(cfg: dict) -> Path:
    return _results_dir(cfg) / "best_params.json"


def load_best_params(cfg: dict, override_path: str = None) -> dict:
    path = Path(override_path) if override_path else _best_params_path(cfg)
    with open(path) as f:
        return json.load(f)


# ── Evaluation pass ───────────────────────────────────────────────────────────
def eval_with_metrics(
    model:      STGATGRU,
    dataloader:  DataLoader, # Updated to accept your new batched loader
    edge_index: torch.Tensor,
    device:     torch.device,
    log_scale:  bool = True,
) -> tuple[float, dict, list, list]:
    
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_masks = [], [], []

    with torch.no_grad():
        for x, y, mask in dataloader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            # Parallelized sequence pass matching your new architecture
            pred = model(x, edge_index, mask=mask) 
            if pred.dim() == 3 and y.dim() == 2:
                pred = pred.squeeze(-1)

            # Track the mask belonging to the target step (the last window frame)
            target_mask = mask[:, -1, :].squeeze(-1)
            
            loss = masked_huber_loss(pred, y, target_mask)
            total_loss += loss.item()

            p = torch.expm1(pred) if log_scale else pred
            t = torch.expm1(y)    if log_scale else y
            
            all_preds.append(p.cpu().numpy())
            all_targets.append(t.cpu().numpy())
            all_masks.append(target_mask.cpu().numpy()) # Save the masks!

    mean_loss = total_loss / max(len(dataloader), 1)

    # Flatten everything across the entire evaluation split
    preds   = np.concatenate(all_preds,   axis=0)   
    targets = np.concatenate(all_targets, axis=0)
    masks   = np.concatenate(all_masks,   axis=0).astype(bool) # Convert to boolean index

    # CRITICAL FIX: Filter out missing data points before calculating final summary stats
    valid_preds   = preds[masks]
    valid_targets = targets[masks]

    mae = float(np.abs(valid_preds - valid_targets).mean())
    mse = float(((valid_preds - valid_targets) ** 2).mean())
    metrics = {"mae": mae, "mse": mse, "rmse": mse ** 0.5}

    return mean_loss, metrics, all_preds, all_targets

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_predictions(
    all_preds:   list,
    all_targets: list,
    out_dir:     Path,
    n_nodes:     int = 6,
) -> Path:
    """Plot pred vs target IR for the first n_nodes nodes across all windows."""
    preds   = np.stack(all_preds,   axis=0)   # (W, N)
    targets = np.stack(all_targets, axis=0)   # (W, N)
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
    log_scale   = cfg.get("log_scale", True)

    # ── Data ─────────────────────────────────────────────────────────────────
    tensors     = load_tensors(cfg, window_size)
    windows     = make_windows(tensors, split)
    edge_index  = load_edge_index(cfg, device)
    in_channels = windows[0][0].shape[-1]

    # ── Rebuild model & load weights ─────────────────────────────────────────
    model = STGATGRU(
        in_channels  = in_channels,
        gat1_hidden  = params["gat1_hidden"],
        gat1_heads   = params["gat1_heads"],
        mlp_hidden   = params["gat1_hidden"] * params["gat1_heads"],
        mlp_layers   = params["mlp_layers"],
        gat2_hidden  = params["gat2_hidden"],
        gat2_heads   = params["gat2_heads"],
        gru_hidden   = params["gat2_hidden"],
        pred_horizon = 1,
        dropout      = params["dropout"],
    ).to(device)

    print(f"Loading model from {model_path} ...")
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # ── Evaluate ──────────────────────────────────────────────────────────────
    loss, metrics, preds, targets = eval_with_metrics(
        model, windows, edge_index, device, log_scale=log_scale
    )

    print(
        f"\n{split} results | Huber {loss:.4f} | "
        f"MAE {metrics['mae']:.4f} | RMSE {metrics['rmse']:.4f}"
    )

    # ── Save predictions ──────────────────────────────────────────────────────
    preds_path = out_dir / f"{split}_predictions.npz"
    np.savez(
        preds_path,
        predictions = np.stack(preds,   axis=0),
        targets     = np.stack(targets, axis=0),
    )
    print(f"Predictions saved to {preds_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    pred_plot    = plot_predictions(preds, targets, out_dir)
    scatter_plot = plot_scatter(preds, targets, out_dir)

    # ── Metrics summary ───────────────────────────────────────────────────────
    summary = {
        f"{split}_huber": loss,
        f"{split}_mae":   metrics["mae"],
        f"{split}_mse":   metrics["mse"],
        f"{split}_rmse":  metrics["rmse"],
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