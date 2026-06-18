"""
Train the final STGATGRU model from a given hyperparameter JSON.

Outputs (results/STGNN/<name>/):
    best_model.pt          — state dict of the best (lowest val loss) epoch
    loss_curves.png        — train/val Huber loss curves
    train_val_losses.json  — raw per-epoch losses + best_val_loss

Run via Snakemake (script mode) or directly:
    python train.py --config config.yaml [--params best_params.json]
"""

import json
import yaml
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from model import STGATGRU
from loss  import masked_huber_loss
from tune  import (
    STGNNDataset, load_tensors, load_edge_index,
    check_isolated_nodes, run_epoch,
)


# ── Paths / params ────────────────────────────────────────────────────────────

def _results_dir(cfg: dict) -> Path:
    return Path("results/STGNN") / cfg["name"]


def _best_params_path(cfg: dict) -> Path:
    return _results_dir(cfg) / "best_params.json"


def load_best_params(cfg: dict, override_path: str = None) -> dict:
    path = Path(override_path) if override_path else _best_params_path(cfg)
    with open(path) as f:
        return json.load(f)


# ── Evaluation pass (used for val metrics during training) ────────────────────

def eval_with_metrics(
    model:      STGATGRU,
    dataloader: DataLoader,
    edge_index: torch.Tensor,
    device:     torch.device,
    log_scale:  bool = True,
) -> tuple[float, dict, list, list]:
    """
    Run a full evaluation pass over parallel sequence streams.

    Returns:
        mean_loss   — mean masked Huber loss
        metrics     — dict with mae, mse, rmse (filtered to keep valid nodes only)
        all_preds   — list of batched numpy arrays
        all_targets — list of batched numpy arrays
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_masks = [], [], []

    with torch.no_grad():
        for x, y, mask in dataloader:
            x    = x.to(device, non_blocking=True)
            y    = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Vectorized multi-step forward pass
            pred = model(x, edge_index, mask=mask)
            if pred.dim() == 3 and y.dim() == 2:
                pred = pred.squeeze(-1)

            # Isolate the missingness flag corresponding to our prediction step target
            target_mask = mask[:, -1, :]  # Shape: (B, N)
            loss = masked_huber_loss(pred, y, target_mask)
            total_loss += loss.item()

            p = torch.expm1(pred) if log_scale else pred
            t = torch.expm1(y)    if log_scale else y
            
            all_preds.append(p.cpu().numpy())
            all_targets.append(t.cpu().numpy())
            all_masks.append(target_mask.cpu().numpy())

    mean_loss = total_loss / max(len(dataloader), 1)

    # Flatten tensors across entire dataset split
    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks   = np.concatenate(all_masks,   axis=0).astype(bool)

    # CRITICAL FIX: Strip out missing observation data points before calculating statistics
    valid_preds   = preds[masks]
    valid_targets = targets[masks]

    mae = float(np.abs(valid_preds - valid_targets).mean()) if valid_preds.size > 0 else 0.0
    mse = float(((valid_preds - valid_targets) ** 2).mean()) if valid_preds.size > 0 else 0.0
    metrics = {"mae": mae, "mse": mse, "rmse": mse ** 0.5}

    return mean_loss, metrics, all_preds, all_targets


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_loss_curves(
    train_losses: list[float],
    val_losses:   list[float],
    out_dir:      Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train (Huber)", linewidth=1.5)
    ax.plot(epochs, val_losses,   label="Val (Huber)",   linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked Huber Loss")
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "loss_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg: dict, params: dict):
    """Full training run with the given hyperparameters."""
    out_dir = _results_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project = cfg.get("wandb_project", "stgnn-dengue"),
        name    = f"{cfg['name']}_final",
        config  = params,
    )

    device      = torch.device(cfg.get("device", "cpu"))
    window_size = params["window_size"]
    num_epochs  = params["num_epochs"]
    batch_size  = params.get("batch_size", 32) # Injected optimal sweep configuration 
    log_scale   = cfg.get("log_scale", True)

    # ── Data Pipeline ────────────────────────────────────────────────────────
    tensors       = load_tensors(cfg, window_size)
    train_dataset = STGNNDataset(tensors, "train")
    val_dataset   = STGNNDataset(tensors, "val")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=2, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=2, persistent_workers=True
    )

    edge_index  = load_edge_index(cfg, device)
    in_channels = train_dataset.x.shape[-1]
    n_nodes     = train_dataset.x.shape[2]

    check_isolated_nodes(edge_index, n_nodes, device)

    # ── Model Construction ───────────────────────────────────────────────────
    model = STGATGRU(
        in_channels  = in_channels,
        gat1_hidden  = params["gat1_hidden"],
        gat1_heads   = params["gat1_heads"],
        mlp_hidden   = params.get("mlp_hidden", params["gat1_hidden"] * params["gat1_heads"]),
        mlp_layers   = params["mlp_layers"],
        gat2_hidden  = params["gat2_hidden"],
        gat2_heads   = params["gat2_heads"],
        gru_hidden   = params["gat2_hidden"],
        pred_horizon = 1,
        dropout      = params["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    # ── Unified Vector Training ──────────────────────────────────────────────
    best_val_loss   = float("inf")
    train_losses    = []
    val_losses      = []
    best_model_path = out_dir / "best_model.pt"

    for epoch in range(num_epochs):
        train_loss = run_epoch(model, train_loader, edge_index, device, optimizer)
        val_loss   = run_epoch(model, val_loader,   edge_index, device)

        if train_loss != train_loss or val_loss != val_loss:
            print(f"NaN detected at epoch {epoch + 1} — stopping.")
            break

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Run vectorized evaluation over full loaders
        _, val_metrics, _, _ = eval_with_metrics(
            model, val_loader, edge_index, device, log_scale=log_scale
        )

        wandb.log({
            "epoch":             epoch + 1,
            "train/huber_loss":  train_loss,
            "val/huber_loss":    val_loss,
            "val/mae":           val_metrics["mae"],
            "val/mse":           val_metrics["mse"],
            "val/rmse":          val_metrics["rmse"],
        })

        print(
            f"Epoch {epoch+1:>4}/{num_epochs} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"MAE {val_metrics['mae']:.4f} | RMSE {val_metrics['rmse']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            wandb.run.summary["best_val_loss"] = best_val_loss

    # ── Loss curves + raw losses ──────────────────────────────────────────────
    loss_plot = plot_loss_curves(train_losses, val_losses, out_dir)
    wandb.log({"plots/loss_curves": wandb.Image(str(loss_plot))})

    with open(out_dir / "train_val_losses.json", "w") as f:
        json.dump(
            {
                "train":         train_losses,
                "val":           val_losses,
                "best_val_loss": best_val_loss,
            },
            f,
            indent=2,
        )

    run.finish()
    print(f"\nTraining complete. Best model saved to {best_model_path}")


# ── Entry point (Snakemake script mode or CLI) ────────────────────────────────

def _main_cli():
    import argparse

    ap = argparse.ArgumentParser(description="Train the final STGATGRU model.")
    ap.add_argument("--config", required=True, help="Path to config YAML.")
    ap.add_argument("--params", default=None,
                    help="Path to best_params.json (default: results dir).")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    params = load_best_params(cfg, override_path=args.params)
    train(cfg, params)


if __name__ == "__main__":
    if "snakemake" in globals():
        with open(snakemake.params.cfg) as f:          # noqa: F821
            cfg = yaml.safe_load(f)
        override = getattr(snakemake.params, "best_params", None)  # noqa: F821
        params   = load_best_params(cfg, override_path=override)
        train(cfg, params)
    else:
        _main_cli()