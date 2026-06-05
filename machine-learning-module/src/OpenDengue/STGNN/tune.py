# tune.py
import json
import yaml
import torch
import wandb
from pathlib import Path

from model import STGATGRU
from loss  import masked_huber_loss


# ── Helpers ───────────────────────────────────────────────────────────────────

def _processed_dir(cfg: dict) -> Path:
    return Path("data/processed/STGNN") / cfg["name"]


def load_tensors(cfg: dict, window_size: int) -> dict:
    path = _processed_dir(cfg) / f"window_{window_size}" / "tensors.pt"
    return torch.load(path, weights_only=True)


def load_edge_index(cfg: dict, device: torch.device) -> torch.Tensor:
    path = _processed_dir(cfg) / "edge_index.pt"
    return torch.load(path, weights_only=True).to(device)


def make_windows(tensors: dict, split: str) -> list[tuple]:
    x    = tensors[f"{split}_x"]
    y    = tensors[f"{split}_y"]
    mask = tensors[f"{split}_mask"]
    return list(zip(x, y, mask))


def check_isolated_nodes(edge_index: torch.Tensor, n_nodes: int, device: torch.device) -> bool:
    """
    Warn if any nodes have no edges. Isolated nodes cause NaN in GAT
    (softmax over empty neighbourhood is undefined).

    Returns True if all nodes have edges, False if any are isolated.
    """
    nodes_with_edges = edge_index.unique()
    all_nodes        = torch.arange(n_nodes, device=device)
    isolated         = all_nodes[~torch.isin(all_nodes, nodes_with_edges)]

    if isolated.numel() > 0:
        print(f"WARNING: {isolated.numel()} isolated nodes at indices: "
              f"{isolated.tolist()} — these will produce NaN in GAT layers.")
        return False

    print(f"Graph OK: all {n_nodes} nodes have edges.")
    return True


def run_epoch(
    model:      STGATGRU,
    windows:    list,
    edge_index: torch.Tensor,
    device:     torch.device,
    optimizer:  torch.optim.Optimizer = None,
) -> float:
    """
    Run one full epoch over all windows.

    Returns mean loss, or float('nan') if a NaN gradient is detected
    (signals the sweep trial to abort cleanly).
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for x, y, mask in windows:
            x    = x.to(device)
            y    = y.to(device)
            mask = mask.to(device)

            # Diagnose which features are NaN
            if torch.isnan(x).any():
                nan_features = torch.isnan(x).any(dim=0).any(dim=0)  # (F,)
                print(f"NaN in feature indices: {nan_features.nonzero(as_tuple=True)[0].tolist()}")
                print(f"x shape: {x.shape}")
                raise AssertionError("NaN in x — see feature indices above")

            assert not torch.isnan(x).any(), "NaN in x — check imputation pipeline"
            assert not torch.isnan(y).any(), "NaN in y — check incidence tensor"
            assert not torch.isinf(x).any(), "Inf in x — check scaling"

            H = None
            for t in range(x.size(0)):
                node_mask = mask.unsqueeze(-1)              # (N, 1)
                pred, H   = model(x[t], edge_index,
                                  mask=node_mask, H=H)      # pred: (N, 1)

            loss = masked_huber_loss(pred, y, mask)

            if training:
                optimizer.zero_grad()
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                if torch.isnan(total_norm):
                    print("NaN gradient detected — aborting trial.")
                    return float("nan")
                optimizer.step()

            total_loss += loss.item()

    return total_loss / max(len(windows), 1)


# ── Sweep ─────────────────────────────────────────────────────────────────────

def build_sweep_config(cfg: dict) -> dict:
    ss = cfg["tune"]["search_space"]["stgnn"]

    def param(spec):
        if spec["type"] == "log_uniform":
            return {"distribution": "log_uniform_values",
                    "min": spec["low"], "max": spec["high"]}
        if spec["type"] == "categorical":
            return {"values": spec["choices"]}
        raise ValueError(f"Unknown search space type: {spec['type']}")

    return {
        "method": "bayes",
        "metric": {"name": cfg["tune"]["metric"], "goal": "minimize"},
        "parameters": {k: param(v) for k, v in ss.items()},
    }


def train_sweep(cfg: dict):
    """Single W&B sweep agent run — called once per trial."""
    run = wandb.init()
    wc  = run.config

    device      = torch.device(cfg["tune"].get("device", "cpu"))
    window_size = wc.window_size

    # ── Data ─────────────────────────────────────────────────────────────────
    tensors       = load_tensors(cfg, window_size)
    train_windows = make_windows(tensors, "train")
    val_windows   = make_windows(tensors, "val")

    edge_index  = load_edge_index(cfg, device)
    in_channels = train_windows[0][0].shape[-1]   # F from (window_size, N, F)
    n_nodes     = train_windows[0][0].shape[1]     # N from (window_size, N, F)

    # ── Graph sanity check ────────────────────────────────────────────────────
    graph_ok = check_isolated_nodes(edge_index, n_nodes, device)
    if not graph_ok:
        wandb.log({"masked_huber_loss": float("nan")})
        run.finish()
        return

    # ── Model ─────────────────────────────────────────────────────────────────
    model = STGATGRU(
        in_channels  = in_channels,
        gat1_hidden  = wc.gat1_hidden,
        gat1_heads   = wc.gat1_heads,
        mlp_hidden   = wc.gat1_hidden * wc.gat1_heads,
        mlp_layers   = wc.mlp_layers,
        gat2_hidden  = wc.gat2_hidden,
        gat2_heads   = wc.gat2_heads,
        gru_hidden   = wc.gat2_hidden,
        pred_horizon = 1,
        dropout      = wc.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=wc.learning_rate)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(wc.num_epochs):
        train_loss = run_epoch(model, train_windows, edge_index, device, optimizer)
        val_loss   = run_epoch(model, val_windows,   edge_index, device)

        # Abort trial cleanly if NaN gradients were detected
        if train_loss != train_loss or val_loss != val_loss:  # nan check
            wandb.log({"masked_huber_loss": float("nan")})
            run.finish()
            return

        wandb.log({
            "epoch":             epoch + 1,
            "train/huber_loss":  train_loss,
            "val/huber_loss":    val_loss,
            "masked_huber_loss": val_loss,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wandb.run.summary["best_val_loss"] = best_val_loss

    run.finish()


def run_sweep(cfg: dict):
    """Initialise and run the W&B sweep, then save best params to disk."""
    sweep_config = build_sweep_config(cfg)

    sweep_id = wandb.sweep(
        sweep_config,
        project = cfg.get("wandb_project", "stgnn-dengue"),
    )

    wandb.agent(
        sweep_id,
        function = lambda: train_sweep(cfg),
        count    = cfg["tune"]["n_trials"],
    )

    # ── Fetch best run and save params for Snakemake output ───────────────────
    api      = wandb.Api()
    sweep    = api.sweep(f"{cfg.get('wandb_project', 'stgnn-dengue')}/{sweep_id}")
    best_run = sweep.best_run()

    best_params = dict(best_run.config)
    best_params["best_val_loss"] = best_run.summary.get("best_val_loss")

    out_path = Path("results/STGNN") / cfg["name"] / "best_params.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)


# ── Snakemake entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    with open(snakemake.params.cfg) as f:
        cfg = yaml.safe_load(f)
    run_sweep(cfg)