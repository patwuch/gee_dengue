"""
Overfit-a-batch diagnostic for STGATGRU.

Purpose
-------
Take a handful of TRAINING windows, disable ALL dropout, remove gradient
clipping, and try to memorise them. A correctly wired model with adequate
capacity MUST be able to drive this loss toward zero and produce per-node
predictions that vary (and match the targets).

Interpreting the result
-----------------------
  * Loss collapses toward 0 AND prediction std > 0
        -> Architecture + optimisation are fine. The mean-collapse you see in
           the full run is caused by REGULARISATION (the stacked dropout, esp.
           pre-GRU) and/or training setup. Fix: cut dropout, batch windows,
           add an LR scheduler, then re-evaluate.

  * Loss plateaus AND prediction std ~ 0 (stays constant)
        -> Even at full capacity the model cannot fit a tiny batch. This is a
           BUG or a DEAD-SIGNAL problem (features uninformative after x*mask,
           targets/wiring broken, gradients not reaching the head). Tuning the
           full run will NOT help; investigate data + forward path.

Usage
-----
    python overfit_test.py \
        --data-dir data/processed/machine-learning/STGNN/baseline_logIR \
        --best-params results/STGNN/baseline_logIR/best_params.json \
        --n-windows 1 --steps 2000 --lr 1e-2

Run with --n-windows 1 first (easiest possible case). If even one window
won't fit, you have your answer immediately.
"""
import argparse
import torch

from model import STGATGRU
from loss import masked_huber_loss


# ── Core (kept import-light so it can be unit-tested in isolation) ─────────────

def run_forward(model, x, edge_index, mask):
    """Mirror exactly the per-timestep loop used in tune.run_epoch."""
    H = None
    node_mask = mask.unsqueeze(-1)              # (N, 1) — static per window, as in your code
    for t in range(x.size(0)):
        pred, H = model(x[t], edge_index, mask=node_mask, H=H)
    return pred                                 # (N, 1) — only last step is supervised


def overfit(model, windows, edge_index, device, steps=2000, lr=1e-2, log_every=100):
    """
    Gradient-accumulate over the (few) windows each step. No clipping, no dropout.
    Returns the loss history.
    """
    model.train()                              # dropout already 0, so mode is harmless
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for step in range(steps):
        opt.zero_grad()
        total = 0.0
        for x, y, mask in windows:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred = run_forward(model, x, edge_index, mask)
            loss = masked_huber_loss(pred, y, mask)
            loss.backward()                    # accumulate across windows (= batch them)
            total += loss.item()
        opt.step()

        mean_loss = total / len(windows)
        history.append(mean_loss)
        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step:>5} | loss {mean_loss:.6f}")

    return history


@torch.no_grad()
def report(model, windows, edge_index, device, n_show=12):
    """Inspect predictions on the (last) overfit window for the collapse signature."""
    model.eval()
    x, y, mask = windows[-1]
    x, y, mask = x.to(device), y.to(device), mask.to(device)
    pred = run_forward(model, x, edge_index, mask).squeeze(-1)
    y = y.squeeze(-1)
    m = mask.bool()

    p_valid = pred[m]
    y_valid = y[m]
    pred_std = p_valid.std().item()
    targ_std = y_valid.std().item()

    print("\n── Prediction inspection (last window, valid nodes) ──")
    print(f"  pred  : min {p_valid.min():+.4f}  max {p_valid.max():+.4f}  std {pred_std:.6f}")
    print(f"  target: min {y_valid.min():+.4f}  max {y_valid.max():+.4f}  std {targ_std:.6f}")
    k = min(n_show, p_valid.numel())
    print(f"\n  {'pred':>10}  {'target':>10}")
    for i in range(k):
        print(f"  {p_valid[i].item():>10.4f}  {y_valid[i].item():>10.4f}")

    return pred_std, targ_std


def verdict(history, pred_std, targ_std):
    init, final = history[0], history[-1]
    drop = (init - final) / max(init, 1e-12)
    collapsed = pred_std < 1e-4 or (targ_std > 0 and pred_std < 0.02 * targ_std)
    print("\n" + "=" * 64)
    print(f"  initial loss {init:.6f} -> final loss {final:.6f}   ({drop*100:.1f}% drop)")
    print(f"  prediction std on valid nodes: {pred_std:.6f}")
    if final < 0.05 * init and not collapsed:
        print("  VERDICT: model CAN fit. Architecture is mechanically sound.")
        print("           => Your full-run collapse is REGULARISATION/setup, not a bug.")
        print("           Next: cut dropout, batch windows, add LR schedule.")
    elif collapsed:
        print("  VERDICT: predictions stayed ~CONSTANT despite full capacity.")
        print("           => BUG or DEAD SIGNAL. Tuning won't help.")
        print("           Next: check features after `x*mask`, target tensor, grad flow.")
    else:
        print("  VERDICT: partial fit. Loss moved but didn't collapse to ~0.")
        print("           Try more steps / higher lr; if it stalls, suspect signal/wiring.")
    print("=" * 64)


# ── Entry point against YOUR real data ────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="Path to processed data dir, e.g. data/processed/machine-learning/STGNN/baseline_logIR")
    ap.add_argument("--best-params", required=True, help="Path to best_params.json")
    ap.add_argument("--n-windows", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import json
    from pathlib import Path
    from tune import make_windows, check_isolated_nodes

    data_dir = Path(args.data_dir)
    with open(args.best_params) as f:
        params = json.load(f)

    device = torch.device(args.device)

    tensors_path = data_dir / f"window_{params['window_size']}" / "tensors.pt"
    tensors = torch.load(tensors_path, weights_only=True)
    train_windows = make_windows(tensors, "train")[: args.n_windows]

    edge_index = torch.load(data_dir / "edge_index.pt", weights_only=True).to(device)

    in_channels = train_windows[0][0].shape[-1]
    n_nodes = train_windows[0][0].shape[1]
    check_isolated_nodes(edge_index, n_nodes, device)

    print(f"\nOverfitting {len(train_windows)} window(s) | "
          f"N={n_nodes} F={in_channels} | dropout FORCED to 0.0\n")

    model = STGATGRU(
        in_channels=in_channels,
        gat1_hidden=params["gat1_hidden"],
        gat1_heads=params["gat1_heads"],
        mlp_hidden=params["gat1_hidden"] * params["gat1_heads"],
        mlp_layers=params["mlp_layers"],
        gat2_hidden=params["gat2_hidden"],
        gat2_heads=params["gat2_heads"],
        gru_hidden=params["gat2_hidden"],
        pred_horizon=1,
        dropout=0.0,                            # <- the whole point
    ).to(device)

    history = overfit(model, train_windows, edge_index, device,
                      steps=args.steps, lr=args.lr)
    pred_std, targ_std = report(model, train_windows, edge_index, device)
    verdict(history, pred_std, targ_std)


if __name__ == "__main__":
    main()
