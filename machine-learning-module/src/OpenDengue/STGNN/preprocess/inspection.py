import torch
from tune import load_tensors, make_windows
cfg = ...  # your cfg dict
tensors = load_tensors(cfg, 6)
x, y, mask = make_windows(tensors, "train")[0]   # the window you just tried to fit

m = mask.bool()
print("valid nodes:", int(m.sum()), "of", mask.numel())
print("x shape:", tuple(x.shape), "NaNs:", int(torch.isnan(x).sum()))

# Does each feature vary ACROSS nodes? (std over the node axis, per feature)
node_std = x.std(dim=1)                 # (window, F)
print("per-feature std across nodes (mean):", node_std.mean(dim=0))
print("fraction of x exactly 0:", (x == 0).float().mean().item())

# Do the valid-node TARGETS actually have spread for the model to capture?
print("y over valid nodes -> std:", y.squeeze(-1)[m].std().item(),
      "min/max:", y.squeeze(-1)[m].min().item(), y.squeeze(-1)[m].max().item())