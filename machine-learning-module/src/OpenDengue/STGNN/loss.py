# loss.py
import torch
import torch.nn.functional as F


def masked_huber_loss(
    pred:  torch.Tensor,
    target: torch.Tensor,
    mask:  torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Huber loss computed only over masked (valid surveillance) nodes.

    Args:
        pred:   (N,) or (N, 1) — model predictions for this timestep.
        target: (N,)           — ground truth incidence.
        mask:   (N,)           — bool, True = valid node.
        delta:  Huber threshold. Below delta: MSE. Above: MAE.

    Returns:
        Scalar loss over valid nodes. Returns 0 if no valid nodes.
    """
    pred   = pred.squeeze(-1)     # (N,)
    target = target.squeeze(-1)   # (N,)
    mask   = mask.bool()

    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return F.huber_loss(pred[mask], target[mask], delta=delta, reduction="mean")