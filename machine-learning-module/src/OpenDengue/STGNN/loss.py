import torch
import torch.nn.functional as F


def masked_huber_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    mask:   torch.Tensor,
    delta:  float = 1.0,
) -> torch.Tensor:
    """
    Huber loss computed only over masked (valid surveillance) nodes across a batch.

    Args:
        pred:   (B, N) or (B, N, 1) — model predictions.
        target: (B, N) or (B, N, 1) — ground truth incidence.
        mask:   (B, N)              — bool/float, True/1 = valid node.
        delta:  Huber threshold.

    Returns:
        Scalar loss averaged across valid elements.
    """
    # ── 1. Safe Dimension Alignment ──────────────────────────────────────────
    # Avoid general .squeeze(-1) which can destroy a batch dimension of size 1.
    if pred.dim() == 3 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)
    if target.dim() == 3 and target.size(-1) == 1:
        target = target.squeeze(-1)
        
    mask = mask.bool()

    # ── 2. Differentiable Fallback ───────────────────────────────────────────
    if mask.sum() == 0:
        # Multiply a prediction element by 0.0 to keep the tensor attached 
        # to the autograd graph, preventing broken gradients.
        return pred.sum() * 0.0

    # ── 3. Masked Extracted Statistics ───────────────────────────────────────
    # Flat continuous indexing maps safely regardless of batch configurations
    return F.huber_loss(pred[mask], target[mask], delta=delta, reduction="mean")