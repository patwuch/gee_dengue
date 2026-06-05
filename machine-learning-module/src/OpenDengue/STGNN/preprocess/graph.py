from __future__ import annotations

import torch


def build_edge_index(node_index: dict) -> torch.Tensor:
    """Fully connected graph — let attention learn spatial relationships."""
    n = len(node_index)
    
    src = torch.arange(n).repeat_interleave(n)
    dst = torch.arange(n).repeat(n)
    
    return torch.stack([src, dst])  # [2, n*n]

def add_self_loops(edge_index: torch.Tensor) -> torch.Tensor:
    """Add self-loops to existing edge_index."""
    n = edge_index.max().item() + 1
    
    self_loops = torch.arange(n)
    self_loops = torch.stack([self_loops, self_loops])  # [2, n]
    
    return torch.cat([edge_index, self_loops], dim=1)   # [2, n*n + n]