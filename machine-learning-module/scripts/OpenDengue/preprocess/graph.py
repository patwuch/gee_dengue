from __future__ import annotations

import torch


def build_edge_index(df, cfg: dict) -> torch.Tensor:
    """Construct edge_index [2, num_edges] in COO format."""
    # TODO: build edge_index from adjacency or spatial relationship
    pass


def add_self_loops(edge_index: torch.Tensor) -> torch.Tensor:
    """Add self-loops to an existing edge_index."""
    # TODO: add self-loops
    pass