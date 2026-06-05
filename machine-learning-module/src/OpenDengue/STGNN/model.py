import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class STGATGRU(nn.Module):
    """
    Spatio-Temporal Graph Attention Network with GRU.

    Changes from original:
    - Tightly coupled: GRU hidden state H is concatenated to node features
      before GAT at each timestep, so spatial attention is temporally aware.
    - Explicit missingness flag: a binary feature appended to node features
      indicating which nodes have missing values at each timestep. This lets
      the model learn to distinguish zero from missing rather than treating
      them identically.

    Args:
        in_channels:    Number of input node features (excluding missingness flag).
        gat1_hidden:    Hidden units per head in first GAT layer.
        gat1_heads:     Number of attention heads in first GAT layer.
        mlp_hidden:     Hidden units in MLP between GAT layers.
        mlp_layers:     Number of MLP layers.
        gat2_hidden:    Hidden units per head in second GAT layer.
        gat2_heads:     Number of attention heads in second GAT layer (concat=False).
        gru_hidden:     GRU hidden state size.
        pred_horizon:   Number of future timesteps to predict.
        dropout:        Dropout rate.
    """

    def __init__(
        self,
        in_channels,
        gat1_hidden, gat1_heads,
        mlp_hidden, mlp_layers,
        gat2_hidden, gat2_heads,
        gru_hidden,
        pred_horizon,
        dropout=0.5
    ):
        super(STGATGRU, self).__init__()

        # +1 for the explicit missingness flag appended to node features
        # +gru_hidden for the hidden state concatenated at each timestep
        gat1_in = in_channels + 1 + gru_hidden

        self.gat1 = GATConv(
            gat1_in,
            gat1_hidden,
            heads=gat1_heads,
            concat=True,
            dropout=dropout
        )
        self.mlp = self._build_mlp(
            gat1_hidden * gat1_heads,
            mlp_hidden,
            mlp_layers,
            dropout
        )
        self.gat2 = GATConv(
            mlp_hidden,
            gat2_hidden,
            heads=gat2_heads,
            concat=False
        )
        self.gru = nn.GRUCell(gat2_hidden, gru_hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(gru_hidden, pred_horizon)

        self.gru_hidden = gru_hidden

    def _build_mlp(self, input_size, hidden_size, num_layers, dropout):
        layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x, edge_index, mask=None, H=None):
        """
        Args:
            x:          Node features. Shape: (num_nodes, in_channels).
            edge_index: Graph connectivity. Shape: (2, num_edges).
            mask:       Binary missingness flag. Shape: (num_nodes, 1).
                        1 = valid, 0 = missing. If None, all nodes assumed valid.
            H:          GRU hidden state from previous timestep.
                        Shape: (num_nodes, gru_hidden). If None, initialised to zeros.

        Returns:
            out:        Predictions. Shape: (num_nodes, pred_horizon).
            H:          Updated GRU hidden state. Shape: (num_nodes, gru_hidden).
        """
        num_nodes = x.size(0)

        # Initialise hidden state if not provided
        if H is None:
            H = torch.zeros(num_nodes, self.gru_hidden, device=x.device)

        # --- Missingness flag ---
        # Append explicit binary flag so model distinguishes zero from missing.
        # 1 = observed, 0 = missing.
        if mask is None:
            mask = torch.ones(num_nodes, 1, device=x.device)
        else:
            mask = mask.float()

        # Zero out features of missing nodes so they don't inject corrupted
        # signal into the spatial mixing, while the flag still tells the model
        # why those features are zero.
        x = x * mask                          # (num_nodes, in_channels)
        x = torch.cat([x, mask], dim=-1)      # (num_nodes, in_channels + 1)

        # --- Tight coupling: inject H before spatial attention ---
        # GAT now sees both current (masked) features and accumulated temporal
        # context, so attention scores reflect what has been happening over time.
        x = torch.cat([x, H], dim=-1)         # (num_nodes, in_channels + 1 + gru_hidden)

        # --- Spatial attention stack ---
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.mlp(x)
        x = self.gat2(x, edge_index)
        x = self.dropout(x)

        # --- Temporal update ---
        H = self.gru(x, H)

        # --- Prediction ---
        out = self.linear(H)

        return out, H