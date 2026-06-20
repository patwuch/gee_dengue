import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class STGATGRU(nn.Module):
    """
    Spatio-Temporal Graph Attention Network with GRU.

    Spatial (GAT) and temporal (GRU) steps are both sequential over B and T
    to bound peak GPU memory on low-VRAM devices.
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

        gat1_in = in_channels + 1 

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
        
        self.gru = nn.GRU(gat2_hidden, gru_hidden, batch_first=True)

        self.missing_emb = nn.Parameter(torch.zeros(in_channels))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(gru_hidden, pred_horizon)
        self.gru_hidden = gru_hidden
        self._gat_out = gat2_hidden

    def _build_mlp(self, input_size, hidden_size, num_layers, dropout):
        layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x, edge_index, mask=None):
        """
        Args:
            x:          Node features sequence. Shape: (B, T, N, F)
            edge_index: Graph connectivity. Shape: (2, E)
            mask:       Binary missingness flag. Shape: (B, T, N) or (B, N)
        Returns:
            out:        Predictions shape: (B, N, pred_horizon)
        """
        B, T, N, F_dim = x.size()

        # --- Handle missingness masks ---
        if mask is None:
            mask = torch.ones(B, T, N, 1, device=x.device)
        else:
            mask = mask.float()
            if mask.dim() == 2:    # (B, N) → (B, 1, N, 1)
                mask = mask.unsqueeze(1).unsqueeze(-1)
            elif mask.dim() == 3:  # (B, T, N) → (B, T, N, 1)
                mask = mask.unsqueeze(-1)

        # Change 3: substitute learned embedding for missing nodes instead of zeroing.
        # tanh keeps the embedding in (-1, 1) — same range as normalized features —
        # and prevents it from exploding under high learning rates.
        missing_emb = torch.tanh(self.missing_emb).view(1, 1, 1, F_dim)
        x = x * mask + missing_emb * (1 - mask)
        x = torch.cat([x, mask.expand(B, T, N, 1)], dim=-1)

        # Prepare per-timestep missingness for GRU carry-forward (Change 2)
        # Expand to (B, T, N, 1) first — mask may be (B, 1, N, 1) if input was 2D.
        # gru_mask: (T, B*N) so gru_mask[t, b*N+n] = mask[b, t, n]
        gru_mask = mask.expand(B, T, N, 1)[:, :, :, 0].permute(1, 0, 2).reshape(T, B * N)

        # ── 2. Spatial layers — B and T sequentially ─────────────────────────
        # Loop over each sample independently so each GAT call sees N nodes,
        # not B×N. The shared edge_index is used directly — no disjoint-graph
        # construction needed. Force fp32: fp16 scatter/gather is unreliable
        # on sm<70 and some PyG builds.
        x_seq = torch.empty(B, N, T, self._gat_out, dtype=torch.float32, device=x.device)
        for b in range(B):
            for t in range(T):
                x_bt = x[b, t].float()                    # (N, F+1)
                with torch.autocast("cuda", enabled=False):
                    h = self.gat1(x_bt, edge_index)
                    h = F.elu(h)
                    h = self.mlp(h)
                    h = self.gat2(h, edge_index)           # (N, gat2_hidden)
                x_seq[b, :, t, :] = h

        # ── 3. Reshape for GRU: (B*N, T, gat2_hidden) ────────────────────────
        # (B, N, T, F) is already C-contiguous so reshape(B*N, T, F) is a view.
        x_seq = self.dropout(x_seq)
        x_seq = x_seq.reshape(B * N, T, -1)              # (B*N, T, gat2_hidden)

        # ── 4. Temporal layer & output ────────────────────────────────────────
        # Change 2: carry forward hidden state for missing nodes rather than updating
        h = torch.zeros(1, B * N, self.gru_hidden, device=x_seq.device, dtype=torch.float32)
        for t in range(T):
            m_t = gru_mask[t].unsqueeze(0).unsqueeze(-1).bool()  # (1, B*N, 1)
            with torch.autocast("cuda", enabled=False):
                _, h_new = self.gru(x_seq[:, t:t+1, :].float(), h)
            # torch.where avoids h_new * 0 = NaN when h_new overflows to inf in fp16
            h = torch.where(m_t.expand_as(h_new), h_new, h)
        del x_seq
        final_state = h.squeeze(0)         # (B*N, gru_hidden)

        out = self.linear(final_state)     # (B*N, pred_horizon)
        out = out.view(B, N, -1)           # (B, N, pred_horizon)

        return out