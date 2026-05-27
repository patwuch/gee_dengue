from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.nn import GATConv
import torch_geometric_temporal

class STGATGRU(nn.Module):
    def __init__(self, in_channels,
                 gat1_hidden, gat1_heads,
                 mlp_hidden, mlp_layers,
                 gat2_hidden, gat2_heads,
                 gru_hidden,
                 pred_horizon,
                 dropout=0.5):
        super(STGATGRU, self).__init__()
        
        self.gat1 = GATConv(in_channels, gat1_hidden, heads=gat1_heads, concat=True, dropout=dropout)
        self.mlp = self.build_mlp(gat1_hidden * gat1_heads, mlp_hidden, mlp_layers, dropout)
        self.gat2 = GATConv(mlp_hidden, gat2_hidden, heads=gat2_heads, concat=False)
        self.gru = nn.GRUCell(gat2_hidden, gru_hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(gru_hidden, pred_horizon)

    def build_mlp(self, input_size, hidden_size, num_layers, dropout):
        layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x, edge_index, H=None):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.mlp(x)
        x = self.gat2(x, edge_index)
        x = self.dropout(x)
        H = self.gru(x, H)
        out = self.linear(H)
        return out, H