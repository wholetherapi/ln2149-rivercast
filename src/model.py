import torch
import torch.nn as nn

class GatedRecurrentMixture(nn.Module):
    """Gated recurrent mixture over basin-specific hidden states."""

    def __init__(self, input_dim: int, hidden_dim: int, n_basins: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gates = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_basins)]
        )
        self.mixing = nn.Softmax(dim=-1)

    def forward(self, x, basin_ids):
        h = torch.tanh(self.input_proj(x))
        states = torch.stack([torch.sigmoid(g(h)) * h for g in self.gates], dim=1)
        weights = self.mixing(states.mean(dim=-1))
        return (states * weights.unsqueeze(-1)).sum(dim=1)
