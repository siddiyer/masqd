import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Multi-headed actor network with per-head hidden layers.
    Each head represents actions for a specific agent.
    """

    def __init__(self, state_dim, action_dim, hidden_size, num_heads):
        super().__init__()

        # Shared trunk layer
        self.linear1 = nn.Linear(state_dim, hidden_size)

        # Per-head layers
        self.per_head_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, hidden_size // 8),
                nn.ReLU(),
                nn.Linear(hidden_size // 8, action_dim)
            ) for _ in range(num_heads)
        ])

    def forward(self, state, head=0):
        x = torch.relu(self.linear1(state))
        return torch.tanh(self.per_head_layers[head](x))


class DCCritic(nn.Module):
    """
    Descriptor-conditioned twin critic: Q(s, a | d') for a target behavior descriptor d'.
    """

    def __init__(self, state_dim, action_dim, num_bds, hidden_size=256):
        super().__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim + num_bds, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim + num_bds, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, 1)

    def forward(self, state, action, bds):
        sa = torch.cat([state, action, bds], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action, bds):
        sa = torch.cat([state, action, bds], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


def get_flat_params(model: nn.Module) -> torch.Tensor:
    """Returns all parameters of model as one flat CPU tensor (the vector CEM samples over)."""
    params = []
    for p in model.parameters():
        params.append(p.data.view(-1).cpu())
    return torch.cat(params, dim=0)


def set_flat_params(model: nn.Module, flat_params: torch.Tensor):
    """Copies a flat parameter vector (as produced by get_flat_params) into model."""
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        new_vals = flat_params[idx:idx + numel]
        idx += numel
        p.data.copy_(new_vals.view(p.shape))
