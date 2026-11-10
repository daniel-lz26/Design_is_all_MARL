"""RoleActor and Critic MLPs (LayerNorm, orthogonal init)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

MASKED_LOGIT = -1e9


def _layer(i: int, o: int, gain: float) -> nn.Linear:
    lin = nn.Linear(i, o)
    nn.init.orthogonal_(lin.weight, gain)
    nn.init.zeros_(lin.bias)
    return lin


def mlp(in_dim: int, hidden: Sequence[int], out_dim: int, out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
    d = in_dim
    for h in hidden:
        layers += [_layer(d, h, float(np.sqrt(2))), nn.LayerNorm(h), nn.ReLU()]
        d = h
    layers.append(_layer(d, out_dim, out_gain))
    return nn.Sequential(*layers)


class RoleActor(nn.Module):
    """Shared by every agent of one role. Outputs masked logits over K + 1 slots."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int]) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.net = mlp(obs_dim, hidden, n_actions, out_gain=0.01)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.net(obs)
        return logits.masked_fill(mask == 0, MASKED_LOGIT)

    def dist(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self(obs, mask))


class Critic(nn.Module):
    """V(s) from the agent-specific global state (MAPPO) or the agent's own obs (IPPO)."""

    def __init__(self, in_dim: int, hidden: Sequence[int]) -> None:
        super().__init__()
        self.net = mlp(in_dim, hidden, 1, out_gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
