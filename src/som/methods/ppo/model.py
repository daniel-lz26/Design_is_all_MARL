"""Networks for the PPO method: observation normalization, a candidate-scoring actor and a critic.

The actor follows spec §11.3. One shared MLP scores each of the K board tasks from (task row ⊕
context). The context is built from the robot's own features, the station rows, the global
features and a masked mean/max pool over the other-robot rows. Park and wait get their own
logits from the context, and NOOP gets a constant logit (it is masked out whenever a robot
decides). Scores do not depend on the order of the tasks or of the other robots, nor on how
many robots there are.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from som.cell_env.obs import TASK_FEATURES, ObsLayout

MASKED_LOGIT = -1e9
STD_FLOOR = 0.1  # features that barely vary in the fitting data are not blown up
NORM_CLIP = 10.0


def _lin(i: int, o: int, gain: float = float(np.sqrt(2))) -> nn.Linear:
    lin = nn.Linear(i, o)
    nn.init.orthogonal_(lin.weight, gain)
    nn.init.zeros_(lin.bias)
    return lin


def work_conserving_mask(mask: torch.Tensor, k: int) -> torch.Tensor:
    """Clear park (slot ``k``) and wait (``k + 1``) in every row where a task slot is valid.

    A robot may then only idle when there is nothing to pick up, like the rules and the auction.
    Without this, the spec §8 reward lets a learner leave late orders undelivered almost for
    free (order abandonment, see docs/method_ppo.md). A claimable task always exists when a slot
    is valid (``open_tasks`` only lists boxes whose next station has room), so no row is left
    without a legal action. NOOP (``k + 2``) is untouched."""
    any_task = (mask[..., :k] != 0).any(-1, keepdim=True)
    out = mask.clone()
    out[..., k : k + 2] = out[..., k : k + 2].masked_fill(any_task, 0)
    return out


def mlp(i: int, hidden: Sequence[int], o: int, out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = i
    for h in hidden:
        layers += [_lin(d, h), nn.LayerNorm(h), nn.ReLU()]
        d = h
    layers.append(_lin(d, o, out_gain))
    return nn.Sequential(*layers)


class ObsNorm(nn.Module):
    """Fixed per-feature standardization, fitted once on demonstration data and then frozen,
    so a warm-started actor sees the same inputs during PPO as during behaviour cloning."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def fit(self, x: np.ndarray) -> None:
        self.mean.copy_(torch.as_tensor(x.mean(0), dtype=torch.float32))
        self.std.copy_(torch.as_tensor(np.maximum(x.std(0), STD_FLOOR), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / self.std).clamp(-NORM_CLIP, NORM_CLIP)


class CandidateActor(nn.Module):
    """``work_conserving`` applies :func:`work_conserving_mask` inside :meth:`logits`, the one
    place every path (sampling in rollouts, the PPO update, behaviour cloning and greedy
    evaluation) turns a mask into action probabilities, so training and evaluation always match.
    Checkpoints from before the fix load with it off and keep their original behaviour."""

    def __init__(self, layout: ObsLayout, hidden: int = 128, work_conserving: bool = False) -> None:
        super().__init__()
        if layout.mode == "global":
            raise ValueError("the PPO actor is decentralized; use obs_mode board or local")
        self.lay = layout
        self.work_conserving = work_conserving
        self.n_task = len(TASK_FEATURES)
        self.n_other = (layout.global_at - layout.others_at) // (layout.max_robots - 1)
        self.norm = ObsNorm(layout.dim)
        ctx_in = (
            layout.self_dim
            + (layout.others_at - layout.stations_at)
            + (layout.orders_at - layout.global_at)
            + 2 * hidden // 2
        )
        self.other_enc = mlp(self.n_other, [hidden // 2], hidden // 2, out_gain=1.0)
        self.ctx = mlp(ctx_in, [hidden], hidden, out_gain=1.0)
        self.score = mlp(self.n_task + hidden, [hidden, hidden // 2], 1, out_gain=0.01)
        self.idle = _lin(hidden, 2, gain=0.01)  # park, wait

    def logits(self, raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lay, b = self.lay, raw.shape[0]
        x = self.norm(raw)
        selfv = x[:, : lay.self_dim]
        tasks = x[:, lay.tasks_at : lay.stations_at].reshape(b, lay.k, self.n_task)
        stations = x[:, lay.stations_at : lay.others_at]
        others = x[:, lay.others_at : lay.global_at].reshape(b, -1, self.n_other)
        glob = x[:, lay.global_at : lay.orders_at]
        present = raw[:, lay.others_at : lay.global_at].reshape(b, -1, self.n_other)[..., :1] > 0.5
        e = self.other_enc(others)
        n = present.sum(1).clamp(min=1)
        mean = (e * present).sum(1) / n
        mx = e.masked_fill(~present, -1e4).max(1).values
        mx = torch.where(present.any(1), mx, torch.zeros_like(mx))
        ctx = self.ctx(torch.cat([selfv, stations, glob, mean, mx], -1))
        scores = self.score(
            torch.cat([tasks, ctx.unsqueeze(1).expand(b, lay.k, ctx.shape[-1])], -1)
        ).squeeze(-1)
        noop = torch.zeros(b, 1, dtype=scores.dtype)
        out = torch.cat([scores, self.idle(ctx), noop], -1)
        if self.work_conserving:
            mask = work_conserving_mask(mask, lay.k)
        return out.masked_fill(mask == 0, MASKED_LOGIT)

    def dist(self, raw: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.logits(raw, mask))


class Critic(nn.Module):
    """V(s): IPPO feeds the robot's own observation, MAPPO its observation ⊕ the critic state."""

    def __init__(self, dim: int, hidden: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        self.norm = ObsNorm(dim)
        self.net = mlp(dim, hidden, 1, out_gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x)).squeeze(-1)
