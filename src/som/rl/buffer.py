"""Rollout buffer over (time, env, agent) with dt, masks and decision flags, plus semi-MDP GAE.

One ``RoleBuffer`` per role holds arrays shaped ``[T, E, n_role, ...]``. Env-level arrays
(``dt``, ``done``, ``truncated``) are shared by both roles and live on ``RolloutBuffer``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RoleBuffer:
    obs: np.ndarray  # [T, E, n, obs_dim]
    mask: np.ndarray  # [T, E, n, K + 1]
    critic_in: np.ndarray  # [T, E, n, critic_dim]
    action: np.ndarray  # [T, E, n]
    logp: np.ndarray  # [T, E, n]
    value: np.ndarray  # [T, E, n], denormalized
    decision: np.ndarray  # [T, E, n] bool
    reward: np.ndarray  # [T, E, n]
    trunc_value: np.ndarray  # [T, E, n], V(final state) where the episode was truncated
    adv: np.ndarray | None = None
    ret: np.ndarray | None = None

    @classmethod
    def empty(cls, t: int, e: int, n: int, obs_dim: int, k1: int, critic_dim: int) -> RoleBuffer:
        f = np.float32
        return cls(
            obs=np.zeros((t, e, n, obs_dim), f),
            mask=np.zeros((t, e, n, k1), np.int8),
            critic_in=np.zeros((t, e, n, critic_dim), f),
            action=np.zeros((t, e, n), np.int64),
            logp=np.zeros((t, e, n), f),
            value=np.zeros((t, e, n), f),
            decision=np.zeros((t, e, n), bool),
            reward=np.zeros((t, e, n), f),
            trunc_value=np.zeros((t, e, n), f),
        )


@dataclass
class RolloutBuffer:
    roles: dict[str, RoleBuffer]
    dt: np.ndarray  # [T, E]
    done: np.ndarray  # [T, E] bool: episode ended after step t
    truncated: np.ndarray  # [T, E] bool


def semi_mdp_gae(
    reward: np.ndarray,
    value: np.ndarray,
    last_value: np.ndarray,
    trunc_value: np.ndarray,
    dt: np.ndarray,
    done: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE with per-step discount ``gamma ** dt``.

    ``reward``, ``value``, ``trunc_value``: [T, E, n]; ``last_value``: [E, n]; ``dt``, ``done``,
    ``truncated``: [T, E]. A terminated episode bootstraps 0; a truncated one bootstraps
    ``trunc_value`` (the horizon is not a real end of the task).
    """
    t_len = reward.shape[0]
    adv = np.zeros_like(reward)
    gae = np.zeros_like(last_value)
    for t in reversed(range(t_len)):
        next_v = last_value if t == t_len - 1 else value[t + 1]
        d = done[t][:, None]
        boot = np.where(truncated[t][:, None], trunc_value[t], 0.0)
        next_v = np.where(d, boot, next_v)
        disc = (gamma ** dt[t])[:, None]
        delta = reward[t] + disc * next_v - value[t]
        gae = delta + disc * lam * (~d) * gae
        adv[t] = gae
    return adv, adv + value
