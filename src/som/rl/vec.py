"""SyncVecEnv over several SomParallelEnv copies.

Arrays are grouped by role, shaped ``[E, n_role, ...]`` with agents in each env's
``possible_agents`` order. Finished envs reset immediately; the step result says which ones ended
and keeps the final critic state of truncated episodes so the trainer can bootstrap from it.
Training episode seeds start at ``TRAIN_SEED_FLOOR``, away from the small eval seeds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from som.core.history import EpisodeLog
from som.core.ids import role_of
from som.env.parallel_env import SomParallelEnv

ROLES = ("robot", "machine")
TRAIN_SEED_FLOOR = 1_000_000


@dataclass
class VecObs:
    obs: dict[str, np.ndarray]  # [E, n, obs_dim]
    mask: dict[str, np.ndarray]  # [E, n, K + 1]
    decision: dict[str, np.ndarray]  # [E, n] bool
    state: dict[str, np.ndarray]  # [E, n, state_dim]


@dataclass
class VecStep:
    reward: dict[str, np.ndarray]  # [E, n]
    dt: np.ndarray  # [E]
    done: np.ndarray  # [E] bool: the episode ended on this step (env already reset)
    truncated: np.ndarray  # [E] bool
    final_state: dict[str, np.ndarray]  # [E, n, state_dim]; meaningful where truncated
    final_obs: dict[str, np.ndarray]  # [E, n, obs_dim]; same, for the local (IPPO) critic
    episodes: list[EpisodeLog] = field(default_factory=list)


class SyncVecEnv:
    def __init__(self, envs: Sequence[SomParallelEnv], seed: int) -> None:
        self.envs = list(envs)
        self.rng = np.random.default_rng(seed)
        agents = self.envs[0].possible_agents
        self.agents = {r: [a for a in agents if role_of(a) == r] for r in ROLES}
        self._last: list[tuple[dict, dict]] = []

    @property
    def n_envs(self) -> int:
        return len(self.envs)

    def _seed(self) -> int:
        return int(self.rng.integers(TRAIN_SEED_FLOOR, 2**31 - 1))

    def reset(self) -> VecObs:
        self._last = [env.reset(seed=self._seed()) for env in self.envs]
        return self._collect()

    def _collect(self) -> VecObs:
        out = VecObs({}, {}, {}, {})
        states = [env.agent_states() for env in self.envs]
        for role, names in self.agents.items():
            out.obs[role] = np.stack(
                [np.stack([o[a]["observation"] for a in names]) for o, _ in self._last]
            )
            out.mask[role] = np.stack(
                [np.stack([o[a]["action_mask"] for a in names]) for o, _ in self._last]
            )
            out.decision[role] = np.array(
                [[i[a]["decision"] for a in names] for _, i in self._last], dtype=bool
            )
            out.state[role] = np.stack([np.stack([s[a] for a in names]) for s in states])
        return out

    def step(self, actions: dict[str, np.ndarray]) -> tuple[VecObs, VecStep]:
        e = self.n_envs
        res = VecStep(
            reward={r: np.zeros((e, len(n)), np.float32) for r, n in self.agents.items()},
            dt=np.zeros(e, np.float32),
            done=np.zeros(e, bool),
            truncated=np.zeros(e, bool),
            final_state={},
            final_obs={},
        )
        finals: list[tuple[dict[str, np.ndarray], dict] | None] = [None] * e
        for i, env in enumerate(self.envs):
            acts = {
                a: int(actions[r][i, j]) for r, ns in self.agents.items() for j, a in enumerate(ns)
            }
            obs, rew, term, trunc, infos = env.step(acts)
            first = env.possible_agents[0]
            res.dt[i] = infos[first]["dt"]
            for r, ns in self.agents.items():
                res.reward[r][i] = [rew[a] for a in ns]
            if term[first] or trunc[first]:
                res.done[i] = True
                res.truncated[i] = bool(trunc[first])
                if trunc[first]:
                    finals[i] = (env.agent_states(), obs)
                res.episodes.append(env.episode_log())
                obs, infos = env.reset(seed=self._seed())
            self._last[i] = (obs, infos)
        nxt = self._collect()
        for r, ns in self.agents.items():
            fs, fo = nxt.state[r].copy(), nxt.obs[r].copy()
            for i, f in enumerate(finals):
                if f is not None:
                    fs[i] = np.stack([f[0][a] for a in ns])
                    fo[i] = np.stack([f[1][a]["observation"] for a in ns])
            res.final_state[r], res.final_obs[r] = fs, fo
        return nxt, res
