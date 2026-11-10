"""CellEnv: the cell DES as a PettingZoo ParallelEnv over decision points (spec §5, §7, §8).

Agents ``agv_1..agv_N``. Actions ``Discrete(K + 3)``: ``0..K-1`` claim the k-th task on the
board, ``K`` park, ``K+1`` wait, ``K+2`` NOOP. A robot at a decision point may not NOOP; every
other robot (busy, down, absent) may only NOOP. Masks are in ``infos[agent]["action_mask"]``
(and ``env.mask(agent)``). ``step`` returns the PettingZoo 5-tuple with terminations (all
orders delivered) kept apart from truncations (t_max), so trainers can bootstrap correctly.
One step is one decision point: every pending robot's action is applied in id order; if an
earlier robot already claimed the box a later one chose, the later robot waits (counted in
``infos[agent]["collisions"]``).

``observe`` / ``mask`` / ``apply`` / ``advance`` give the same env one robot at a time, with a
fresh observation per decision; the evaluation harness uses that path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from som.cell_env.des import REPO_ROOT, CellSim
from som.cell_env.obs import ObsEncoder
from som.cell_env.reward import CellReward
from som.config.cell_schema import CellLayoutCfg, CellRewardCfg, CellScenarioCfg
from som.config.loader import load_yaml


class CellEnv(ParallelEnv):  # type: ignore[misc]
    metadata: ClassVar[dict[str, Any]] = {"name": "cell_v1", "render_modes": []}

    def __init__(
        self,
        scenario: CellScenarioCfg | str | Path,
        layout: CellLayoutCfg | None = None,
        root: Path = REPO_ROOT,
        obs_mode: str = "board",
        reward: CellRewardCfg | None = None,
        strict: bool = True,
        n_robots: int | None = None,
    ) -> None:
        if not isinstance(scenario, CellScenarioCfg):
            p = Path(scenario)
            scenario = load_yaml(p if p.is_absolute() else root / p, CellScenarioCfg)
        if n_robots is not None:
            scenario = scenario.model_copy(update={"n_robots": n_robots})
        layout = layout or load_yaml(root / scenario.layout, CellLayoutCfg)
        self.sim = CellSim(scenario, layout, root)
        self.enc = ObsEncoder(self.sim, obs_mode)
        self.reward = CellReward(reward)
        self.strict = strict
        self.k = self.sim.k
        self.PARK, self.WAIT, self.NOOP = self.k, self.k + 1, self.k + 2
        self.possible_agents: list[str] = list(self.sim.robot_ids)
        self.agents: list[str] = []
        dim = self.enc.layout.dim
        self._obs_space = spaces.Box(-np.inf, np.inf, (dim,), np.float32)
        self._act_space = spaces.Discrete(self.k + 3)
        self._seed_rng = np.random.default_rng()
        self._pending: list[str] = []
        self._slots: dict[str, list[str]] = {}
        self.seed: int | None = None

    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._act_space

    # ------------------------------------------------------------------ one robot at a time

    def deciding(self) -> list[str]:
        return self.sim.pending()

    def mask(self, agent: str) -> np.ndarray:
        m = np.zeros(self.k + 3, dtype=np.int8)
        if agent in self.sim.pending():
            m[: self.k + 2] = self.sim.mask(agent)
        else:
            m[self.NOOP] = 1
        return m

    def observe(self, agent: str) -> np.ndarray:
        self._slots[agent] = [b.id for b in self.sim.tasks(agent)]
        return self.enc.encode(agent)

    def apply(self, agent: str, action: int) -> None:
        """Apply one robot's action against its current mask (raises if masked out)."""
        m = self.mask(agent)
        if not 0 <= action < len(m) or not m[action]:
            raise ValueError(f"{agent}: action {action} is masked out")
        if action != self.NOOP:
            self.sim.act(agent, action)

    def advance(self) -> list[str]:
        self._pending = self.sim.advance()
        return self._pending

    def global_state(self) -> dict[str, Any]:
        """Full simulator access, for centralized controllers only (never decentralized ones)."""
        return {"sim": self.sim}

    def state(self) -> np.ndarray:
        return self.enc.critic_state()

    def scenario_info(self) -> dict[str, Any]:
        return {
            "scenario": self.sim.cfg.name,
            "seed": self.seed,
            "n_robots": len(self.possible_agents),
            "k": self.k,
            "actions": {"park": self.PARK, "wait": self.WAIT, "noop": self.NOOP},
            "obs_layout": self.enc.layout,
            "agents": list(self.possible_agents),
            "stations": list(self.sim.stations),
        }

    # ------------------------------------------------------------------ PettingZoo API

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._seed_rng = np.random.default_rng(seed)
            self.seed = seed
        else:
            self.seed = int(self._seed_rng.integers(2**31 - 1))
        self._pending = self.sim.reset(self.seed)
        self.agents = list(self.possible_agents)
        return self._observe_all(), self._infos(0.0, {}, 0, 0)

    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
        sim = self.sim
        before = self.reward.snapshot(sim)
        collisions = invalid = 0
        for agent in self._pending:
            a = int(actions.get(agent, self.NOOP))
            if not 0 <= a < self.k + 3 or not self._masks[agent][a]:
                if self.strict:
                    raise ValueError(f"{agent}: action {a} is masked out")
                invalid += 1
                a = self.WAIT
            if agent not in sim.pending():
                continue  # an earlier robot took the last task it could have claimed
            if a < self.k:
                box = self._slots[agent][a]
                board = [b.id for b in sim.tasks(agent)]
                if box in board:
                    sim.act(agent, board.index(box))
                else:
                    collisions += 1
                    sim.act(agent, self.WAIT)
            elif sim.mask(agent)[a]:
                sim.act(agent, a)
            else:
                sim.act(agent, self.WAIT)
        agents = list(self.agents)
        self._pending = sim.advance()
        terms = self.reward.step(sim, before)
        r = float(sum(terms.values()))
        terminated = sim.done
        truncated = sim.log.truncated
        obs = self._observe_all()
        infos = self._infos(sim.now - before.t, terms, collisions, invalid)
        if terminated or truncated:
            self.agents = []
        return (
            obs,
            dict.fromkeys(agents, r),
            dict.fromkeys(agents, terminated),
            dict.fromkeys(agents, truncated),
            infos,
        )

    def _observe_all(self) -> dict[str, np.ndarray]:
        obs = self.enc.encode_all(self.possible_agents)
        self._masks = {a: self.mask(a) for a in self.possible_agents}
        for a in self.possible_agents:
            self._slots[a] = [b.id for b in self.sim.tasks(a)]
        return obs

    def _infos(
        self, dt: float, terms: dict[str, float], collisions: int, invalid: int
    ) -> dict[str, Any]:
        deciding = set(self._pending)
        return {
            a: {
                "action_mask": self._masks[a],
                "decision": a in deciding,
                "dt": dt,
                "reward_terms": terms,
                "collisions": collisions,
                "invalid_actions": invalid,
            }
            for a in self.possible_agents
        }
