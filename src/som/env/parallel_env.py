"""SomParallelEnv: semi-MDP over decision points, PettingZoo ParallelEnv API.

One ``step`` resolves every pending decision at the current time, then runs the simulator to the
next decision point. Agents with nothing to decide still get an observation, but their mask allows
only slot 0 (NOOP) and ``infos[agent]["decision"]`` is False so the trainer leaves them out of the
policy loss. ``infos[agent]["dt"]`` is the simulated time the step covered, for ``gamma ** dt``.

Robots that decide at the same instant are resolved one at a time in the core's order. If an
earlier robot already took the order a later robot chose, the later robot waits until the next
event (counted in ``infos[agent]["collisions"]``).

Seeds match ``som.eval.evaluate.run_one``: episode seed ``s`` builds the same orders and travel
noise as evaluating a policy on seed ``s``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from som.config.schema import EnvCfg, LayoutCfg, RewardCfg, ScenarioCfg
from som.core.decisions import WAIT, Candidate, Choice, DecisionRequest
from som.core.history import EpisodeLog
from som.core.ids import robot_id, role_of
from som.core.layout import Layout
from som.env.candidates import action_mask, slot_candidates
from som.env.global_state import agent_onehot, base_state, state_dim
from som.env.observations import ObsContext, encode, obs_dim
from som.env.rewards import RewardFn, open_orders
from som.sim.runtime import SimRuntime
from som.sim.scenario import make_runtime
from som.sim.travel import StraightLineTravel


def make_context(layout_cfg: LayoutCfg, scenario_cfg: ScenarioCfg, env_cfg: EnvCfg) -> ObsContext:
    return ObsContext(
        layout=Layout.from_cfg(layout_cfg),
        travel=StraightLineTravel.from_cfg(layout_cfg.travel_calibration),
        horizon=scenario_cfg.horizon,
        k_robot=env_cfg.k_robot,
        k_machine=env_cfg.k_machine,
        max_robots=env_cfg.max_robots,
    )


class SomParallelEnv(ParallelEnv):  # type: ignore[misc]
    metadata: ClassVar[dict[str, Any]] = {"name": "som_v1", "render_modes": []}

    def __init__(
        self,
        layout_cfg: LayoutCfg,
        scenario_cfg: ScenarioCfg,
        env_cfg: EnvCfg | None = None,
        reward_cfg: RewardCfg | None = None,
    ) -> None:
        self.layout_cfg = layout_cfg
        self.scenario_cfg = scenario_cfg
        self.env_cfg = env_cfg or EnvCfg()
        self.reward_fn = RewardFn(reward_cfg or RewardCfg())
        self.ctx = make_context(layout_cfg, scenario_cfg, self.env_cfg)
        robots = [robot_id(i) for i in range(self.env_cfg.n_robots)]
        machines = [s.machine for s in self.ctx.layout.machines if s.machine is not None]
        self.possible_agents: list[str] = [*robots, *machines]
        self.agents: list[str] = []
        self.slot = {a: i for i, a in enumerate(robots)} | {m: i for i, m in enumerate(machines)}
        self.k = {"robot": self.env_cfg.k_robot, "machine": self.env_cfg.k_machine}
        self.wait = {
            "robot": self.env_cfg.robot_wait_allowed,
            "machine": self.env_cfg.machine_wait_allowed,
        }
        self.state_dim = state_dim(self.ctx, self.env_cfg.max_orders)
        self._obs_spaces = {a: self._make_obs_space(role_of(a)) for a in self.possible_agents}
        self._act_spaces = {
            a: spaces.Discrete(self.k[role_of(a)] + 1) for a in self.possible_agents
        }
        self._seed_rng = np.random.default_rng()
        self.runtime: SimRuntime | None = None
        self.episode_seed: int | None = None
        self._reqs: list[DecisionRequest] = []
        self._slots: dict[str, list[Candidate]] = {}
        self._masks: dict[str, np.ndarray] = {}
        self.dropped_candidates = 0
        self.collisions = 0

    def _make_obs_space(self, role: str) -> spaces.Dict:
        n = obs_dim(self.ctx, role)
        return spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, (n,), np.float32),
                "action_mask": spaces.Box(0, 1, (self.k[role] + 1,), np.int8),
            }
        )

    def observation_space(self, agent: str) -> spaces.Dict:
        return self._obs_spaces[agent]

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._act_spaces[agent]

    # ------------------------------------------------------------------ episode

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._seed_rng = np.random.default_rng(seed)
            ep_seed = seed
        else:
            ep_seed = int(self._seed_rng.integers(2**31 - 1))
        self.runtime = make_runtime(self.layout_cfg, self.scenario_cfg, self.env_cfg, ep_seed)
        self.episode_seed = ep_seed
        self.agents = list(self.possible_agents)
        self.dropped_candidates = 0
        self.collisions = 0
        self._reqs = self.runtime.advance_to_next_decision()
        return self._observe(), self._infos(0.0)

    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
        rt = self._rt()
        core = rt.core
        t0 = rt.now
        before = open_orders(core)
        for req in self._reqs:
            agent = req.agent
            act = int(actions.get(agent, 0))
            if not self._masks[agent][act]:
                raise ValueError(f"{agent}: action {act} is masked out")
            fresh = next((r for r in core.pending_decisions(rt.now) if r.agent == agent), None)
            if fresh is None:
                continue  # an earlier robot took every order this one could carry
            rt.assign(agent, self._choice(agent, act, fresh))
        agents = list(self.agents)
        self._reqs = rt.advance_to_next_decision()
        rewards = self.reward_fn.compute(core, before, t0, rt.now, agents)
        terminated = core.done
        truncated = rt.truncated and not terminated
        obs = self._observe()
        infos = self._infos(rt.now - t0)
        terms = dict.fromkeys(agents, terminated)
        truncs = dict.fromkeys(agents, truncated)
        if terminated or truncated:
            self.agents = []
        return obs, rewards, terms, truncs, infos

    def _choice(self, agent: str, act: int, fresh: DecisionRequest) -> Choice:
        if act == 0:
            return WAIT
        c = self._slots[agent][act - 1]
        if c in fresh.candidates:
            return c
        self.collisions += 1
        return WAIT  # robots may always WAIT in the core; machines never collide

    # ------------------------------------------------------------------ views

    def _rt(self) -> SimRuntime:
        if self.runtime is None:
            raise RuntimeError("call reset() first")
        return self.runtime

    def _observe(self) -> dict[str, Any]:
        snap = self._rt().core.snapshot()
        by_agent = {r.agent: r for r in self._reqs}
        out: dict[str, Any] = {}
        for agent in self.possible_agents:
            role = role_of(agent)
            req = by_agent.get(agent)
            slots, dropped = slot_candidates(req, self.k[role])
            self.dropped_candidates += dropped
            self._slots[agent] = slots
            self._masks[agent] = action_mask(req, self.k[role], self.wait[role])
            out[agent] = {
                "observation": encode(self.ctx, snap, agent, role, slots),
                "action_mask": self._masks[agent],
            }
        return out

    def _infos(self, dt: float) -> dict[str, Any]:
        deciding = {r.agent for r in self._reqs}
        return {
            a: {"decision": a in deciding, "dt": dt, "collisions": self.collisions}
            for a in self.possible_agents
        }

    def state(self) -> np.ndarray:
        snap = self._rt().core.snapshot()
        return base_state(self.ctx, snap, self.env_cfg.max_orders)

    def agent_states(self) -> dict[str, np.ndarray]:
        """Agent-specific global state for the centralized critic."""
        base = self.state()
        return {
            a: np.concatenate([base, agent_onehot(self.ctx, role_of(a), self.slot[a])])
            for a in self.possible_agents
        }

    def episode_log(self) -> EpisodeLog:
        log = self._rt().log()
        log.meta.update(scenario=self.scenario_cfg.name, seed=self.episode_seed)
        return log
