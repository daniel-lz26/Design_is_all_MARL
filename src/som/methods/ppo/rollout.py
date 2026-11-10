"""Experience collection for the PPO method.

Training uses the evaluation harness's decision protocol: robots at a decision point act one at
a time, each with a fresh observation, so a claim is on the board before the next robot looks.
Several environments run side by side and every network call is batched over them.

Each robot's experience is a semi-MDP chain: a transition runs from one of its decisions to its
next one (or to the episode end). The team reward earned in between is credited to every open
transition, discounted by ``gamma ** (t - t_decision)``, and the bootstrap is discounted by
``gamma ** dt``. Rollouts always end on episode boundaries, so no chain is cut. A terminated
episode bootstraps 0; a truncated one (t_max) bootstraps V of the robot's final state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from som.cell_env.cell_env import CellEnv
from som.cell_env.metrics import compute

TERMINAL, BOOTSTRAP = -1, -2
ROOT = Path(__file__).resolve().parents[4]


def make_envs(scenarios: list[str], n: int) -> list[dict[str, CellEnv]]:
    """``n`` slots, each holding one CellEnv per training scenario."""
    return [
        {s: CellEnv(ROOT / "configs" / "cell" / f"{s}.yaml") for s in scenarios} for _ in range(n)
    ]


def critic_input(env: CellEnv, rid: str, obs: np.ndarray, central: bool) -> np.ndarray:
    return np.concatenate([obs, env.state()]) if central else obs


@dataclass
class Batch:
    """Flat transitions plus their chain links (``nxt``: index, TERMINAL or BOOTSTRAP)."""

    obs: list[np.ndarray] = field(default_factory=list)
    mask: list[np.ndarray] = field(default_factory=list)
    cin: list[np.ndarray] = field(default_factory=list)
    action: list[int] = field(default_factory=list)
    logp: list[float] = field(default_factory=list)
    value: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    t0: list[float] = field(default_factory=list)
    dt: list[float] = field(default_factory=list)
    nxt: list[int] = field(default_factory=list)
    boot_in: dict[int, np.ndarray] = field(default_factory=dict)
    episodes: list[dict[str, float]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.action)


class _Slot:
    def __init__(self, envs: dict[str, CellEnv], rng: np.random.Generator) -> None:
        self.envs, self.rng = envs, rng
        self.env: CellEnv
        self.queue: list[str] = []
        self.open: dict[str, int] = {}
        self.alive = False

    def start(self, seed_range: tuple[int, int], machine_rule: str) -> None:
        name = list(self.envs)[int(self.rng.integers(len(self.envs)))]
        self.env = self.envs[name]
        self.env.sim.machine_rule = machine_rule
        self.env.sim.machine_priority = None
        self.env.reset(int(self.rng.integers(seed_range[0], seed_range[1] + 1)))
        self.queue = list(self.env._pending)
        self.open = {}
        self.alive = True


Act = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
Value = Callable[[np.ndarray], np.ndarray]


def collect(
    slots: list[_Slot],
    act: Act,
    value: Value,
    n_decisions: int,
    gamma: float,
    central: bool,
    seed_range: tuple[int, int],
    machine_rule: str,
) -> Batch:
    """Run until at least ``n_decisions`` are recorded, then finish every running episode.

    ``act(obs, mask) -> (action, logp)`` and ``value(critic_in) -> V`` are batched."""
    b = Batch()
    for s in slots:
        s.start(seed_range, machine_rule)
    while any(s.alive for s in slots):
        ready: list[tuple[_Slot, str]] = []
        for s in slots:
            rid = _next_decider(s, b, gamma, value, central, len(b) < n_decisions)
            if rid is not None:
                ready.append((s, rid))
            elif s.alive and len(b) < n_decisions:
                s.start(seed_range, machine_rule)
                rid = _next_decider(s, b, gamma, value, central, True)
                if rid is not None:
                    ready.append((s, rid))
        if not ready:
            continue
        obs = np.stack([s.env.observe(r) for s, r in ready])
        mask = np.stack([s.env.mask(r) for s, r in ready])
        cin = np.stack(
            [critic_input(s.env, r, o, central) for (s, r), o in zip(ready, obs, strict=True)]
        )
        a, logp = act(obs, mask)
        v = value(cin)
        for i, (s, rid) in enumerate(ready):
            now = s.env.sim.now
            j = len(b)
            if rid in s.open:  # close this robot's previous transition
                k = s.open[rid]
                b.nxt[k], b.dt[k] = j, now - b.t0[k]
            b.obs.append(obs[i])
            b.mask.append(mask[i])
            b.cin.append(cin[i])
            b.action.append(int(a[i]))
            b.logp.append(float(logp[i]))
            b.value.append(float(v[i]))
            b.reward.append(0.0)
            b.t0.append(now)
            b.dt.append(0.0)
            b.nxt.append(TERMINAL)
            s.open[rid] = j
            s.env.apply(rid, int(a[i]))
    return b


def _next_decider(
    s: _Slot, b: Batch, gamma: float, value: Value, central: bool, keep_going: bool
) -> str | None:
    """Advance the slot to its next deciding robot. At an episode end close every open
    transition; return None (and mark the slot idle unless more episodes are wanted)."""
    if not s.alive:
        return None
    env, sim = s.env, s.env.sim
    while True:
        while s.queue:
            rid = s.queue.pop(0)
            if rid in sim.pending():
                return rid
        before = env.reward.snapshot(sim)
        s.queue = list(env.advance())
        r = float(sum(env.reward.step(sim, before).values()))
        for k in s.open.values():
            b.reward[k] += (gamma ** (sim.now - b.t0[k])) * r
        if s.queue:
            continue
        truncated = sim.log.truncated
        for rid, k in s.open.items():
            b.dt[k] = sim.now - b.t0[k]
            if truncated:
                b.nxt[k] = BOOTSTRAP
                o = env.observe(rid)
                b.boot_in[k] = critic_input(env, rid, o, central)
        m = compute(sim.log)
        b.episodes.append(
            {
                "scenario": sim.cfg.name,
                "on_time_pct": m.on_time_pct,
                "mean_tardiness": m.mean_tardiness,
                "truncated": float(truncated),
            }
        )
        s.open = {}
        s.alive = keep_going
        return None


def gae(b: Batch, boot_v: dict[int, float], gamma: float, lam: float) -> tuple[np.ndarray, ...]:
    """Semi-MDP GAE over the robot chains; returns (advantages, returns)."""
    n = len(b)
    v = np.asarray(b.value, np.float64)
    adv = np.zeros(n)
    for i in range(n - 1, -1, -1):
        disc = gamma ** b.dt[i]
        nx = b.nxt[i]
        if nx >= 0:
            delta = b.reward[i] + disc * v[nx] - v[i]
            adv[i] = delta + disc * lam * adv[nx]
        else:
            boot = boot_v.get(i, 0.0) if nx == BOOTSTRAP else 0.0
            adv[i] = b.reward[i] + disc * boot - v[i]
    return adv, adv + v


def demos(
    policy_factory: Callable[[], object],
    scenarios: list[str],
    n_decisions: int,
    seed_range: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(obs, mask, action) from running a decentralized controller through the harness
    protocol on the training mix, until ``n_decisions`` are recorded."""
    from som.cell_env.harness import run_episode

    envs = {s: CellEnv(ROOT / "configs" / "cell" / f"{s}.yaml") for s in scenarios}
    obs, masks, acts = [], [], []

    class Recorder:
        decentralized = True

        def __init__(self, inner: object) -> None:
            self.inner = inner
            self.name = inner.name  # type: ignore[attr-defined]
            self.machine_rule = getattr(inner, "machine_rule", None)

        def reset(self, info: dict) -> None:
            self.inner.reset(info)  # type: ignore[attr-defined]

        def act(self, o: dict, m: dict, gs: object = None) -> dict[str, int]:
            out = self.inner.act(o, m, gs)  # type: ignore[attr-defined]
            for a in o:
                obs.append(o[a])
                masks.append(m[a])
                acts.append(out[a])
            return out

    while len(acts) < n_decisions:
        name = scenarios[int(rng.integers(len(scenarios)))]
        run_episode(envs[name], Recorder(policy_factory()), int(rng.integers(*seed_range)))
    return np.stack(obs), np.stack(masks), np.asarray(acts)
