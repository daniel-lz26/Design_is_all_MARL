"""IPPO / MAPPO trainer for the cell env, with an optional behaviour-cloning warm-start.

1. **Demonstrations** from a family-A rule (default BOTTLENECK/SPT) on the training mix. They fit
   the frozen observation normalization and, with ``bc_decisions > 0``, warm-start the actor by
   cross-entropy imitation. The critic's input normalization is fitted on the first rollout.
2. **PPO**: clipped surrogate, semi-MDP GAE, value targets normalized with running statistics,
   linear decay of learning rate and entropy coefficient, early epoch stop on a KL target. The
   first ``critic_warmup`` rollouts train only the critic, so a cloned actor is not pushed
   around by an untrained value function.
3. **Selection**: every ``eval_every`` decisions the greedy actor runs on the validation seeds
   of every training scenario; the best mean on-time % (then the lowest tardiness) is kept as
   ``best/actor.pt``.

Budget (spec §11.2): demonstration decisions and PPO decisions both count toward
``budget``; validation runs do not (they are selection, not training).
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from som.cell_env.cell_env import CellEnv
from som.cell_env.harness import run_episode
from som.cell_env.metrics import compute
from som.cell_env.obs import CELL_OBS_VERSION
from som.methods.ppo.model import CandidateActor, Critic
from som.methods.ppo.policy import ACTOR_FILE, PPOPolicy, save_actor
from som.methods.ppo.rollout import ROOT, _Slot, collect, demos, gae, make_envs
from som.rl.normalization import RunningMeanStd

TRAIN_MIX = ["cell_random_10", "cell_random_10_disrupt", "cell_10_disrupt"]


@dataclass
class PPOConfig:
    name: str = "MAPPO_s0"
    critic: str = "central"  # "central" (MAPPO) or "local" (IPPO)
    seed: int = 0
    scenarios: list[str] = field(default_factory=lambda: list(TRAIN_MIX))
    machine_rule: str = "SPT"
    # park/wait masked out while a task is claimable (fix for order abandonment, 2026-11-11)
    work_conserving: bool = True
    budget: int = 1_500_000  # env decisions: demonstrations + PPO (spec §11.2)
    # behaviour cloning
    teacher: str = "BOTTLENECK/SPT"
    bc_decisions: int = 50_000  # 0: no warm-start (demos are still used to fit normalization)
    norm_decisions: int = 10_000  # demos collected when bc_decisions == 0
    bc_epochs: int = 30
    bc_lr: float = 1e-3
    # PPO (spec §11.3 starting values)
    lr: float = 3e-4
    lr_end: float = 3e-5
    ent: float = 0.01
    ent_end: float = 0.001
    gamma: float = 0.995  # per second (semi-MDP)
    lam: float = 0.95
    clip: float = 0.2
    rollout: int = 4096
    epochs: int = 4
    minibatch: int = 512
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    critic_warmup: int = 2  # rollouts that train the critic only
    hidden: int = 128
    n_envs: int = 8
    eval_every: int = 20_000
    threads: int = 1

    @property
    def method(self) -> str:
        return "MAPPO" if self.critic == "central" else "IPPO"


class PPOTrainer:
    def __init__(self, cfg: PPOConfig, run_dir: Path) -> None:
        if cfg.critic not in ("central", "local"):
            raise ValueError("critic must be 'central' (MAPPO) or 'local' (IPPO)")
        self.cfg, self.run_dir = cfg, run_dir
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(cfg.threads)
        self.rng = np.random.default_rng(cfg.seed)
        self.central = cfg.critic == "central"
        probe = CellEnv(ROOT / "configs" / "cell" / f"{cfg.scenarios[0]}.yaml")
        probe.reset(seed=0)
        self.layout = probe.enc.layout
        cdim = self.layout.dim + (len(probe.state()) if self.central else 0)
        self.actor = CandidateActor(self.layout, cfg.hidden, cfg.work_conserving)
        self.critic = Critic(cdim)
        self.vnorm = RunningMeanStd()
        self.opt = torch.optim.Adam(
            [*self.actor.parameters(), *self.critic.parameters()], lr=cfg.lr, eps=1e-5
        )
        self.slots = [_Slot(e, self.rng) for e in make_envs(cfg.scenarios, cfg.n_envs)]
        lo, hi = probe.sim.cfg.seeds.train
        self.train_seeds = (int(lo), int(hi))
        self.val_seeds = list(range(*_inclusive(probe.sim.cfg.seeds.val)))
        self.used = 0  # decisions spent from the budget
        self.best: tuple[float, float] | None = None
        self.critic_fitted = False
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ helpers

    def meta(self, **extra: Any) -> dict[str, Any]:
        c = self.cfg
        return {
            "name": c.name,
            "method": c.method,
            "critic": c.critic,
            "seed": c.seed,
            "machine_rule": c.machine_rule,
            "obs_version": CELL_OBS_VERSION,
            "layout": asdict(self.layout),
            "hidden": c.hidden,
            "work_conserving": c.work_conserving,
            "trained_on": c.scenarios,
            "teacher": c.teacher if c.bc_decisions else None,
            "decisions_used": self.used,
            **extra,
        }

    @torch.no_grad()
    def _act(self, obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self.actor.dist(torch.as_tensor(obs), torch.as_tensor(mask))
        a = d.sample()
        return a.numpy(), d.log_prob(a).numpy()

    @torch.no_grad()
    def _value(self, cin: np.ndarray) -> np.ndarray:
        v = self.critic(torch.as_tensor(cin, dtype=torch.float32)).numpy()
        return self.vnorm.denormalize(v)

    def validate(self, seeds: list[int] | None = None) -> dict[str, dict[str, float]]:
        pol = PPOPolicy(self.actor, self.meta())
        out = {}
        for s in self.cfg.scenarios:
            env = CellEnv(ROOT / "configs" / "cell" / f"{s}.yaml")
            ms = [compute(run_episode(env, pol, sd)) for sd in seeds or self.val_seeds]
            out[s] = {
                "on_time_pct": float(np.mean([m.on_time_pct for m in ms])),
                "mean_tardiness": float(np.mean([m.mean_tardiness for m in ms])),
            }
        self.actor.train()
        return out

    # ------------------------------------------------------------------ behaviour cloning

    def warm_start(self, log: Any = print) -> dict[str, float]:
        from som.policies.registry import make

        c = self.cfg
        n = c.bc_decisions or c.norm_decisions
        t0 = time.time()
        obs, mask, act = demos(lambda: make(c.teacher), c.scenarios, n, self.train_seeds, self.rng)
        self.used += len(act)
        self.actor.norm.fit(obs)
        out: dict[str, float] = {"demo_decisions": float(len(act))}
        if not c.bc_decisions:
            return out
        o, m, a = (torch.as_tensor(x) for x in (obs, mask, act))
        opt = torch.optim.Adam(self.actor.parameters(), lr=c.bc_lr)
        for ep in range(c.bc_epochs):
            perm = torch.randperm(len(a))
            losses, hits = [], []
            for idx in perm.split(512):
                logits = self.actor.logits(o[idx], m[idx])
                loss = torch.nn.functional.cross_entropy(logits, a[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
                hits.append((logits.argmax(-1) == a[idx]).float().mean().item())
            if ep == c.bc_epochs - 1 or ep % 10 == 0:
                log(f"bc epoch {ep + 1}/{c.bc_epochs}  loss {np.mean(losses):.4f}  "
                    f"match {100 * np.mean(hits):.2f}%")  # fmt: skip
        out.update(bc_loss=float(np.mean(losses)), bc_match=float(np.mean(hits)))
        out["bc_seconds"] = time.time() - t0
        return out

    # ------------------------------------------------------------------ PPO

    def update(self, b: Any, frac: float, actor_on: bool) -> dict[str, float]:
        c = self.cfg
        if not self.critic_fitted:
            self.critic.norm.fit(np.stack(b.cin))
            self.critic_fitted = True
        boot = {}
        if b.boot_in:
            keys = list(b.boot_in)
            vals = self._value(np.stack([b.boot_in[k] for k in keys]))
            boot = dict(zip(keys, vals.tolist(), strict=True))
        adv, ret = gae(b, boot, c.gamma, c.lam)
        self.vnorm.update(ret)
        lr = c.lr + (c.lr_end - c.lr) * frac
        ent_coef = c.ent + (c.ent_end - c.ent) * frac
        for g in self.opt.param_groups:
            g["lr"] = lr
        obs = torch.as_tensor(np.stack(b.obs))
        mask = torch.as_tensor(np.stack(b.mask))
        cin = torch.as_tensor(np.stack(b.cin))
        act = torch.as_tensor(b.action)
        old = torch.as_tensor(b.logp)
        adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), dtype=torch.float32)
        ret_t = torch.as_tensor(self.vnorm.normalize(ret), dtype=torch.float32)
        stats: dict[str, list[float]] = {}
        stop = False
        for _ in range(c.epochs):
            for idx in torch.randperm(len(act)).split(c.minibatch):
                v_loss = 0.5 * ((self.critic(cin[idx]) - ret_t[idx]) ** 2).mean()
                loss = c.value_coef * v_loss
                st = {"value_loss": v_loss.item()}
                if actor_on:
                    d = self.actor.dist(obs[idx], mask[idx])
                    logp = d.log_prob(act[idx])
                    ratio = torch.exp(logp - old[idx])
                    a_ = adv_t[idx]
                    pg = -torch.min(ratio * a_, ratio.clamp(1 - c.clip, 1 + c.clip) * a_).mean()
                    ent = d.entropy().mean()
                    loss = loss + pg - ent_coef * ent
                    with torch.no_grad():
                        kl = float(((ratio - 1) - torch.log(ratio)).mean())
                    st.update(
                        policy_loss=pg.item(),
                        entropy=ent.item(),
                        kl=kl,
                        clipfrac=float(((ratio - 1).abs() > c.clip).float().mean()),
                    )
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [*self.actor.parameters(), *self.critic.parameters()], c.max_grad_norm
                )
                self.opt.step()
                for k, v in st.items():
                    stats.setdefault(k, []).append(v)
                if actor_on and st["kl"] > 1.5 * c.target_kl:
                    stop = True
                    break
            if stop:
                break
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out.update(lr=lr, ent_coef=ent_coef, early_stop=float(stop))
        return out

    def train(self, log: Any = print) -> dict[str, Any]:
        c = self.cfg
        t_start = time.time()
        progress = _Csv(self.run_dir / "progress.csv")
        evals = _Csv(self.run_dir / "eval.csv")
        ws = self.warm_start(log)
        log(f"warm start: {ws}")
        self._evaluate(evals, 0, log, time.time() - t_start)
        ppo_budget = c.budget - self.used
        rollouts, next_eval = 0, self.used + c.eval_every
        while self.used < c.budget:
            frac = 1 - (c.budget - self.used) / max(ppo_budget, 1)
            want = min(c.rollout, c.budget - self.used)
            b = collect(self.slots, self._act, self._value, want, c.gamma, self.central,
                        self.train_seeds, c.machine_rule)  # fmt: skip
            self.used += len(b)
            rollouts += 1
            st = self.update(b, frac, actor_on=rollouts > c.critic_warmup)
            eps = b.episodes
            row = {
                "decisions": self.used,
                "rollout": rollouts,
                "seconds": time.time() - t_start,
                "episodes": len(eps),
                "train_on_time_pct": float(np.mean([e["on_time_pct"] for e in eps])),
                "train_tardiness": float(np.mean([e["mean_tardiness"] for e in eps])),
                "truncated": float(np.mean([e["truncated"] for e in eps])),
                **dict.fromkeys(("policy_loss", "entropy", "kl", "clipfrac"), float("nan")),
                **st,
            }
            progress.write(row)
            if self.used >= next_eval or self.used >= c.budget:
                next_eval += c.eval_every
                self._evaluate(evals, rollouts, log, time.time() - t_start, row)
        save_actor(self.run_dir / "final" / ACTOR_FILE, self.actor, self.meta(final=True))
        torch.save(
            {"critic": self.critic.state_dict(), "vnorm": self.vnorm.state_dict()},
            self.run_dir / "final" / "critic.pt",
        )
        summary = {
            "decisions": self.used,
            "seconds": time.time() - t_start,
            "best_val": self.best,
            **ws,
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _evaluate(
        self, evals: _Csv, rollout: int, log: Any, seconds: float, row: dict | None = None
    ) -> None:
        res = self.validate()
        on = float(np.mean([r["on_time_pct"] for r in res.values()]))
        tard = float(np.mean([r["mean_tardiness"] for r in res.values()]))
        evals.write(
            {
                "decisions": self.used,
                "rollout": rollout,
                "seconds": seconds,
                "val_on_time_pct": on,
                "val_tardiness": tard,
                **{f"{s}/on_time_pct": r["on_time_pct"] for s, r in res.items()},
            }
        )
        key = (on, -tard)
        tag = ""
        if self.best is None or key > self.best:
            self.best = key
            save_actor(self.run_dir / "best" / ACTOR_FILE, self.actor, self.meta(val=res))
            tag = "  * best"
        train = f"  train {row['train_on_time_pct']:.1f}%" if row else ""
        log(
            f"[{self.used:>8} dec  {seconds / 60:5.1f} min]{train}  val {on:.1f}% "
            + " ".join(f"{s.replace('cell_', '')} {r['on_time_pct']:.1f}" for s, r in res.items())
            + tag
        )


def _inclusive(r: tuple[int, int]) -> tuple[int, int]:
    return int(r[0]), int(r[1]) + 1


class _Csv:
    def __init__(self, path: Path) -> None:
        self.path, self.fields = path, None

    def write(self, row: dict[str, Any]) -> None:
        new = self.fields is None
        if new:
            self.fields = list(row)
        with self.path.open("a" if not new else "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
