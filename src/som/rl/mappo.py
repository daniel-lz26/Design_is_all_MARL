"""MAPPO trainer; critic_mode 'central' (MAPPO) or 'local' (IPPO ablation).

One actor and one critic per role, shared by every agent of that role. Only decision samples
(``infos["decision"]``) enter the policy loss; every sample trains the critic. Advantages use
semi-MDP GAE (``gamma ** dt``) and are normalized per role; value targets are normalized with
running statistics.

Model selection uses the training scenario on the fixed eval seeds (disjoint from training
seeds). Other eval scenarios, such as ``abstract_10``, are logged but never used for selection.
"""

from __future__ import annotations

import csv
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from som.config.schema import LayoutCfg, ScenarioCfg, TrainCfg
from som.env.observations import obs_dim
from som.env.parallel_env import SomParallelEnv
from som.eval.evaluate import run_one
from som.eval.metrics import compute_metrics
from som.policies.learned import LearnedPolicy
from som.rl.artifact import export_artifact
from som.rl.buffer import RoleBuffer, RolloutBuffer, semi_mdp_gae
from som.rl.networks import Critic, RoleActor
from som.rl.normalization import RunningMeanStd
from som.rl.vec import ROLES, SyncVecEnv, VecObs

EVAL_FIELDS = [
    "update",
    "env_steps",
    "scenario",
    "on_time_rate",
    "mean_tardiness",
    "makespan",
    "delivered",
]


@dataclass
class EvalResult:
    scenario: str
    on_time_rate: float
    mean_tardiness: float
    makespan: float
    delivered: float


class MAPPOTrainer:
    def __init__(
        self,
        cfg: TrainCfg,
        layout_cfg: LayoutCfg,
        scenario_cfg: ScenarioCfg,
        eval_scenarios: list[ScenarioCfg],
        run_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.layout_cfg = layout_cfg
        self.scenario_cfg = scenario_cfg
        self.eval_scenarios = [scenario_cfg, *eval_scenarios]
        self.run_dir = run_dir
        torch.manual_seed(cfg.seed)
        envs = [
            SomParallelEnv(layout_cfg, scenario_cfg, cfg.env, cfg.reward) for _ in range(cfg.n_envs)
        ]
        self.vec = SyncVecEnv(envs, seed=cfg.seed)
        self.ctx = envs[0].ctx
        self.central = cfg.critic_mode == "central"
        self.obs_dim = {r: obs_dim(self.ctx, r) for r in ROLES}
        self.n_act = {"robot": cfg.env.k_robot + 1, "machine": cfg.env.k_machine + 1}
        critic_dim = {r: envs[0].state_dim if self.central else self.obs_dim[r] for r in ROLES}
        self.critic_dim = critic_dim
        self.actors = {r: RoleActor(self.obs_dim[r], self.n_act[r], cfg.hidden) for r in ROLES}
        self.critics = {r: Critic(critic_dim[r], cfg.hidden) for r in ROLES}
        self.vnorm = {r: RunningMeanStd() for r in ROLES}
        params = [
            p for m in [*self.actors.values(), *self.critics.values()] for p in m.parameters()
        ]
        self.params = params
        self.opt = torch.optim.Adam(params, lr=cfg.lr, eps=1e-5)
        self.recent: deque[dict[str, Any]] = deque(maxlen=50)
        self.env_steps = 0
        self.episodes = 0
        self.best: tuple[float, float] | None = None  # (on_time_rate, -mean_tardiness)

    # ------------------------------------------------------------------ rollout

    def _critic_in(self, obs: VecObs, role: str) -> np.ndarray:
        return obs.state[role] if self.central else obs.obs[role]

    @torch.no_grad()
    def _values(self, role: str, x: np.ndarray) -> np.ndarray:
        shape = x.shape[:-1]
        v = self.critics[role](torch.as_tensor(x).reshape(-1, x.shape[-1])).numpy()
        return self.vnorm[role].denormalize(v).reshape(shape)

    @torch.no_grad()
    def collect(self, obs: VecObs) -> tuple[RolloutBuffer, VecObs]:
        cfg = self.cfg
        t_len, e = cfg.rollout_steps, self.vec.n_envs
        n = {r: len(self.vec.agents[r]) for r in ROLES}
        buf = RolloutBuffer(
            roles={
                r: RoleBuffer.empty(
                    t_len, e, n[r], self.obs_dim[r], self.n_act[r], self.critic_dim[r]
                )
                for r in ROLES
            },
            dt=np.zeros((t_len, e), np.float32),
            done=np.zeros((t_len, e), bool),
            truncated=np.zeros((t_len, e), bool),
        )
        for t in range(t_len):
            actions = {}
            for r in ROLES:
                rb = buf.roles[r]
                o = torch.as_tensor(obs.obs[r]).reshape(e * n[r], -1)
                m = torch.as_tensor(obs.mask[r]).reshape(e * n[r], -1)
                dist = self.actors[r].dist(o, m)
                a = dist.sample()
                rb.obs[t], rb.mask[t] = obs.obs[r], obs.mask[r]
                rb.critic_in[t] = self._critic_in(obs, r)
                rb.action[t] = a.reshape(e, n[r]).numpy()
                rb.logp[t] = dist.log_prob(a).reshape(e, n[r]).numpy()
                rb.value[t] = self._values(r, rb.critic_in[t])
                rb.decision[t] = obs.decision[r]
                actions[r] = rb.action[t]
            obs, st = self.vec.step(actions)
            buf.dt[t], buf.done[t], buf.truncated[t] = st.dt, st.done, st.truncated
            for r in ROLES:
                rb = buf.roles[r]
                rb.reward[t] = st.reward[r]
                if st.truncated.any():
                    final = st.final_state[r] if self.central else st.final_obs[r]
                    rb.trunc_value[t] = self._values(r, final)
            for log in st.episodes:
                self.recent.append(compute_metrics(log).as_row())
            self.episodes += len(st.episodes)
        self.env_steps += t_len * e
        for r in ROLES:
            rb = buf.roles[r]
            last = self._values(r, self._critic_in(obs, r))
            rb.adv, rb.ret = semi_mdp_gae(
                rb.reward,
                rb.value,
                last,
                rb.trunc_value,
                buf.dt,
                buf.done,
                buf.truncated,
                cfg.gamma,
                cfg.gae_lambda,
            )
        return buf, obs

    # ------------------------------------------------------------------ update

    def update(self, buf: RolloutBuffer) -> dict[str, float]:
        cfg = self.cfg
        flat: dict[str, dict[str, torch.Tensor]] = {}
        for r in ROLES:
            rb = buf.roles[r]
            assert rb.adv is not None and rb.ret is not None
            self.vnorm[r].update(rb.ret)
            dec = rb.decision.reshape(-1)
            adv = rb.adv.reshape(-1)
            if dec.any():
                mu, sd = adv[dec].mean(), adv[dec].std() + 1e-8
                adv = (adv - mu) / sd
            flat[r] = {
                "obs": torch.as_tensor(rb.obs.reshape(-1, rb.obs.shape[-1])),
                "mask": torch.as_tensor(rb.mask.reshape(-1, rb.mask.shape[-1])),
                "critic_in": torch.as_tensor(rb.critic_in.reshape(-1, rb.critic_in.shape[-1])),
                "action": torch.as_tensor(rb.action.reshape(-1)),
                "logp": torch.as_tensor(rb.logp.reshape(-1)),
                "adv": torch.as_tensor(adv, dtype=torch.float32),
                "ret": torch.as_tensor(
                    self.vnorm[r].normalize(rb.ret.reshape(-1)), dtype=torch.float32
                ),
                "dec": torch.as_tensor(dec),
            }
        stats: dict[str, list[float]] = {}

        def log(k: str, v: float) -> None:
            stats.setdefault(k, []).append(v)

        for _ in range(cfg.update_epochs):
            perms = {r: torch.randperm(len(flat[r]["action"])) for r in ROLES}
            for i in range(cfg.minibatches):
                loss = torch.zeros(())
                for r in ROLES:
                    idx = perms[r].chunk(cfg.minibatches)[i]
                    mb = {k: v[idx] for k, v in flat[r].items()}
                    v = self.critics[r](mb["critic_in"])
                    v_loss = 0.5 * ((v - mb["ret"]) ** 2).mean()
                    loss = loss + cfg.value_coef * v_loss
                    log(f"value_loss/{r}", v_loss.item())
                    d = mb["dec"]
                    if not d.any():
                        continue
                    dist = self.actors[r].dist(mb["obs"][d], mb["mask"][d])
                    logp = dist.log_prob(mb["action"][d])
                    ratio = torch.exp(logp - mb["logp"][d])
                    adv = mb["adv"][d]
                    pg = -torch.min(
                        ratio * adv, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv
                    ).mean()
                    ent = dist.entropy().mean()
                    loss = loss + pg - cfg.entropy_coef * ent
                    with torch.no_grad():
                        log(f"entropy/{r}", float(ent))
                        log(f"kl/{r}", float(((ratio - 1) - torch.log(ratio)).mean()))
                        log(f"clipfrac/{r}", float(((ratio - 1).abs() > cfg.clip).float().mean()))
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
                self.opt.step()
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        for r in ROLES:
            out[f"decision_frac/{r}"] = float(flat[r]["dec"].float().mean())
        return out

    # ------------------------------------------------------------------ eval / export

    def policy(self) -> LearnedPolicy:
        return LearnedPolicy(self.actors, self.ctx, self.cfg.env)

    def evaluate(self) -> list[EvalResult]:
        pol = self.policy()
        out = []
        for sc in self.eval_scenarios:
            ms = [
                compute_metrics(run_one(self.layout_cfg, sc, self.cfg.env, pol, s))
                for s in self.cfg.eval_seeds
            ]
            out.append(
                EvalResult(
                    sc.name,
                    float(np.mean([m.on_time_rate for m in ms])),
                    float(np.mean([m.mean_tardiness for m in ms])),
                    float(np.mean([m.makespan for m in ms])),
                    float(np.mean([m.delivered for m in ms])),
                )
            )
        return out

    def meta(self, update: int) -> dict[str, Any]:
        return {
            "K_robot": self.cfg.env.k_robot,
            "K_machine": self.cfg.env.k_machine,
            "net": {"hidden": list(self.cfg.hidden), "obs_dim": self.obs_dim},
            "layout": self.layout_cfg.name,
            "trained_on": self.scenario_cfg.name,
            "env": self.cfg.env.model_dump(mode="json"),
            "horizon": self.scenario_cfg.horizon,
            "critic_mode": self.cfg.critic_mode,
            "run": self.cfg.name,
            "update": update,
            "env_steps": self.env_steps,
        }

    def save_checkpoint(self, path: Path, update: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "update": update,
                "env_steps": self.env_steps,
                "actors": {r: a.state_dict() for r, a in self.actors.items()},
                "critics": {r: c.state_dict() for r, c in self.critics.items()},
                "vnorm": {r: v.state_dict() for r, v in self.vnorm.items()},
                "opt": self.opt.state_dict(),
            },
            path,
        )

    # ------------------------------------------------------------------ loop

    def train(self, on_log: Callable[[str], None] = print) -> dict[str, Any]:
        cfg = self.cfg
        n_updates = max(1, cfg.total_steps // (cfg.rollout_steps * cfg.n_envs))
        writer = _tb_writer(self.run_dir / "tb")
        progress = _CsvLog(self.run_dir / "progress.csv")
        evals = _CsvLog(self.run_dir / "eval.csv", EVAL_FIELDS)
        obs = self.vec.reset()
        t_start = time.time()
        best_row: dict[str, Any] = {}
        for u in range(1, n_updates + 1):
            buf, obs = self.collect(obs)
            stats = self.update(buf)
            row: dict[str, Any] = {
                "update": u,
                "env_steps": self.env_steps,
                "episodes": self.episodes,
                "sps": self.env_steps / (time.time() - t_start),
            }
            if self.recent:
                for k in ("on_time_rate", "mean_tardiness", "makespan", "delivered"):
                    row[f"train/{k}"] = float(np.mean([m[k] for m in self.recent]))
            row.update(stats)
            progress.write(row)
            for k, v in row.items():
                if writer is not None and k not in ("update", "env_steps"):
                    writer.add_scalar(k, v, self.env_steps)
            if u % cfg.eval_every == 0 or u == n_updates:
                results = self.evaluate()
                for res in results:
                    evals.write({"update": u, "env_steps": self.env_steps, **res.__dict__})
                    if writer is not None:
                        writer.add_scalar(
                            f"eval/{res.scenario}/on_time_rate", res.on_time_rate, self.env_steps
                        )
                        writer.add_scalar(
                            f"eval/{res.scenario}/mean_tardiness",
                            res.mean_tardiness,
                            self.env_steps,
                        )
                sel = results[0]
                key = (sel.on_time_rate, -sel.mean_tardiness)
                tag = ""
                if self.best is None or key > self.best:
                    self.best = key
                    export_artifact(self.run_dir / "policy", self.actors, self.meta(u))
                    best_row = {"update": u, **{r.scenario: r.__dict__ for r in results}}
                    tag = "  * new best"
                self.save_checkpoint(self.run_dir / "checkpoints" / "last.pt", u)
                on_log(
                    f"update {u}/{n_updates}  steps {self.env_steps}  "
                    + "  ".join(
                        f"{r.scenario} on-time {100 * r.on_time_rate:.1f}%" for r in results
                    )
                    + tag
                )
        export_artifact(self.run_dir / "policy_final", self.actors, self.meta(n_updates))
        if writer is not None:
            writer.close()
        return best_row


class _CsvLog:
    """Append rows to a csv; the header comes from the first row unless given."""

    def __init__(self, path: Path, fields: list[str] | None = None) -> None:
        self.path = path
        self.fields = fields
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict[str, Any]) -> None:
        new = not self.path.exists()
        if self.fields is None:
            self.fields = list(row)
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def _tb_writer(path: Path) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:  # tensorboard is optional for tests
        return None
    return SummaryWriter(str(path))
