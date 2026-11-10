from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from som.cell_env.cell_env import CellEnv
from som.cell_env.harness import run_episode
from som.cell_env.metrics import compute
from som.methods.ppo.model import CandidateActor
from som.methods.ppo.rollout import BOOTSTRAP, TERMINAL, Batch, gae
from som.methods.ppo.trainer import PPOConfig, PPOTrainer
from som.policies.registry import family, make


def _obs(n: int = 6) -> tuple[CellEnv, np.ndarray, np.ndarray]:
    env = CellEnv("configs/cell/cell_random_10.yaml")
    env.reset(seed=20000)
    obs, masks = [], []
    while len(obs) < n:
        for rid in env.deciding():
            obs.append(env.observe(rid))
            masks.append(env.mask(rid))
            env.apply(rid, int(np.flatnonzero(env.mask(rid))[0]))
        env.advance()
    return env, np.stack(obs[:n]), np.stack(masks[:n])


def test_actor_scores_do_not_depend_on_task_or_robot_order() -> None:
    env, obs, mask = _obs()
    lay = env.enc.layout
    actor = CandidateActor(lay)
    torch.nn.init.normal_(actor.score[-1].weight)  # break the near-zero output init
    o, m = torch.as_tensor(obs), torch.as_tensor(mask)
    base = actor.logits(o, m)
    tperm = torch.tensor([3, 0, 7, 1, 6, 2, 5, 4])
    o2 = o.clone()
    tasks = o2[:, lay.tasks_at : lay.stations_at].reshape(len(o), lay.k, -1)
    o2[:, lay.tasks_at : lay.stations_at] = tasks[:, tperm].reshape(len(o), -1)
    m2 = m.clone()
    m2[:, : lay.k] = m[:, tperm]
    out = actor.logits(o2, m2)
    assert torch.allclose(out[:, : lay.k], base[:, tperm], atol=1e-5)
    assert torch.allclose(out[:, lay.k :], base[:, lay.k :], atol=1e-5)
    o3 = o.clone()
    others = o3[:, lay.others_at : lay.global_at].reshape(len(o), lay.max_robots - 1, -1)
    o3[:, lay.others_at : lay.global_at] = others[:, [2, 0, 1]].reshape(len(o), -1)
    assert torch.allclose(actor.logits(o3, m), base, atol=1e-5)


def test_semi_mdp_gae_on_a_hand_built_chain() -> None:
    b = Batch()
    # robot A: 0 -> 2 -> end (terminal); robot B: 1 -> end (truncated, bootstrap 4.0)
    b.value, b.reward = [1.0, 2.0, 3.0], [0.5, 1.0, 2.0]
    b.dt, b.nxt = [10.0, 5.0, 20.0], [2, BOOTSTRAP, TERMINAL]
    b.action = [0, 0, 0]
    g, lam = 0.99, 0.9
    adv, ret = gae(b, {1: 4.0}, g, lam)
    a2 = 2.0 - 3.0
    a1 = 1.0 + g**5 * 4.0 - 2.0
    a0 = 0.5 + g**10 * 3.0 - 1.0 + g**10 * lam * a2
    assert adv == pytest.approx([a0, a1, a2])
    assert ret == pytest.approx([a0 + 1.0, a1 + 2.0, a2 + 3.0])


@pytest.mark.parametrize("critic", ["central", "local"])
def test_short_training_run_produces_a_decentralized_policy(tmp_path: Path, critic: str) -> None:
    cfg = PPOConfig(
        name="t", critic=critic, budget=4000, bc_decisions=1500, bc_epochs=15, rollout=1000,
        minibatch=256, n_envs=4, eval_every=10**9, critic_warmup=1,
    )  # fmt: skip
    trainer = PPOTrainer(cfg, tmp_path)
    trainer.val_seeds = [10000, 10001]
    summary = trainer.train(log=lambda m: None)
    assert 4000 <= summary["decisions"] < 4000 + 2000
    assert (tmp_path / "best" / "actor.pt").exists() and (tmp_path / "final" / "actor.pt").exists()
    method = "MAPPO" if critic == "central" else "IPPO"
    other = "IPPO" if critic == "central" else "MAPPO"
    pol = make(method, str(tmp_path / "best"))
    assert pol.decentralized and family(method) == "L" and pol.machine_rule == "SPT"
    log = run_episode(CellEnv("configs/cell/cell_random_10_flex.yaml", n_robots=3), pol, 20000)
    assert compute(log).delivered == len(log.orders)
    with pytest.raises(ValueError, match="checkpoint"):
        make(other, str(tmp_path / "best"))


def test_behaviour_cloning_imitates_the_teacher(tmp_path: Path) -> None:
    cfg = PPOConfig(name="bc", bc_decisions=3000, bc_epochs=30, n_envs=1)
    trainer = PPOTrainer(cfg, tmp_path)
    out = trainer.warm_start(log=lambda m: None)
    assert out["bc_match"] > 0.97
