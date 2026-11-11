from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from som.cell_env.cell_env import CellEnv
from som.cell_env.harness import run_episode
from som.cell_env.metrics import compute
from som.cell_env.obs import CELL_OBS_VERSION
from som.methods.ppo.model import MASKED_LOGIT, CandidateActor, work_conserving_mask
from som.methods.ppo.policy import PPOPolicy, save_actor
from som.methods.ppo.rollout import BOOTSTRAP, TERMINAL, Batch, demos, gae
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


def test_work_conserving_mask_clears_park_and_wait_only_when_a_task_is_claimable() -> None:
    k = 3  # slots: 3 tasks, park, wait, noop
    m = torch.tensor(
        [
            [1, 0, 0, 1, 1, 0],  # a task is claimable: park and wait go
            [0, 0, 0, 1, 1, 0],  # nothing to claim: park and wait stay
            [0, 0, 0, 0, 1, 0],  # already parked, nothing to claim: wait stays
            [0, 0, 0, 0, 0, 1],  # not deciding: NOOP only
        ],
        dtype=torch.int8,
    )
    out = work_conserving_mask(m, k)
    assert out.tolist() == [
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]
    assert m[0, k] == 1  # the input is not modified
    assert bool((out.sum(-1) > 0).all())  # every row keeps a legal action


def test_work_conserving_actor_never_idles_while_a_task_is_claimable() -> None:
    env, obs, mask = _obs(12)
    k = env.enc.layout.k
    assert (mask[:, :k].any(-1) & mask[:, k : k + 2].any(-1)).any()  # the case is exercised
    actor = CandidateActor(env.enc.layout, work_conserving=True)
    torch.nn.init.normal_(actor.idle.weight, std=10.0)  # make idling attractive
    logits = actor.logits(torch.as_tensor(obs), torch.as_tensor(mask))
    has_task = torch.as_tensor(mask[:, :k].any(-1))
    assert (logits[has_task, k : k + 2] <= MASKED_LOGIT).all()
    free = CandidateActor(env.enc.layout, work_conserving=False)
    free.load_state_dict(actor.state_dict())
    assert (free.logits(torch.as_tensor(obs), torch.as_tensor(mask))[has_task, k + 1] > -1e3).all()


def test_teacher_demonstrations_are_legal_under_the_work_conserving_mask() -> None:
    """BC trains the masked actor on the teacher's actions, so none may be masked out."""
    _, mask, act = demos(
        lambda: make("BOTTLENECK/SPT"), ["cell_random_10_disrupt"], 400, (0, 50),
        np.random.default_rng(0),
    )  # fmt: skip
    k = CellEnv("configs/cell/cell_random_10.yaml").k
    wc = work_conserving_mask(torch.as_tensor(mask), k).numpy()
    assert wc[np.arange(len(act)), act].all()


def test_work_conserving_flag_round_trips_and_old_checkpoints_load_without_it(
    tmp_path: Path,
) -> None:
    env, _, _ = _obs(1)
    lay = env.enc.layout
    meta = {"obs_version": CELL_OBS_VERSION, "layout": dataclasses.asdict(lay), "hidden": 128}
    save_actor(
        tmp_path / "new" / "actor.pt", CandidateActor(lay), {**meta, "work_conserving": True}
    )
    save_actor(tmp_path / "old" / "actor.pt", CandidateActor(lay), meta)  # pre-fix meta
    assert PPOPolicy.load(tmp_path / "new").actor.work_conserving
    assert not PPOPolicy.load(tmp_path / "old").actor.work_conserving


@pytest.mark.parametrize("seed", [20000, 20001])
def test_an_untrained_work_conserving_policy_delivers_every_order(seed: int) -> None:
    env = CellEnv("configs/cell/cell_random_10_disrupt.yaml")
    env.reset(seed=seed)
    actor = CandidateActor(env.enc.layout, work_conserving=True)
    torch.nn.init.normal_(actor.idle.weight, std=10.0)
    torch.nn.init.constant_(actor.idle.bias, 50.0)  # always prefers to idle when allowed
    pol = PPOPolicy(actor, {"name": "MAPPO", "machine_rule": "SPT"})
    log = run_episode(env, pol, seed)
    assert compute(log).delivered == len(log.orders)
