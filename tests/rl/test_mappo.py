from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from som.config.schema import EnvCfg, LayoutCfg, ScenarioCfg, TrainCfg
from som.eval.evaluate import run_one
from som.eval.metrics import compute_metrics
from som.policies.learned import LearnedPolicy
from som.rl.artifact import META_FILE, ArtifactMismatch, load_artifact
from som.rl.buffer import semi_mdp_gae
from som.rl.mappo import MAPPOTrainer
from som.rl.networks import RoleActor
from som.rl.normalization import RunningMeanStd

# One robot, two Turning-only orders released together. FIFO slot order puts the loose order
# first; only taking the tight order first gets both on time (see the timing in the test).
TOY = {
    "name": "toy_2",
    "kind": "fixed",
    "horizon": 200,
    "default_process_times": {"Turning": 10, "Milling": 10},
    "job_types": {"J1": {"part": "Shaft", "route": ["Turning"]}},
    "orders": [
        {"id": 1, "job": "J1", "release": 0, "due": 100},
        {"id": 2, "job": "J1", "release": 0, "due": 30},
    ],
}
TOY_ENV = EnvCfg(n_robots=1, machine_wait_allowed=False, travel_noise_sigma=0.0)


def _toy_cfg(**kw: object) -> TrainCfg:
    base = {
        "name": "toy",
        "layout": "-",
        "scenario": "-",
        "env": TOY_ENV.model_dump(),
        "total_steps": 3072,
        "n_envs": 4,
        "rollout_steps": 32,
        "lr": 3e-3,
        "hidden": [32, 32],
        "eval_every": 1000,
        "eval_seeds": [0],
    }
    return TrainCfg.model_validate({**base, **kw})


def test_masked_actions_are_never_sampled() -> None:
    torch.manual_seed(0)
    actor = RoleActor(8, 6, [16])
    obs = torch.randn(2000, 8)
    mask = (torch.rand(2000, 6) < 0.4).to(torch.int8)
    mask[:, 0] = torch.where(mask[:, 1:].any(1), mask[:, 0], torch.ones(2000, dtype=torch.int8))
    acts = actor.dist(obs, mask).sample()
    assert bool(mask[torch.arange(2000), acts].all())


def test_gae_discounts_by_elapsed_time_and_bootstraps_truncation() -> None:
    g, lam = 0.9, 1.0
    # One env, one agent, two steps: dt = 2 then 1; the episode ends after step 1.
    reward = np.array([[[1.0]], [[2.0]]])
    value = np.array([[[0.5]], [[0.25]]])
    last = np.array([[9.0]])  # value of the next episode's first state: must be ignored
    dt = np.array([[2.0], [1.0]])
    done = np.array([[False], [True]])
    tv = np.array([[[0.0]], [[4.0]]])
    for truncated, boot in ((False, 0.0), (True, 4.0)):
        tr = np.array([[False], [truncated]])
        adv, ret = semi_mdp_gae(reward, value, last, tv, dt, done, tr, g, lam)
        r1 = 2.0 + g**1 * boot
        assert ret[1, 0, 0] == pytest.approx(r1)
        assert ret[0, 0, 0] == pytest.approx(1.0 + g**2 * r1)
        assert adv[0, 0, 0] == pytest.approx(ret[0, 0, 0] - 0.5)


def test_running_mean_std_matches_numpy() -> None:
    rms = RunningMeanStd(eps=0.0)
    xs = np.random.default_rng(0).normal(3.0, 2.0, 1000)
    for chunk in np.split(xs, 10):
        rms.update(chunk)
    assert rms.mean == pytest.approx(xs.mean())
    assert rms.var == pytest.approx(xs.var())


def test_trainer_overfits_toy_scenario(tiny_layout_cfg: LayoutCfg, tmp_path: Path) -> None:
    toy = ScenarioCfg.model_validate(TOY)
    policy_before = MAPPOTrainer(_toy_cfg(), tiny_layout_cfg, toy, [], tmp_path / "a").policy()
    # Timing (v_eff 0.5 m/s, 2 s docks, 10 s turning): the first order is delivered at ~24 s,
    # the second at ~38 s, so the 30 s order is on time only if it goes first.
    trainer = MAPPOTrainer(_toy_cfg(), tiny_layout_cfg, toy, [], tmp_path / "b")
    trainer.train(on_log=lambda s: None)
    m = compute_metrics(run_one(tiny_layout_cfg, toy, TOY_ENV, trainer.policy(), 0))
    assert m.on_time == 2, "trained policy should take the tight order first"
    before = compute_metrics(run_one(tiny_layout_cfg, toy, TOY_ENV, policy_before, 0))
    assert before.delivered == 2  # sanity: untrained policies still finish the toy
    assert (tmp_path / "b" / "policy" / META_FILE).exists()
    assert (tmp_path / "b" / "progress.csv").exists()


def test_artifact_round_trip(tiny_layout_cfg: LayoutCfg, tmp_path: Path) -> None:
    toy = ScenarioCfg.model_validate(TOY)
    trainer = MAPPOTrainer(_toy_cfg(total_steps=128), tiny_layout_cfg, toy, [], tmp_path)
    trainer.train(on_log=lambda s: None)
    actors, meta = load_artifact(tmp_path / "policy")
    assert meta["obs_schema_version"] == 1 and meta["K_robot"] == 6 and meta["K_machine"] == 5
    assert set(actors) == {"robot", "machine"}
    loaded = LearnedPolicy.from_artifact(tmp_path / "policy", tiny_layout_cfg)
    a = run_one(tiny_layout_cfg, toy, TOY_ENV, trainer.policy(), 0)
    b = run_one(tiny_layout_cfg, toy, TOY_ENV, loaded, 0)
    assert [d.order for d in a.decisions] == [d.order for d in b.decisions]

    meta["obs_schema_version"] = 999
    (tmp_path / "policy" / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ArtifactMismatch):
        load_artifact(tmp_path / "policy")
