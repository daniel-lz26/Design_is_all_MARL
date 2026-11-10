from __future__ import annotations

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from som.config.schema import EnvCfg, LayoutCfg, RewardCfg, ScenarioCfg
from som.env.parallel_env import SomParallelEnv
from som.core.layout import Layout
from som.eval.evaluate import run_one
from som.policies.heuristics import make_policy
from som.sim.travel import StraightLineTravel


def _random_rollout(env: SomParallelEnv, seed: int) -> tuple[list[dict], list[dict], dict]:
    """Play one episode with uniformly random legal actions; return rewards, infos, last infos."""
    rng = np.random.default_rng(seed)
    obs, infos = env.reset(seed=seed)
    rewards, all_infos = [], [infos]
    while env.agents:
        acts = {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}
        obs, rew, _, _, infos = env.step(acts)
        rewards.append(rew)
        all_infos.append(infos)
    return rewards, all_infos, infos


def test_pettingzoo_parallel_api(arena_cfg: LayoutCfg, random_10: ScenarioCfg) -> None:
    parallel_api_test(SomParallelEnv(arena_cfg, random_10), num_cycles=300)


def test_shapes_do_not_depend_on_robot_count(arena_cfg: LayoutCfg, random_10: ScenarioCfg) -> None:
    envs = [SomParallelEnv(arena_cfg, random_10, EnvCfg(n_robots=n)) for n in (2, 4)]
    for role_agent in ("robot_0", "machine_milling"):
        shapes = {e.observation_space(role_agent)["observation"].shape for e in envs}
        assert len(shapes) == 1
    assert envs[0].state_dim == envs[1].state_dim
    for e in envs:
        e.reset(seed=3)
        assert all(s.shape == (e.state_dim,) for s in e.agent_states().values())


def test_idle_agents_may_only_noop(arena_cfg: LayoutCfg, random_10: ScenarioCfg) -> None:
    env = SomParallelEnv(arena_cfg, random_10)
    obs, infos = env.reset(seed=5)
    for _ in range(30):
        for a in env.agents:
            mask = obs[a]["action_mask"]
            if infos[a]["decision"]:
                assert mask[1:].any(), f"{a} must decide but has no candidate slot"
            else:
                assert mask.tolist() == [1] + [0] * (len(mask) - 1)
        acts = {a: int(np.flatnonzero(obs[a]["action_mask"])[-1]) for a in env.agents}
        obs, _, _, _, infos = env.step(acts)
        if not env.agents:
            break


def test_masked_action_is_rejected(arena_cfg: LayoutCfg, random_10: ScenarioCfg) -> None:
    env = SomParallelEnv(arena_cfg, random_10)
    obs, infos = env.reset(seed=0)
    agent = next(a for a in env.agents if infos[a]["decision"])
    bad = int(np.flatnonzero(obs[agent]["action_mask"] == 0)[0])
    with pytest.raises(ValueError, match="masked"):
        env.step({agent: bad})


@pytest.mark.parametrize("seed", [11, 12])  # 11: a machine WAITs forever and the run truncates
def test_episode_reward_matches_metrics(
    arena_cfg: LayoutCfg, random_10: ScenarioCfg, seed: int
) -> None:
    """Dense team reward summed over the episode equals the episode-level objective."""
    cfg = RewardCfg(w_deliver=1.0, w_tardy=0.01, w_ontime=2.0, w_time=0.001)
    env = SomParallelEnv(arena_cfg, random_10, reward_cfg=cfg)
    rewards, infos, _ = _random_rollout(env, seed=seed)
    log = env.episode_log()
    total = sum(r["robot_0"] for r in rewards)
    assert all(len(set(r.values())) == 1 for r in rewards)  # team reward
    end = log.end_time
    delivered = [o for o in log.orders if o.delivered is not None]
    tardiness = sum(
        max(0.0, (o.delivered if o.delivered is not None else end) - o.due) for o in log.orders
    )
    on_time = sum(o.delivered <= o.due for o in delivered)
    expected = len(delivered) * cfg.w_deliver + on_time * cfg.w_ontime
    expected -= cfg.w_tardy * tardiness + cfg.w_time * end
    assert total == pytest.approx(expected, abs=1e-6)
    assert sum(i["robot_0"]["dt"] for i in infos) == pytest.approx(end)


def test_horizon_truncates_instead_of_terminating(
    arena_cfg: LayoutCfg, random_10: ScenarioCfg
) -> None:
    short = random_10.model_copy(update={"horizon": 120.0})
    env = SomParallelEnv(arena_cfg, short)
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=1)
    while True:
        acts = {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}
        obs, _, term, trunc, _ = env.step(acts)
        if not env.agents:
            break
    assert all(trunc.values()) and not any(term.values())
    assert env.episode_log().truncated


def test_seed_matches_evaluation_seed(arena_cfg: LayoutCfg, random_10: ScenarioCfg) -> None:
    env = SomParallelEnv(arena_cfg, random_10)
    env.reset(seed=1234)
    env_orders = [(o.spec.release_time, o.spec.due_time) for o in env.runtime.core.orders.values()]
    planner = StraightLineTravel.from_cfg(arena_cfg.travel_calibration)
    pol = make_policy("FIFO/FIFO", Layout.from_cfg(arena_cfg), planner)
    log = run_one(arena_cfg, random_10, EnvCfg(), pol, 1234)
    assert env_orders == [(o.release, o.due) for o in log.orders]


def test_local_reward_credits_deliverers(arena_cfg: LayoutCfg, random_10: ScenarioCfg) -> None:
    env = SomParallelEnv(
        arena_cfg,
        random_10,
        EnvCfg(machine_wait_allowed=False),
        RewardCfg(mode="local", w_tardy=0.0),
    )
    rewards, _, _ = _random_rollout(env, seed=2)
    totals = {a: sum(r[a] for r in rewards) for a in env.possible_agents}
    assert sum(totals[a] for a in ("robot_0", "robot_1")) >= 10  # each delivery credits a robot
    assert totals["machine_turning"] == 0.0 or totals["machine_grinding"] > 0  # P1 ends at grinding
