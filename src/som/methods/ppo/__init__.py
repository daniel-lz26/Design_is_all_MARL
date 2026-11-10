"""PPO family for the cell env (spec §11.3, track 1): IPPO and MAPPO from one code path.

``critic="local"`` (IPPO) values each robot's own observation; ``critic="central"`` (MAPPO)
values that observation plus the full critic state. The actor is the same in both, shared by
every robot, and sees only its own observation, so execution is decentralized either way.
"""


def register_all() -> None:
    """Register IPPO and MAPPO; both load a checkpoint (``make(name, checkpoint)``)."""
    from som.policies.registry import register

    def factory(method: str):
        def make(checkpoint: str | None):
            if checkpoint is None:
                raise ValueError(f"{method} needs --checkpoint (a run's best/ folder)")
            from som.methods.ppo.policy import PPOPolicy

            return PPOPolicy.load(checkpoint, name=method)

        return make

    register("IPPO", factory("IPPO"))
    register("MAPPO", factory("MAPPO"))
