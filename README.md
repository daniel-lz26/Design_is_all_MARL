# som-marl

Decentralized multi-agent reinforcement learning (MAPPO) for scheduling in a self-organizing
manufacturing cell, deployed on TurtleBot3 Burger robots. Design-is-ALL Lab, CSUF.

Project docs live in `Context/som-marl-docs/docs/`.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
make install                     # pip install -e .[dev]
make lint
make test
```

On Windows, `make` ships with MSYS2 at `C:\msys64\usr\bin`; add it to `PATH`.
