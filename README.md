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

## Linux (Ubuntu) setup

Tested with Python 3.10 (Ubuntu 22.04 / ROS2 Humble) and 3.13; any 3.10-3.13 works. pip picks
dependency versions that support your Python (on 3.10 that means numpy 2.2 and matplotlib 3.10).
Plots use matplotlib's non-GUI backend, so a headless server is fine.

```bash
sudo apt install python3-venv          # Ubuntu 22.04: python3.10-venv
git clone git@github.com:daniel-lz26/Design_is_all_MARL.git && cd Design_is_all_MARL
python3 -m venv .venv && source .venv/bin/activate
# Optional: CPU-only torch avoids a ~2.5 GB CUDA download (training runs on the CPU anyway)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt && pip install -e .
python -m pytest -q                     # 178 tests, about 2 minutes
```

Long runs survive a closed SSH session with `nohup ... &` or inside `tmux`. Parallel jobs use the
`spawn` start method on every OS. Cell evaluations resume with `--skip-existing`; central-scheduler
jobs use 4 solver threads each, so set `--workers` to about cores / 4 for them. CP-SAT runs in
deterministic mode, so results do not depend on the machine.
