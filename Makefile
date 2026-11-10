# som-marl task runner. Uses the project venv if present, otherwise the active python.
VENV_PY := $(firstword $(wildcard .venv/bin/python .venv/Scripts/python.exe))
PYTHON ?= $(or $(VENV_PY),python)

.PHONY: install test lint format typecheck baselines replay train eval

SCENARIO ?= abstract_10
SEEDS ?= 20
EP ?= 0
CFG ?= configs/train/mappo_2r.yaml
ROBOTS ?= 2

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check . && $(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix . && $(PYTHON) -m ruff format .

typecheck:
	$(PYTHON) -m mypy

baselines:
	$(PYTHON) scripts/compare_baselines.py --scenario $(SCENARIO) --seeds $(SEEDS)

# RUN defaults to the latest baselines run; the gif lands in reports/.
replay:
	$(PYTHON) scripts/replay.py $(if $(RUN),--run $(RUN)) --ep $(EP)

# One training run; SEED overrides the config seed (e.g. make train SEED=1).
train:
	$(PYTHON) scripts/train.py --cfg $(CFG) $(if $(SEED),--seed $(SEED))

# MAPPO vs every baseline. RUN defaults to the newest seeds of mappo_2r; ROBOTS=4 for scaling.
eval:
	$(PYTHON) scripts/evaluate.py $(if $(RUN),--runs $(RUN)) --scenario $(SCENARIO) --robots $(ROBOTS)
