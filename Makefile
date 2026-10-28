# som-marl task runner. Uses the project venv if present, otherwise the active python.
VENV_PY := $(firstword $(wildcard .venv/bin/python .venv/Scripts/python.exe))
PYTHON ?= $(or $(VENV_PY),python)

.PHONY: install test lint format typecheck baselines replay

SCENARIO ?= abstract_10
SEEDS ?= 20
EP ?= 0

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
