# som-marl task runner. Uses the project venv if present, otherwise the active python.
VENV_PY := $(firstword $(wildcard .venv/bin/python .venv/Scripts/python.exe))
PYTHON ?= $(or $(VENV_PY),python)

.PHONY: install test lint format typecheck

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
