# som-marl task runner. Uses the project venv if present, otherwise the active python.
ifeq ($(OS),Windows_NT)
VENV_PY := .venv/Scripts/python.exe
else
VENV_PY := .venv/bin/python
endif
PYTHON ?= $(if $(wildcard $(VENV_PY)),$(VENV_PY),python)

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
