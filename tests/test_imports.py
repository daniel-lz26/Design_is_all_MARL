from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys

import som


def _all_modules() -> list[str]:
    return sorted(m.name for m in pkgutil.walk_packages(som.__path__, prefix="som."))


def test_every_module_imports_and_has_docstring() -> None:
    for name in _all_modules():
        mod = importlib.import_module(name)
        assert mod.__doc__ and mod.__doc__.strip(), f"{name} is missing its module docstring"


def test_core_and_config_stay_pure() -> None:
    # Fresh interpreter so other tests' imports don't leak in.
    code = (
        "import sys, som.core, som.config, pkgutil, importlib\n"
        "for m in pkgutil.walk_packages(som.core.__path__, prefix='som.core.'):\n"
        "    importlib.import_module(m.name)\n"
        "bad = [m for m in ('torch', 'rclpy', 'pettingzoo') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_no_module_imports_rclpy_eagerly() -> None:
    code = (
        "import sys, som, pkgutil, importlib\n"
        "for m in pkgutil.walk_packages(som.__path__, prefix='som.'):\n"
        "    importlib.import_module(m.name)\n"
        "assert 'rclpy' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
