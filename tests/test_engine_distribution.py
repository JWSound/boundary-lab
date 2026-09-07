"""The pre-release dependency switch is explicit and fails closed."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def run_selection(selection, code):
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(
            os.environ,
            BLAB_BEAT_ENGINE_DISTRIBUTION=selection,
            PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        ),
    )


def test_bundled_selection_retains_local_assets():
    process = run_selection(
        "bundled",
        """
from pathlib import Path
from blab.solvers import beat_engine_runtime as runtime
assert runtime.DEFAULT_BEAT_ENGINE_CPU_PROJECT == Path(runtime.__file__).parent / 'julia_local'
""",
    )
    assert process.returncode == 0, process.stderr


def test_external_selection_uses_package_client_contract_and_paths():
    pytest.importorskip("beat_engine")
    process = run_selection(
        "external",
        """
from beat_engine import EngineWorker, engine_paths
from blab.solvers import beat_engine_runtime as runtime, engine_contract
assert issubclass(runtime.BeatEngineWorkerProcess, EngineWorker)
assert runtime.DEFAULT_BEAT_ENGINE_CPU_PROJECT == engine_paths().project
assert engine_contract.validate_solve_request.__module__.startswith('beat_engine.')
""",
    )
    assert process.returncode == 0, process.stderr


def test_unknown_selection_is_not_silently_ignored():
    process = run_selection("typo", "import blab.solvers.engine_distribution")
    assert process.returncode != 0
    assert "must be bundled or external" in process.stderr


def test_missing_external_package_does_not_fall_back():
    process = run_selection(
        "external",
        """
import importlib.abc, sys
class MissingEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'beat_engine':
            raise ImportError('Simulated missing package')
sys.meta_path.insert(0, MissingEngine())
import blab.solvers.engine_distribution
""",
    )
    assert process.returncode != 0
    assert "External BEAT was selected but is not installed" in process.stderr
