"""The released engine owns runtime paths and contracts."""

import os
import subprocess
import sys
from pathlib import Path


def run_import(code):
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src")),
    )


def test_default_uses_package_client_contract_and_paths():
    process = run_import("""
from beat_engine import EngineWorker, engine_paths
from blab.solvers import beat_engine_runtime as runtime, engine_contract
assert issubclass(runtime.BeatEngineWorkerProcess, EngineWorker)
for backend in ('cpu', 'cuda', 'rocm'):
    assert runtime.default_beat_engine_project(backend) == engine_paths(backend).project
assert engine_contract.validate_solve_request.__module__.startswith('beat_engine.')
assert runtime.BeatEngineWorkerProcess._prepare_submission is EngineWorker._prepare_submission
""")
    assert process.returncode == 0, process.stderr


def test_missing_package_does_not_fall_back():
    process = run_import("""
import importlib.abc, sys
class MissingEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'beat_engine':
            raise ImportError('Simulated missing package')
sys.meta_path.insert(0, MissingEngine())
import blab.solvers.engine_distribution
""")
    assert process.returncode != 0
    assert "BEAT Engine is not installed" in process.stderr


def test_incompatible_package_is_rejected():
    process = run_import("""
import sys, types
engine = types.ModuleType('beat_engine')
engine.__version__ = '0.0.0'
sys.modules['beat_engine'] = engine
import blab.solvers.engine_distribution
""")
    assert process.returncode != 0
    assert "requires beat-engine 0.1.0; found 0.0.0" in process.stderr
