"""Subprocess transport tests independent of Julia, NumPy, and application models."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from blab.solvers.beat_worker import WorkerPool, WorkerProcess


@pytest.fixture
def worker_script(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(
        """
import json
import os
import sys

print(json.dumps({"type": "ready"}), flush=True)
for line in sys.stdin:
    submission = json.loads(line)
    operation = submission["operation"]
    if operation == "crash":
        print("synthetic worker crash", file=sys.stderr, flush=True)
        sys.exit(7)
    if operation in {"cancelled", "failed"}:
        print(json.dumps({"type": operation}), flush=True)
        continue
    with open(submission["request"], encoding="utf-8") as stream:
        payload = json.load(stream)
    print("worker progress", flush=True)
    print(json.dumps({"type": "result", "payload": payload, "operation": operation,
                      "pid": os.getpid(), "threads": os.environ["JULIA_NUM_THREADS"],
                      "sdk": os.environ.get("TEST_SDK")}), flush=True)
    print(json.dumps({"type": "completed"}), flush=True)
""",
        encoding="utf-8",
    )
    return script


def options(script, **overrides):
    return dict(julia_executable=sys.executable, solver_script=script, julia_threads=2, julia_project=None) | overrides


def result(worker, request, operation="solve"):
    events = list(worker.submit(request, operation=operation))
    assert events[-1]["type"] == "completed"
    return next(event for event in events if event["type"] == "result")


def test_worker_streams_opaque_requests_and_reuses_process(worker_script, tmp_path):
    request = tmp_path / "request.json"
    payload = {"excitation_ids": ["left", "right"], "pressure": {"real": [1, 2], "imag": [-3, 4]}}
    request.write_text(json.dumps(payload))
    environment = dict(os.environ, TEST_SDK="first")
    worker = WorkerProcess(**options(worker_script, environment=environment))
    environment["TEST_SDK"] = "mutated"
    try:
        first = result(worker, request)
        field = result(worker, request, "bem_field")
        assert first["payload"] == field["payload"] == payload
        assert first["pid"] == field["pid"]
        assert first["threads"] == "2"
        assert first["sdk"] == "first"
        assert field["operation"] == "bem_field"
        assert first["_transport"]["julia_stdout_bytes"] > 0
    finally:
        worker.terminate()


@pytest.mark.parametrize("terminal", ["cancelled", "failed"])
def test_terminal_event_releases_submission_lock(worker_script, tmp_path, terminal):
    request = tmp_path / "request.json"
    request.write_text("{}")
    worker = WorkerProcess(**options(worker_script))
    try:
        assert list(worker.submit(request, operation=terminal)) == [{"type": terminal}]
        assert result(worker, request)["payload"] == {}
    finally:
        worker.terminate()


def test_worker_can_restart_after_process_failure(worker_script, tmp_path):
    request = tmp_path / "request.json"
    request.write_text("{}")
    worker = WorkerProcess(**options(worker_script))
    try:
        first = result(worker, request)
        with pytest.raises(RuntimeError, match="exited with code 7"):
            list(worker.submit(request, operation="crash"))
        assert result(worker, request)["pid"] != first["pid"]
    finally:
        worker.terminate()


def test_pool_keys_include_runtime_environment_and_shutdown_releases_workers(worker_script, tmp_path):
    request = tmp_path / "request.json"
    request.write_text("{}")
    pool = WorkerPool()
    env = dict(os.environ, TEST_SDK="first")
    first = pool.get_worker(**options(worker_script, environment=env))
    try:
        assert pool.get_worker(**options(worker_script, environment=dict(env))) is first
        changed = pool.get_worker(**options(worker_script, environment=env | {"TEST_SDK": "second"}))
        assert changed is not first
        assert pool.get_worker(**options(worker_script, julia_threads=3, environment=env)) is not first
        assert result(first, request)["sdk"] == "first"
        assert result(changed, request)["sdk"] == "second"
        processes = [first._process, changed._process]
        pool.shutdown()
        assert all(process.poll() is not None for process in processes)
        assert pool.get_worker(**options(worker_script, environment=env)) is not first
    finally:
        pool.shutdown()


def test_transport_loads_without_any_application_or_third_party_imports():
    script = Path(__file__).resolve().parents[1] / "src/blab/solvers/beat_worker.py"
    code = """
import importlib.abc
import importlib.util
import sys

class StandardLibraryOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] not in sys.stdlib_module_names:
            raise AssertionError(f'Non-standard-library dependency: {fullname}')

sys.meta_path.insert(0, StandardLibraryOnly())
spec = importlib.util.spec_from_file_location('standalone_worker', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.WorkerPool
"""
    process = subprocess.run(
        [sys.executable, "-I", "-c", code, str(script)], capture_output=True, text=True, timeout=30
    )
    assert process.returncode == 0, process.stderr


def test_production_imports_do_not_load_source_request_adapter():
    code = """
import importlib.abc
import sys

class RejectSourceAdapter(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'blab.solvers.beat_engine_backend', 'blab.solvers.julia_local_backend'}:
            raise AssertionError(f'Production imported source adapter: {fullname}')

sys.meta_path.insert(0, RejectSourceAdapter())
import blab.headless
import blab.solvers.coupled_backend
import blab.solvers.coupled_field
import blab.deploy_worker
"""
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, PYTHONPATH=str(root / "src")),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr


def test_compatibility_exports_share_the_runtime_pool(worker_script):
    from blab.solvers import beat_engine_backend as legacy
    from blab.solvers import beat_engine_runtime as runtime
    from blab.solvers import julia_local_backend as older

    assert older.BeatEngineWorkerProcess is runtime.BeatEngineWorkerProcess
    assert legacy._get_julia_worker is runtime.get_beat_engine_worker
    assert legacy.shutdown_beat_engine_workers is runtime.shutdown_beat_engine_workers
    try:
        assert legacy._get_julia_worker(**options(worker_script)) is runtime.get_beat_engine_worker(
            **options(worker_script)
        )
    finally:
        runtime.shutdown_beat_engine_workers()
