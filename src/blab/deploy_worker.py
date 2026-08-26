"""Persistent JSON-lines worker used by the Boundary Lab Deploy desktop shell."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from blab.deploy_solve import prepare_deploy_solve_request
from blab.solvers.beat_engine_backend import (
    DEFAULT_BEAT_ENGINE_CPU_PROJECT,
    DEFAULT_BEAT_ENGINE_CUDA_PROJECT,
    DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT,
    BeatEngineWorkerProcess,
    shutdown_beat_engine_workers,
)


def _emit(event_type: str, *, request_id: object | None = None, **values: Any) -> None:
    payload = {"type": event_type, **values}
    if request_id is not None:
        payload["id"] = request_id
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def _worker(backend: str) -> BeatEngineWorkerProcess:
    normalized = backend.strip().lower()
    if normalized not in {"cuda", "cpu"}:
        raise ValueError("Deploy worker backend must be cuda or cpu.")
    project = DEFAULT_BEAT_ENGINE_CUDA_PROJECT if normalized == "cuda" else DEFAULT_BEAT_ENGINE_CPU_PROJECT
    return BeatEngineWorkerProcess(
        julia_executable=os.environ.get("BLAB_JULIA_EXE", "julia"),
        solver_script=DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT,
        julia_threads=os.environ.get("BLAB_JULIA_THREADS", "auto"),
        julia_project=project,
    )


def _solve(request_id: object, payload: object, workers: dict[str, BeatEngineWorkerProcess]) -> None:
    backend = str(payload.get("backend", "cuda")) if isinstance(payload, dict) else "cuda"
    normalized = backend.strip().lower()
    worker = workers.get(normalized)
    if worker is None:
        worker = _worker(normalized)
        workers[normalized] = worker

    with tempfile.TemporaryDirectory(prefix="blab-deploy-") as temp_dir:
        request_path, _request = prepare_deploy_solve_request(payload, temp_dir)
        events = worker.submit(
            Path(request_path),
            status_callback=lambda message: _emit("status", request_id=request_id, message=message),
            operation="solve",
        )
        for event in events:
            event_type = str(event.get("type", ""))
            if event_type == "status":
                _emit("status", request_id=request_id, message=str(event.get("message", "")))
            elif event_type == "initialized":
                _emit("initialized", request_id=request_id, metadata=event)
            elif event_type == "result":
                _emit("result", request_id=request_id, result=event.get("result"))
            elif event_type == "completed":
                _emit("completed", request_id=request_id)
            elif event_type == "cancelled":
                _emit("cancelled", request_id=request_id)
            elif event_type == "failed":
                raise RuntimeError(str(event.get("error", "BEAT Engine Deploy solve failed.")))


def main() -> int:
    workers: dict[str, BeatEngineWorkerProcess] = {}
    _emit("ready", protocol="boundary_lab_deploy_worker", pid=os.getpid())
    try:
        try:
            for line in sys.stdin:
                text = line.strip()
                if not text:
                    continue
                request_id: object | None = None
                try:
                    message = json.loads(text)
                    if not isinstance(message, dict):
                        raise ValueError("Deploy worker message must be an object.")
                    request_id = message.get("id")
                    if message.get("operation") != "solve":
                        raise ValueError("Unsupported Deploy worker operation.")
                    _solve(request_id, message.get("payload"), workers)
                except Exception as exc:
                    _emit("failed", request_id=request_id, error=str(exc))
        except KeyboardInterrupt:
            pass
    finally:
        for worker in workers.values():
            worker.terminate()
        shutdown_beat_engine_workers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
