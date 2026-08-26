"""Persistent JSON-lines worker used by the Boundary Lab Deploy desktop shell."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from blab.deploy_solve import DeploySolveCache, prepare_deploy_field_request, prepare_deploy_solve_request
from blab.solvers.beat_engine_backend import (
    DEFAULT_BEAT_ENGINE_CPU_PROJECT,
    DEFAULT_BEAT_ENGINE_CUDA_PROJECT,
    DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT,
    BeatEngineWorkerProcess,
    shutdown_beat_engine_workers,
)


def _emit(event_type: str, *, request_id: object | None = None, **values: Any) -> dict[str, float | int]:
    payload = {"type": event_type, **values}
    if request_id is not None:
        payload["id"] = request_id
    encode_started = time.perf_counter()
    encoded = json.dumps(payload, separators=(",", ":"))
    encode_seconds = time.perf_counter() - encode_started
    sys.stdout.write(encoded + "\n")
    sys.stdout.flush()
    return {
        "json_encode_s": encode_seconds,
        "stdout_bytes": len(encoded.encode("utf-8")),
    }


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


def _solve(
    request_id: object,
    payload: object,
    workers: dict[str, BeatEngineWorkerProcess],
    input_transport: dict[str, float | int],
    solve_cache: DeploySolveCache,
    solution_keys: dict[str, str],
) -> None:
    backend = str(payload.get("backend", "cuda")) if isinstance(payload, dict) else "cuda"
    normalized = backend.strip().lower()
    worker = workers.get(normalized)
    if worker is None:
        worker = _worker(normalized)
        workers[normalized] = worker

    with tempfile.TemporaryDirectory(prefix="blab-deploy-") as temp_dir:
        prepare_started = time.perf_counter()
        requested_solution_key = str(payload.get("solutionKey", "")) if isinstance(payload, dict) else ""
        reuse_boundary = bool(payload.get("reuseBoundary", False)) if isinstance(payload, dict) else False
        field_only = reuse_boundary and bool(requested_solution_key) and (
            solution_keys.get(normalized) == requested_solution_key
        )
        if field_only:
            request_path, _request = prepare_deploy_field_request(payload, temp_dir)
        else:
            request_path, _request = prepare_deploy_solve_request(payload, temp_dir, cache=solve_cache)
        prepare_seconds = time.perf_counter() - prepare_started
        request_bytes = request_path.stat().st_size
        julia_started = time.perf_counter()
        events = worker.submit(
            Path(request_path),
            status_callback=lambda message: _emit("status", request_id=request_id, message=message),
            operation="field" if field_only else "solve",
        )
        for event in events:
            event_type = str(event.get("type", ""))
            if event_type == "status":
                _emit("status", request_id=request_id, message=str(event.get("message", "")))
            elif event_type == "initialized":
                _emit("initialized", request_id=request_id, metadata=event)
            elif event_type == "result":
                result = event.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError("BEAT Engine Deploy solve returned an invalid result payload.")
                julia_transport = event.get("_transport", {})
                result["pipeline"] = {
                    "python_input_json_parse_s": float(input_transport.get("json_parse_s", 0.0)),
                    "electron_python_stdin_bytes": int(input_transport.get("stdin_bytes", 0)),
                    "python_prepare_s": prepare_seconds,
                    "julia_request_json_bytes": request_bytes,
                    "python_julia_result_wait_s": time.perf_counter() - julia_started,
                    "julia_python_stdout_bytes": int(julia_transport.get("julia_stdout_bytes", 0)),
                    "python_julia_json_parse_s": float(julia_transport.get("python_julia_json_parse_s", 0.0)),
                    "field_only": int(field_only),
                }
                if not field_only:
                    solution_keys[normalized] = str(_request.get("solution_key", requested_solution_key))
                result_emit = _emit("result", request_id=request_id, result=result)
                _emit(
                    "profile",
                    request_id=request_id,
                    metrics={
                        "python_result_json_encode_s": result_emit["json_encode_s"],
                        "python_electron_stdout_bytes": result_emit["stdout_bytes"],
                    },
                )
            elif event_type == "completed":
                _emit("completed", request_id=request_id)
            elif event_type == "cancelled":
                _emit("cancelled", request_id=request_id)
            elif event_type == "failed":
                raise RuntimeError(str(event.get("error", "BEAT Engine Deploy solve failed.")))


def main() -> int:
    workers: dict[str, BeatEngineWorkerProcess] = {}
    solve_cache = DeploySolveCache()
    solution_keys: dict[str, str] = {}
    _emit("ready", protocol="boundary_lab_deploy_worker", pid=os.getpid())
    try:
        try:
            for line in sys.stdin:
                text = line.strip()
                if not text:
                    continue
                request_id: object | None = None
                try:
                    parse_started = time.perf_counter()
                    message = json.loads(text)
                    input_transport = {
                        "json_parse_s": time.perf_counter() - parse_started,
                        "stdin_bytes": len(text.encode("utf-8")),
                    }
                    if not isinstance(message, dict):
                        raise ValueError("Deploy worker message must be an object.")
                    request_id = message.get("id")
                    operation = message.get("operation")
                    if operation == "warmup":
                        backend = str(message.get("backend", "cuda")).strip().lower()
                        worker = workers.get(backend)
                        if worker is None:
                            worker = _worker(backend)
                            workers[backend] = worker
                        worker.ensure_started()
                        _emit("completed", request_id=request_id)
                        continue
                    if operation != "solve":
                        raise ValueError("Unsupported Deploy worker operation.")
                    _solve(
                        request_id,
                        message.get("payload"),
                        workers,
                        input_transport,
                        solve_cache,
                        solution_keys,
                    )
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
