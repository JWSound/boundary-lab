"""Persistent JSON-lines worker used by the Boundary Lab Deploy desktop shell."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import threading
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

_EMIT_LOCK = threading.Lock()


def _emit(event_type: str, *, request_id: object | None = None, **values: Any) -> dict[str, float | int]:
    payload = {"type": event_type, **values}
    if request_id is not None:
        payload["id"] = request_id
    encode_started = time.perf_counter()
    encoded = json.dumps(payload, separators=(",", ":"))
    encode_seconds = time.perf_counter() - encode_started
    with _EMIT_LOCK:
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


def _microphone_sweep(
    request_id: object,
    payload: object,
    workers: dict[str, BeatEngineWorkerProcess],
    solve_cache: DeploySolveCache,
    cancel_event: threading.Event,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Deploy microphone sweep request must be an object.")
    package_path = Path(str(payload.get("packagePath", ""))).expanduser().resolve()
    package_data = solve_cache.load_package(package_path)
    frequencies = sorted({float(value) for value in package_data.frequencies})
    if not frequencies:
        raise ValueError("Speaker package contains no frequencies for the microphone sweep.")
    raw_microphones = payload.get("microphones")
    if not isinstance(raw_microphones, list) or not raw_microphones:
        raise ValueError("Deploy microphone sweep requires at least one microphone.")
    if len(raw_microphones) > 64:
        raise ValueError("Deploy microphone sweep supports at most 64 microphones.")
    microphone_ids: list[str] = []
    observation_points: list[list[float]] = []
    for index, raw in enumerate(raw_microphones):
        if not isinstance(raw, dict):
            raise ValueError(f"microphones[{index}] must be an object.")
        microphone_id = str(raw.get("id", "")).strip()
        if not microphone_id:
            raise ValueError(f"microphones[{index}].id must not be empty.")
        point = [
            float(raw.get("positionX", 0.0)),
            float(raw.get("positionHeightM", 0.0)),
            float(raw.get("positionZ", 0.0)),
        ]
        if not all(math.isfinite(value) for value in point):
            raise ValueError(f"microphones[{index}] position must be finite.")
        if point[1] < 0.0:
            raise ValueError(f"microphones[{index}] cannot be below the ground plane.")
        microphone_ids.append(microphone_id)
        observation_points.append(point)
    if len(set(microphone_ids)) != len(microphone_ids):
        raise ValueError("Deploy microphone ids must be unique.")

    backend = str(payload.get("backend", "cuda")).strip().lower()
    worker = workers.get(backend)
    if worker is None:
        worker = _worker(backend)
        workers[backend] = worker
    spl_rows: list[list[float]] = [[] for _ in microphone_ids]
    pressure_real_rows: list[list[float]] = [[] for _ in microphone_ids]
    pressure_imag_rows: list[list[float]] = [[] for _ in microphone_ids]
    completed_count = 0
    with tempfile.TemporaryDirectory(prefix="blab-deploy-microphones-") as temp_dir:
        for frequency_hz in frequencies:
            if cancel_event.is_set():
                _emit("cancelled", request_id=request_id, completed_count=completed_count)
                return
            frequency_payload = {
                **payload,
                "frequencyHz": frequency_hz,
                "observationPointsM": observation_points,
                "includeComplexPressure": True,
                "solutionKey": f"microphone-sweep:{request_id}:{frequency_hz:.9g}",
                "reuseBoundary": False,
            }
            request_path, _request = prepare_deploy_solve_request(
                frequency_payload,
                temp_dir,
                cache=solve_cache,
            )
            frequency_result: dict[str, Any] | None = None
            for event in worker.submit(
                request_path,
                status_callback=lambda message: _emit(
                    "status",
                    request_id=request_id,
                    message=message,
                ),
            ):
                event_type = str(event.get("type", ""))
                if event_type == "status":
                    _emit("status", request_id=request_id, message=str(event.get("message", "")))
                elif event_type == "initialized":
                    _emit("initialized", request_id=request_id, metadata=event)
                elif event_type == "result":
                    result = event.get("result")
                    if not isinstance(result, dict):
                        raise RuntimeError("BEAT Engine microphone sweep returned an invalid result.")
                    frequency_result = result
                elif event_type == "cancelled":
                    _emit("cancelled", request_id=request_id, completed_count=completed_count)
                    return
                elif event_type == "failed":
                    raise RuntimeError(str(event.get("error", "BEAT Engine microphone sweep failed.")))
            if frequency_result is None:
                raise RuntimeError("BEAT Engine microphone sweep completed a frequency without pressure values.")
            spl = frequency_result.get("spl_db")
            pressure = frequency_result.get("field_pressure")
            if not isinstance(spl, list) or len(spl) != len(microphone_ids):
                raise RuntimeError("BEAT Engine microphone SPL result does not match the microphone count.")
            if not isinstance(pressure, dict):
                raise RuntimeError("BEAT Engine microphone sweep did not return complex pressure.")
            pressure_real = pressure.get("real")
            pressure_imag = pressure.get("imag")
            if not isinstance(pressure_real, list) or not isinstance(pressure_imag, list):
                raise RuntimeError("BEAT Engine microphone pressure result is invalid.")
            if len(pressure_real) != len(microphone_ids) or len(pressure_imag) != len(microphone_ids):
                raise RuntimeError("BEAT Engine microphone pressure result does not match the microphone count.")
            for microphone_index in range(len(microphone_ids)):
                spl_rows[microphone_index].append(float(spl[microphone_index]))
                pressure_real_rows[microphone_index].append(float(pressure_real[microphone_index]))
                pressure_imag_rows[microphone_index].append(float(pressure_imag[microphone_index]))
            completed_count += 1
            _emit(
                "microphone-progress",
                request_id=request_id,
                frequency_hz=frequency_hz,
                completed_count=completed_count,
                total_count=len(frequencies),
                microphone_ids=microphone_ids,
                spl_db=spl,
            )
    _emit(
        "result",
        request_id=request_id,
        result={
            "frequencies_hz": frequencies,
            "microphone_ids": microphone_ids,
            "spl_db": spl_rows,
            "pressure": {"real": pressure_real_rows, "imag": pressure_imag_rows},
            "completed_count": completed_count,
            "total_count": len(frequencies),
        },
    )
    _emit("completed", request_id=request_id)


def main() -> int:
    workers: dict[str, BeatEngineWorkerProcess] = {}
    solve_cache = DeploySolveCache()
    solution_keys: dict[str, str] = {}
    active_lock = threading.Lock()
    active: dict[str, Any] = {"thread": None, "request_id": None, "cancel": None, "backend": None}

    def run_job(request_id: object, operation: str, payload: object, input_transport: dict[str, float | int]) -> None:
        cancel_event = active["cancel"]
        try:
            if operation == "microphone_sweep":
                _microphone_sweep(request_id, payload, workers, solve_cache, cancel_event)
            else:
                _solve(request_id, payload, workers, input_transport, solve_cache, solution_keys)
        except Exception as exc:
            if cancel_event.is_set():
                _emit("cancelled", request_id=request_id)
            else:
                _emit("failed", request_id=request_id, error=str(exc))
        finally:
            with active_lock:
                active.update(thread=None, request_id=None, cancel=None, backend=None)

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
                    if operation == "cancel":
                        with active_lock:
                            target_id = message.get("target_id")
                            matches = active["thread"] is not None and (
                                target_id is None or target_id == active["request_id"]
                            )
                            if matches:
                                active["cancel"].set()
                                worker = workers.get(str(active["backend"]))
                                if worker is not None:
                                    worker.terminate()
                        _emit("completed", request_id=request_id, cancelled=bool(matches))
                        continue
                    if operation == "warmup":
                        backend = str(message.get("backend", "cuda")).strip().lower()
                        worker = workers.get(backend)
                        if worker is None:
                            worker = _worker(backend)
                            workers[backend] = worker
                        worker.ensure_started()
                        _emit("completed", request_id=request_id)
                        continue
                    if operation not in {"solve", "microphone_sweep"}:
                        raise ValueError("Unsupported Deploy worker operation.")
                    with active_lock:
                        if active["thread"] is not None:
                            raise RuntimeError("A Deploy solve is already in progress.")
                        payload = message.get("payload")
                        backend = str(payload.get("backend", "cuda")) if isinstance(payload, dict) else "cuda"
                        cancel_event = threading.Event()
                        thread = threading.Thread(
                            target=run_job,
                            args=(request_id, str(operation), payload, input_transport),
                            daemon=True,
                        )
                        active.update(
                            thread=thread,
                            request_id=request_id,
                            cancel=cancel_event,
                            backend=backend.strip().lower(),
                        )
                        thread.start()
                except Exception as exc:
                    _emit("failed", request_id=request_id, error=str(exc))
        except KeyboardInterrupt:
            pass
    finally:
        with active_lock:
            if active["cancel"] is not None:
                active["cancel"].set()
        for worker in workers.values():
            worker.terminate()
        thread = active.get("thread")
        if thread is not None:
            thread.join(timeout=2.0)
        shutdown_beat_engine_workers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
