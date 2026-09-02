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

import numpy as np

from blab.deploy_solve import (
    DeploySolveCache,
    prepare_deploy_coupled_request,
    prepare_deploy_field_request,
    prepare_deploy_microphone_sweep_request,
    prepare_deploy_rom_microphone_sweep_request,
    prepare_deploy_rom_request,
    prepare_deploy_solve_request,
)
from blab.solvers.beat_engine_backend import (
    DEFAULT_BEAT_ENGINE_CPU_PROJECT,
    DEFAULT_BEAT_ENGINE_CUDA_PROJECT,
    DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT,
    BeatEngineWorkerProcess,
    shutdown_beat_engine_workers,
)
from blab.solvers.coupled_backend import DEFAULT_COUPLED_CPU_PROJECT, DEFAULT_COUPLED_SOLVER_SCRIPT
from blab.system_contract import system_frequency_result_from_dict

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


def _worker(backend: str, *, coupled: bool = False) -> BeatEngineWorkerProcess:
    normalized = backend.removeprefix("coupled:").removeprefix("rom:").strip().lower()
    if normalized not in {"cuda", "cpu"}:
        raise ValueError("Deploy worker backend must be cuda or cpu.")
    project = (
        DEFAULT_BEAT_ENGINE_CUDA_PROJECT
        if normalized == "cuda"
        else DEFAULT_COUPLED_CPU_PROJECT
        if coupled
        else DEFAULT_BEAT_ENGINE_CPU_PROJECT
    )
    return BeatEngineWorkerProcess(
        julia_executable=os.environ.get("BLAB_JULIA_EXE", "julia"),
        solver_script=DEFAULT_COUPLED_SOLVER_SCRIPT if coupled else DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT,
        julia_threads=os.environ.get("BLAB_JULIA_THREADS", "auto"),
        julia_project=project,
    )


def _worker_key(payload: object) -> str:
    backend = str(payload.get("backend", "cuda")) if isinstance(payload, dict) else "cuda"
    fidelity = str(payload.get("fidelity", "boundary")) if isinstance(payload, dict) else "boundary"
    normalized = backend.strip().lower()
    return f"coupled:{normalized}" if fidelity.strip().lower() == "coupled" else normalized


def _execution_worker_key(payload: object, solve_cache: DeploySolveCache) -> str:
    """Resolve Level 3 ROM jobs onto the exterior worker that executes them."""

    worker_key = _worker_key(payload)
    if not worker_key.startswith("coupled:") or not isinstance(payload, dict):
        return worker_key
    package_path = Path(str(payload.get("packagePath", ""))).expanduser().resolve()
    package = solve_cache.load_package(package_path)
    representation = package.coupled_model.get("representation") if isinstance(package.coupled_model, dict) else None
    return str(payload.get("backend", "cuda")).strip().lower() if representation == "parity_petrov_galerkin_rom" else worker_key


def _coupled_deploy_result(raw: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    parsed = system_frequency_result_from_dict(raw)
    quantity = next((item for item in parsed.quantities if item.id == "deploy:field-pressure"), None)
    if quantity is None:
        raise RuntimeError("BEAT Engine Level 3 solve returned no synthesized audience pressure.")
    pressure = np.asarray(quantity.values).reshape(-1)
    deploy = request["deploy"]
    sample_indices = [int(value) for value in deploy["sample_indices"]]
    if pressure.shape != (len(sample_indices),):
        raise RuntimeError("BEAT Engine Level 3 audience pressure has an unexpected shape.")
    spl_db = 20.0 * np.log10(np.maximum(np.abs(pressure), np.finfo(np.float32).tiny) / 20.0e-6)
    diagnostics = dict(parsed.diagnostics)
    timings = dict(diagnostics.get("timings", {}))
    diagnostics.update(
        backend=str(diagnostics.get("bem_backend", "unknown")),
        source_count=int(deploy["source_count"]),
        rigid_object_count=int(deploy["rigid_object_count"]),
        fidelity="coupled",
    )
    return {
        "frequency_hz": float(parsed.freq_hz),
        "rows": int(deploy["rows"]),
        "columns": int(deploy["columns"]),
        "spl_db": spl_db.astype(np.float32).tolist(),
        "sample_indices": sample_indices,
        "field_pressure": {
            "real": pressure.real.astype(np.float32).tolist(),
            "imag": pressure.imag.astype(np.float32).tolist(),
        },
        "timings": timings,
        "diagnostics": diagnostics,
    }


def _solve(
    request_id: object,
    payload: object,
    workers: dict[str, BeatEngineWorkerProcess],
    input_transport: dict[str, float | int],
    solve_cache: DeploySolveCache,
    solution_keys: dict[str, str],
) -> None:
    worker_key = _execution_worker_key(payload, solve_cache)
    coupled = worker_key.startswith("coupled:")
    rom = False
    if isinstance(payload, dict) and str(payload.get("fidelity", "boundary")).strip().lower() == "coupled":
        package_path = Path(str(payload.get("packagePath", ""))).expanduser().resolve()
        package = solve_cache.load_package(package_path)
        representation = (
            package.coupled_model.get("representation")
            if isinstance(package.coupled_model, dict)
            else None
        )
        if representation == "parity_petrov_galerkin_rom":
            # The ROM path uses the same BEAT solver process as Level 2, so it
            # benefits from the desktop's background CUDA warmup.
            rom = True
    worker = workers.get(worker_key)
    if worker is None:
        worker = _worker(worker_key, coupled=coupled)
        workers[worker_key] = worker

    with tempfile.TemporaryDirectory(prefix="blab-deploy-") as temp_dir:
        prepare_started = time.perf_counter()
        requested_solution_key = str(payload.get("solutionKey", "")) if isinstance(payload, dict) else ""
        reuse_boundary = bool(payload.get("reuseBoundary", False)) if isinstance(payload, dict) else False
        field_only = not coupled and reuse_boundary and bool(requested_solution_key) and (
            solution_keys.get(worker_key) == requested_solution_key
        )
        if field_only:
            request_path, _request = prepare_deploy_field_request(payload, temp_dir)
        elif coupled:
            request_path, _request = prepare_deploy_coupled_request(
                payload,
                temp_dir,
                cache=solve_cache,
                status_callback=lambda message: _emit("status", request_id=request_id, message=message),
            )
        elif rom:
            request_path, _request = prepare_deploy_rom_request(
                payload,
                temp_dir,
                cache=solve_cache,
                status_callback=lambda message: _emit("status", request_id=request_id, message=message),
            )
        else:
            request_path, _request = prepare_deploy_solve_request(
                payload,
                temp_dir,
                cache=solve_cache,
                status_callback=lambda message: _emit("status", request_id=request_id, message=message),
            )
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
                raw_result = event.get("result")
                if not isinstance(raw_result, dict):
                    raise RuntimeError("BEAT Engine Deploy solve returned an invalid result payload.")
                result = _coupled_deploy_result(raw_result, _request) if coupled else raw_result
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
                    solution_keys[worker_key] = str(_request.get("solution_key", requested_solution_key))
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
    fidelity = str(payload.get("fidelity", "boundary")).strip().lower()
    coupled_model = package_data.coupled_model if fidelity == "coupled" else None
    representation = coupled_model.get("representation") if isinstance(coupled_model, dict) else None
    frequencies = sorted({float(value) for value in package_data.frequencies})
    if representation == "exact_frequency_parametric_fem":
        band = coupled_model.get("frequency_band_hz", ())
        if isinstance(band, list) and len(band) == 2:
            lower, upper = map(float, band)
            tolerance = max(1e-4, max(abs(lower), abs(upper)) * 1e-6)
            frequencies = [value for value in frequencies if lower - tolerance <= value <= upper + tolerance]
    elif representation == "parity_petrov_galerkin_rom":
        arrays = coupled_model.get("arrays")
        rom_frequencies = np.asarray(arrays.get("frequencies_hz", ()), dtype=np.float64) if isinstance(arrays, dict) else np.empty(0)
        frequencies = [
            value for value in frequencies
            if rom_frequencies.size and np.min(np.abs(rom_frequencies - value)) <= max(1e-4, abs(value) * 1e-6)
        ]
    if not frequencies:
        raise ValueError("Speaker package contains no frequencies supported by the selected microphone sweep.")
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

    worker_key = _execution_worker_key(payload, solve_cache)
    exact_coupled = representation == "exact_frequency_parametric_fem"
    rom_coupled = representation == "parity_petrov_galerkin_rom"
    if fidelity == "coupled" and not (exact_coupled or rom_coupled):
        raise ValueError("Deploy Level 3 microphone sweep requires an exact or parity-ROM coupled package.")
    worker = workers.get(worker_key)
    if worker is None:
        worker = _worker(worker_key, coupled=exact_coupled)
        workers[worker_key] = worker
    julia_timing_totals: dict[str, float] = {}
    completed_count = 0
    with tempfile.TemporaryDirectory(prefix="blab-deploy-microphones-") as temp_dir:
        if cancel_event.is_set():
            _emit("cancelled", request_id=request_id, completed_count=completed_count)
            return
        sweep_payload = {
            **payload,
            "observationPointsM": observation_points,
            "includeComplexPressure": True,
            "reuseBoundary": False,
        }
        prepare_started = time.perf_counter()
        if exact_coupled:
            request_path, _request = prepare_deploy_coupled_request(
                {**sweep_payload, "frequenciesHz": frequencies},
                temp_dir,
                cache=solve_cache,
                status_callback=lambda message: _emit("status", request_id=request_id, message=message),
            )
        elif rom_coupled:
            request_path, _request = prepare_deploy_rom_microphone_sweep_request(
                {**sweep_payload, "frequenciesHz": frequencies},
                temp_dir,
                cache=solve_cache,
                status_callback=lambda message: _emit("status", request_id=request_id, message=message),
            )
        else:
            request_path, _request = prepare_deploy_microphone_sweep_request(
                sweep_payload,
                temp_dir,
                cache=solve_cache,
                status_callback=lambda message: _emit("status", request_id=request_id, message=message),
            )
        frequencies = [float(value) for value in _request["frequencies_hz"]]
        spl_rows: list[list[float]] = [[math.nan] * len(frequencies) for _ in microphone_ids]
        pressure_real_rows: list[list[float]] = [[math.nan] * len(frequencies) for _ in microphone_ids]
        pressure_imag_rows: list[list[float]] = [[math.nan] * len(frequencies) for _ in microphone_ids]
        frequency_indices = {frequency: index for index, frequency in enumerate(frequencies)}
        prepare_seconds = time.perf_counter() - prepare_started
        request_bytes = request_path.stat().st_size
        julia_started = time.perf_counter()
        for event in worker.submit(
            request_path,
            status_callback=lambda message: _emit("status", request_id=request_id, message=message),
        ):
            event_type = str(event.get("type", ""))
            if event_type == "status":
                _emit("status", request_id=request_id, message=str(event.get("message", "")))
            elif event_type == "initialized":
                _emit("initialized", request_id=request_id, metadata=event)
            elif event_type == "result":
                frequency_result = event.get("result")
                if not isinstance(frequency_result, dict):
                    raise RuntimeError("BEAT Engine microphone sweep returned an invalid result.")
                if exact_coupled:
                    frequency_result = _coupled_deploy_result(frequency_result, _request)
                frequency_hz = float(frequency_result.get("frequency_hz", math.nan))
                timings = frequency_result.get("timings")
                if isinstance(timings, dict):
                    for name, value in timings.items():
                        if isinstance(value, (int, float)) and math.isfinite(float(value)):
                            julia_timing_totals[name] = julia_timing_totals.get(name, 0.0) + float(value)
                frequency_index = frequency_indices.get(frequency_hz)
                if frequency_index is None:
                    frequency_index = min(
                        range(len(frequencies)),
                        key=lambda index: abs(frequencies[index] - frequency_hz),
                    )
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
                    spl_rows[microphone_index][frequency_index] = float(spl[microphone_index])
                    pressure_real_rows[microphone_index][frequency_index] = float(pressure_real[microphone_index])
                    pressure_imag_rows[microphone_index][frequency_index] = float(pressure_imag[microphone_index])
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
            elif event_type == "cancelled":
                _emit("cancelled", request_id=request_id, completed_count=completed_count)
                return
            elif event_type == "failed":
                raise RuntimeError(str(event.get("error", "BEAT Engine microphone sweep failed.")))
        if completed_count != len(frequencies):
            raise RuntimeError(
                f"BEAT Engine microphone sweep returned {completed_count} of {len(frequencies)} frequencies."
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
            "pipeline": {
                "python_prepare_s": prepare_seconds,
                "julia_request_json_bytes": request_bytes,
                "python_julia_result_wait_s": time.perf_counter() - julia_started,
                "batched_frequency_sweep": 1,
                **{f"julia_{name}_total_s": seconds for name, seconds in julia_timing_totals.items()},
            },
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
                        fidelity = str(message.get("fidelity", "boundary")).strip().lower()
                        worker_key = f"coupled:{backend}" if fidelity == "coupled" else backend
                        worker = workers.get(worker_key)
                        if worker is None:
                            worker = _worker(worker_key, coupled=fidelity == "coupled")
                            workers[worker_key] = worker
                        worker.ensure_started()
                        _emit("completed", request_id=request_id)
                        continue
                    if operation not in {"solve", "microphone_sweep"}:
                        raise ValueError("Unsupported Deploy worker operation.")
                    with active_lock:
                        if active["thread"] is not None:
                            raise RuntimeError("A Deploy solve is already in progress.")
                        payload = message.get("payload")
                        worker_key = _execution_worker_key(payload, solve_cache)
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
                            backend=worker_key,
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
