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


def _worker_key(payload: object) -> str:
    backend = str(payload.get("backend", "cuda")) if isinstance(payload, dict) else "cuda"
    normalized = backend.strip().lower()
    return normalized


def _execution_worker_key(payload: object, solve_cache: DeploySolveCache) -> str:
    """Resolve Level 3 ROM jobs onto the exterior worker that executes them."""

    worker_key = _worker_key(payload)
    if not isinstance(payload, dict) or str(payload.get("fidelity", "boundary")).strip().lower() != "coupled":
        return worker_key
    package_path = Path(str(payload.get("packagePath", ""))).expanduser().resolve()
    package = solve_cache.load_package(package_path)
    representation = package.coupled_model.get("representation") if isinstance(package.coupled_model, dict) else None
    if representation != "parity_petrov_galerkin_rom":
        raise ValueError("Deploy Level 3 requires a parity Petrov–Galerkin ROM package.")
    return worker_key


def _transducer_velocity_result(result: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Flatten per-instance ROM diaphragm velocities into stable scene traces."""

    raw_descriptors = request.get("transducers", [])
    diagnostics = result.get("diagnostics", {})
    raw_instances = diagnostics.get("transducer_velocity", []) if isinstance(diagnostics, dict) else []
    if not isinstance(raw_descriptors, list) or not isinstance(raw_instances, list):
        return {"ids": [], "names": [], "real": [], "imag": []}
    real: list[float] = []
    imag: list[float] = []
    for instance in raw_instances:
        if not isinstance(instance, dict):
            raise RuntimeError("BEAT Engine transducer velocity result is invalid.")
        instance_real = instance.get("real")
        instance_imag = instance.get("imag")
        if not isinstance(instance_real, list) or not isinstance(instance_imag, list):
            raise RuntimeError("BEAT Engine transducer velocity result is invalid.")
        if len(instance_real) != len(instance_imag):
            raise RuntimeError("BEAT Engine transducer velocity real and imaginary counts differ.")
        real.extend(float(value) for value in instance_real)
        imag.extend(float(value) for value in instance_imag)
    if len(real) != len(raw_descriptors):
        raise RuntimeError("BEAT Engine transducer velocity result does not match the scene transducer count.")
    return {
        "ids": [str(item["id"]) for item in raw_descriptors],
        "names": [str(item["name"]) for item in raw_descriptors],
        "real": real,
        "imag": imag,
    }


def _speaker_electrical_result(
    result: dict[str, Any],
    request: dict[str, Any],
    frequency_index: int,
) -> dict[str, Any]:
    """Aggregate ROM coil currents and applied RMS voltage per cabinet instance."""

    speakers = request.get("speakers", [])
    transducers = request.get("transducers", [])
    diagnostics = result.get("diagnostics", {})
    raw_currents = diagnostics.get("transducer_current", []) if isinstance(diagnostics, dict) else []
    if not isinstance(speakers, list) or not isinstance(transducers, list) or not isinstance(raw_currents, list):
        return {"ids": [], "names": [], "voltage_real": [], "voltage_imag": [], "current_real": [], "current_imag": []}
    if not speakers:
        return {"ids": [], "names": [], "voltage_real": [], "voltage_imag": [], "current_real": [], "current_imag": []}
    rom_sweep = request.get("rom_sweep", {})
    sweep_frequencies = rom_sweep.get("frequencies", []) if isinstance(rom_sweep, dict) else []
    if frequency_index >= len(sweep_frequencies):
        raise RuntimeError("BEAT Engine electrical result has no matching ROM drive entry.")
    drive_instances = sweep_frequencies[frequency_index].get("instances", [])
    if len(raw_currents) != len(speakers) or len(drive_instances) != len(speakers):
        raise RuntimeError("BEAT Engine electrical result does not match the scene speaker count.")
    voltage_real: list[float] = []
    voltage_imag: list[float] = []
    current_real: list[float] = []
    current_imag: list[float] = []
    for speaker, raw_current, drive in zip(speakers, raw_currents, drive_instances, strict=True):
        real_values = raw_current.get("real") if isinstance(raw_current, dict) else None
        imag_values = raw_current.get("imag") if isinstance(raw_current, dict) else None
        input_real = drive.get("input_real") if isinstance(drive, dict) else None
        input_imag = drive.get("input_imag") if isinstance(drive, dict) else None
        if not all(
            isinstance(values, list) and values for values in (real_values, imag_values, input_real, input_imag)
        ):
            raise RuntimeError("BEAT Engine electrical current or voltage result is invalid.")
        if len(real_values) != len(imag_values):
            raise RuntimeError("BEAT Engine coil-current real and imaginary counts differ.")
        speaker_transducers = [item for item in transducers if item.get("source_id") == speaker.get("id")]
        if len(speaker_transducers) != len(real_values):
            raise RuntimeError("BEAT Engine coil-current result does not match the cabinet transducer count.")
        total = sum(
            complex(float(real), float(imag)) * int(descriptor.get("physical_driver_orbit_count", 1))
            for real, imag, descriptor in zip(real_values, imag_values, speaker_transducers, strict=True)
        )
        voltage_real.append(float(input_real[0]))
        voltage_imag.append(float(input_imag[0]))
        current_real.append(float(total.real))
        current_imag.append(float(total.imag))
    return {
        "ids": [str(item["id"]) for item in speakers],
        "names": [str(item["name"]) for item in speakers],
        "voltage_real": voltage_real,
        "voltage_imag": voltage_imag,
        "current_real": current_real,
        "current_imag": current_imag,
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
    rom = False
    if isinstance(payload, dict) and str(payload.get("fidelity", "boundary")).strip().lower() == "coupled":
        package_path = Path(str(payload.get("packagePath", ""))).expanduser().resolve()
        package = solve_cache.load_package(package_path)
        representation = (
            package.coupled_model.get("representation") if isinstance(package.coupled_model, dict) else None
        )
        if representation == "parity_petrov_galerkin_rom":
            # The ROM path uses the same BEAT solver process as Level 2, so it
            # benefits from the desktop's background CUDA warmup.
            rom = True
    worker = workers.get(worker_key)
    if worker is None:
        worker = _worker(worker_key)
        workers[worker_key] = worker

    with tempfile.TemporaryDirectory(prefix="blab-deploy-") as temp_dir:
        prepare_started = time.perf_counter()
        requested_solution_key = str(payload.get("solutionKey", "")) if isinstance(payload, dict) else ""
        reuse_boundary = bool(payload.get("reuseBoundary", False)) if isinstance(payload, dict) else False
        field_only = (
            reuse_boundary
            and bool(requested_solution_key)
            and (solution_keys.get(worker_key) == requested_solution_key)
        )
        if field_only:
            request_path, _request = prepare_deploy_field_request(payload, temp_dir)
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
                result = raw_result
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
    if representation == "parity_petrov_galerkin_rom":
        arrays = coupled_model.get("arrays")
        rom_frequencies = (
            np.asarray(arrays.get("frequencies_hz", ()), dtype=np.float64) if isinstance(arrays, dict) else np.empty(0)
        )
        frequencies = [
            value
            for value in frequencies
            if rom_frequencies.size and np.min(np.abs(rom_frequencies - value)) <= max(1e-4, abs(value) * 1e-6)
        ]
    if not frequencies:
        raise ValueError("Speaker package contains no frequencies supported by the selected microphone sweep.")
    raw_microphones = payload.get("microphones")
    if not isinstance(raw_microphones, list) or (not raw_microphones and fidelity != "coupled"):
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
    rom_coupled = representation == "parity_petrov_galerkin_rom"
    if fidelity == "coupled" and not rom_coupled:
        raise ValueError("Deploy Level 3 microphone sweep requires a parity-ROM package.")
    worker = workers.get(worker_key)
    if worker is None:
        worker = _worker(worker_key)
        workers[worker_key] = worker
    julia_timing_totals: dict[str, float] = {}
    completed_count = 0
    with tempfile.TemporaryDirectory(prefix="blab-deploy-microphones-") as temp_dir:
        if cancel_event.is_set():
            _emit("cancelled", request_id=request_id, completed_count=completed_count)
            return
        # The shared coupled analysis sweep can produce transducer motion with
        # no microphones. The exterior solver still needs one evaluation point;
        # its dummy pressure is deliberately discarded below.
        evaluation_points = observation_points or [[0.0, 1.0, 1.0]]
        sweep_payload = {
            **payload,
            "observationPointsM": evaluation_points,
            "includeComplexPressure": True,
            "reuseBoundary": False,
        }
        prepare_started = time.perf_counter()
        if rom_coupled:
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
        raw_transducers = _request.get("transducers", [])
        transducer_ids = [str(item["id"]) for item in raw_transducers] if isinstance(raw_transducers, list) else []
        transducer_names = [str(item["name"]) for item in raw_transducers] if isinstance(raw_transducers, list) else []
        velocity_real_rows: list[list[float]] = [[math.nan] * len(frequencies) for _ in transducer_ids]
        velocity_imag_rows: list[list[float]] = [[math.nan] * len(frequencies) for _ in transducer_ids]
        raw_speakers = _request.get("speakers", [])
        speaker_ids = [str(item["id"]) for item in raw_speakers] if isinstance(raw_speakers, list) else []
        speaker_names = [str(item["name"]) for item in raw_speakers] if isinstance(raw_speakers, list) else []
        voltage_real_rows: list[list[float]] = [[math.nan] * len(frequencies) for _ in speaker_ids]
        voltage_imag_rows: list[list[float]] = [[math.nan] * len(frequencies) for _ in speaker_ids]
        current_real_rows: list[list[float]] = [[math.nan] * len(frequencies) for _ in speaker_ids]
        current_imag_rows: list[list[float]] = [[math.nan] * len(frequencies) for _ in speaker_ids]
        frequency_indices = {frequency: index for index, frequency in enumerate(frequencies)}
        prepare_seconds = time.perf_counter() - prepare_started
        request_bytes = request_path.stat().st_size
        provenance = _request.get("provenance", {})
        rom_stage_metrics = (
            {
                "rom_sweep_stage_cache_hit": int(provenance.get("rom_sweep_stage_cache_hit", 0)),
                "rom_sweep_stage_binary_bytes": int(provenance.get("rom_sweep_stage_binary_bytes", 0)),
                "rom_sweep_stage_binary_bytes_written": int(provenance.get("rom_sweep_stage_binary_bytes_written", 0)),
            }
            if rom_coupled and isinstance(provenance, dict)
            else {}
        )
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
                if microphone_ids and (not isinstance(spl, list) or len(spl) != len(microphone_ids)):
                    raise RuntimeError("BEAT Engine microphone SPL result does not match the microphone count.")
                if microphone_ids and not isinstance(pressure, dict):
                    raise RuntimeError("BEAT Engine microphone sweep did not return complex pressure.")
                pressure_real = pressure.get("real") if isinstance(pressure, dict) else []
                pressure_imag = pressure.get("imag") if isinstance(pressure, dict) else []
                if microphone_ids and (not isinstance(pressure_real, list) or not isinstance(pressure_imag, list)):
                    raise RuntimeError("BEAT Engine microphone pressure result is invalid.")
                if microphone_ids and (
                    len(pressure_real) != len(microphone_ids) or len(pressure_imag) != len(microphone_ids)
                ):
                    raise RuntimeError("BEAT Engine microphone pressure result does not match the microphone count.")
                for microphone_index in range(len(microphone_ids)):
                    spl_rows[microphone_index][frequency_index] = float(spl[microphone_index])
                    pressure_real_rows[microphone_index][frequency_index] = float(pressure_real[microphone_index])
                    pressure_imag_rows[microphone_index][frequency_index] = float(pressure_imag[microphone_index])
                transducer_velocity = (
                    _transducer_velocity_result(frequency_result, _request)
                    if rom_coupled
                    else {
                        "ids": [],
                        "names": [],
                        "real": [],
                        "imag": [],
                    }
                )
                if transducer_velocity["ids"] != transducer_ids:
                    raise RuntimeError("BEAT Engine transducer ordering changed during the frequency sweep.")
                for transducer_index in range(len(transducer_ids)):
                    velocity_real_rows[transducer_index][frequency_index] = transducer_velocity["real"][
                        transducer_index
                    ]
                    velocity_imag_rows[transducer_index][frequency_index] = transducer_velocity["imag"][
                        transducer_index
                    ]
                electrical = (
                    _speaker_electrical_result(frequency_result, _request, frequency_index)
                    if rom_coupled
                    else {
                        "ids": [],
                        "names": [],
                        "voltage_real": [],
                        "voltage_imag": [],
                        "current_real": [],
                        "current_imag": [],
                    }
                )
                if electrical["ids"] != speaker_ids:
                    raise RuntimeError("BEAT Engine speaker ordering changed during the frequency sweep.")
                for speaker_index in range(len(speaker_ids)):
                    voltage_real_rows[speaker_index][frequency_index] = electrical["voltage_real"][speaker_index]
                    voltage_imag_rows[speaker_index][frequency_index] = electrical["voltage_imag"][speaker_index]
                    current_real_rows[speaker_index][frequency_index] = electrical["current_real"][speaker_index]
                    current_imag_rows[speaker_index][frequency_index] = electrical["current_imag"][speaker_index]
                completed_count += 1
                _emit(
                    "microphone-progress",
                    request_id=request_id,
                    frequency_hz=frequency_hz,
                    completed_count=completed_count,
                    total_count=len(frequencies),
                    microphone_ids=microphone_ids,
                    spl_db=spl,
                    transducer_ids=transducer_ids,
                    transducer_names=transducer_names,
                    transducer_velocity={
                        "real": transducer_velocity["real"],
                        "imag": transducer_velocity["imag"],
                    },
                    speaker_ids=speaker_ids,
                    speaker_names=speaker_names,
                    speaker_voltage={"real": electrical["voltage_real"], "imag": electrical["voltage_imag"]},
                    speaker_current={"real": electrical["current_real"], "imag": electrical["current_imag"]},
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
            "transducer_ids": transducer_ids,
            "transducer_names": transducer_names,
            "transducer_velocity": {"real": velocity_real_rows, "imag": velocity_imag_rows},
            "speaker_ids": speaker_ids,
            "speaker_names": speaker_names,
            "speaker_voltage": {"real": voltage_real_rows, "imag": voltage_imag_rows},
            "speaker_current": {"real": current_real_rows, "imag": current_imag_rows},
            "completed_count": completed_count,
            "total_count": len(frequencies),
            "pipeline": {
                "python_prepare_s": prepare_seconds,
                "julia_request_json_bytes": request_bytes,
                "python_julia_result_wait_s": time.perf_counter() - julia_started,
                "batched_frequency_sweep": 1,
                **rom_stage_metrics,
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
                        worker_key = backend
                        worker = workers.get(worker_key)
                        if worker is None:
                            worker = _worker(worker_key)
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
        solve_cache.close()
        shutdown_beat_engine_workers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
