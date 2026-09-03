"""Validate every exported parity-ROM frequency against the exact Deploy oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from blab.deploy_solve import DeploySolveCache, prepare_deploy_coupled_request, prepare_deploy_rom_request
from blab.solvers.beat_engine_backend import (
    DEFAULT_BEAT_ENGINE_CUDA_PROJECT,
    DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT,
    BeatEngineWorkerProcess,
)
from blab.solvers.coupled_backend import DEFAULT_COUPLED_SOLVER_SCRIPT
from blab.system_contract import system_frequency_result_from_dict

ROOT = Path(__file__).resolve().parents[1]


def _payload(package: Path, cabinet_count: int, frequency_hz: float, spacing_m: float) -> dict[str, object]:
    center = (cabinet_count - 1) / 2.0
    return {
        "packagePath": str(package.resolve()),
        "frequencyHz": frequency_hz,
        "backend": "cuda",
        "fidelity": "coupled",
        "includeComplexPressure": True,
        "sources": [
            {
                "id": f"s218bp-{index + 1}",
                "positionX": (index - center) * spacing_m,
                "positionHeightM": 0.7,
                "positionZ": 0.0,
                "pitchDeg": 0.0,
                "yawDeg": 0.0,
                "rollDeg": 0.0,
                "levelDb": 0.0,
                "delayMs": 0.0,
                "polarity": 1,
            }
            for index in range(cabinet_count)
        ],
        "rigidObjects": [],
        "observation": {
            "widthM": 20.0,
            "depthM": 20.0,
            "centerXM": 0.0,
            "nearM": 2.0,
            "heightM": 1.2,
            "pitchDeg": 0.0,
            "yawDeg": 0.0,
            "rollDeg": 0.0,
            "columns": 17,
            "rows": 17,
        },
    }


def _package_frequencies(package: Path) -> np.ndarray:
    with zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    frequencies = np.asarray(manifest.get("frequencies_hz", ()), dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("ROM package contains no exported frequencies.")
    return frequencies


def _submit_all(worker: BeatEngineWorkerProcess, request_path: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for event in worker.submit(request_path):
        event_type = str(event.get("type", ""))
        if event_type == "result" and isinstance(event.get("result"), dict):
            results.append(event["result"])
        elif event_type == "failed":
            raise RuntimeError(str(event.get("error", "Deploy validation failed.")))
    return results


def _complex_payload(raw: object) -> np.ndarray:
    if not isinstance(raw, dict):
        raise ValueError("Complex result payload is missing.")
    return np.asarray(raw["real"], dtype=np.float64) + 1j * np.asarray(raw["imag"], dtype=np.float64)


def _relative_error(exact: np.ndarray, candidate: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(exact)), np.finfo(float).eps)
    return float(np.linalg.norm(candidate - exact) / denominator)


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _exact_cache_key(package: Path, frequencies: np.ndarray, cabinet_count: int, spacing_m: float) -> str:
    stat = package.stat()
    payload = json.dumps(
        {
            "package": str(package.resolve()),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "frequencies": frequencies.tolist(),
            "cabinet_count": cabinet_count,
            "spacing_m": spacing_m,
            "observation": [17, 17, 20.0, 20.0, 2.0, 1.2],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _solve_exact_band(
    package: Path,
    frequencies: np.ndarray,
    cabinet_count: int,
    spacing_m: float,
    julia: str,
    work_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_file = work_root / (f"exact-{_exact_cache_key(package, frequencies, cabinet_count, spacing_m)}.npz")
    if cache_file.is_file():
        with np.load(cache_file, allow_pickle=False) as cached:
            if np.array_equal(cached["frequencies_hz"], frequencies):
                print(f"Reusing exact oracle cache {cache_file}", flush=True)
                return cached["field"], cached["velocity"], cached["current"]

    exact_dir = work_root / "exact"
    exact_dir.mkdir(parents=True, exist_ok=True)
    request_path, request = prepare_deploy_coupled_request(
        _payload(package, cabinet_count, float(frequencies[0]), spacing_m),
        exact_dir,
        cache=DeploySolveCache(),
    )
    request["frequencies_hz"] = frequencies.tolist()
    request["outputs"].extend(
        (
            {
                "id": "validation:diaphragm-velocity",
                "quantity": "diaphragm_velocity",
                "target_ids": [],
                "options": {},
            },
            {
                "id": "validation:voice-coil-current",
                "quantity": "voice_coil_current",
                "target_ids": [],
                "options": {},
            },
        )
    )
    request_path.write_text(json.dumps(request, separators=(",", ":")), encoding="utf-8")
    weights = np.asarray(
        [complex(item["real"], item["imag"]) for item in request["outputs"][0]["options"]["excitation_weights"]]
    )
    worker = BeatEngineWorkerProcess(
        julia_executable=julia,
        solver_script=DEFAULT_COUPLED_SOLVER_SCRIPT,
        julia_threads="auto",
        julia_project=DEFAULT_BEAT_ENGINE_CUDA_PROJECT,
    )
    started = time.perf_counter()
    try:
        raw_results = _submit_all(worker, request_path)
    finally:
        worker.terminate()
    if len(raw_results) != len(frequencies):
        raise RuntimeError(f"Expected {len(frequencies)} exact results, received {len(raw_results)}.")

    fields: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    currents: list[np.ndarray] = []
    for index, raw in enumerate(raw_results):
        result = system_frequency_result_from_dict(raw)
        if not np.isclose(result.freq_hz, frequencies[index], rtol=1e-6, atol=1e-4):
            raise RuntimeError("Exact oracle frequencies were returned out of order.")
        quantities = {quantity.id: quantity for quantity in result.quantities}
        fields.append(np.asarray(quantities["deploy:field-pressure"].values).reshape(-1))
        velocities.append(weights @ np.asarray(quantities["validation:diaphragm-velocity"].values))
        currents.append(weights @ np.asarray(quantities["validation:voice-coil-current"].values))
        print(f"Exact {index + 1}/{len(frequencies)}: {result.freq_hz:.6g} Hz", flush=True)
    field = np.stack(fields)
    velocity = np.stack(velocities)
    current = np.stack(currents)
    np.savez_compressed(
        cache_file,
        frequencies_hz=frequencies,
        field=field,
        velocity=velocity,
        current=current,
    )
    print(f"Exact band completed in {time.perf_counter() - started:.1f}s", flush=True)
    return field, velocity, current


def _metric_summary(rows: list[dict[str, Any]], name: str) -> dict[str, float]:
    values = np.asarray([row[name] for row in rows], dtype=np.float64)
    maximum_index = int(np.argmax(values))
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "maximum": float(values[maximum_index]),
        "worst_frequency_hz": float(rows[maximum_index]["frequency_hz"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exact_package", type=Path)
    parser.add_argument("rom_package", type=Path)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--spacing-m", type=float, default=1.5)
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.count <= 8:
        parser.error("--count must be between 1 and 8")

    exact_package = args.exact_package.resolve()
    rom_package = args.rom_package.resolve()
    frequencies = _package_frequencies(rom_package)
    work_root = ROOT / "runs" / ".deploy_rom_band_validation" / f"{args.count}-cabinets"
    work_root.mkdir(parents=True, exist_ok=True)
    output = (args.output or work_root / "report.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    exact_field, exact_velocity, exact_current = _solve_exact_band(
        exact_package,
        frequencies,
        args.count,
        args.spacing_m,
        args.julia,
        work_root,
    )

    rows: list[dict[str, Any]] = []
    cache = DeploySolveCache()
    worker = BeatEngineWorkerProcess(
        julia_executable=args.julia,
        solver_script=DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT,
        julia_threads="auto",
        julia_project=DEFAULT_BEAT_ENGINE_CUDA_PROJECT,
    )
    geometry_key = f"s218bp-rom-band-{args.count}-{args.spacing_m:g}"
    rom_dir = work_root / "rom"
    rom_dir.mkdir(parents=True, exist_ok=True)
    try:
        worker.ensure_started()
        for index, frequency in enumerate(frequencies):
            request_path, request = prepare_deploy_rom_request(
                _payload(rom_package, args.count, float(frequency), args.spacing_m),
                rom_dir,
                cache=cache,
            )
            request["retain_geometry_cache"] = True
            request["geometry_key"] = geometry_key
            request["solution_key"] = f"{geometry_key}:{frequency:.12g}"
            request_path.write_text(json.dumps(request, separators=(",", ":")), encoding="utf-8")
            started = time.perf_counter()
            raw_results = _submit_all(worker, request_path)
            wall_seconds = time.perf_counter() - started
            if len(raw_results) != 1:
                raise RuntimeError(f"Expected one ROM result at {frequency:g} Hz.")
            raw = raw_results[0]
            rom_field = _complex_payload(raw["field_pressure"])
            diagnostics = raw["diagnostics"]
            rom_velocity = np.concatenate([_complex_payload(item) for item in diagnostics["transducer_velocity"]])
            rom_current = np.concatenate([_complex_payload(item) for item in diagnostics["transducer_current"]])
            exact_spl = 20.0 * np.log10(np.maximum(np.abs(exact_field[index]), np.finfo(float).tiny) / 20e-6)
            rom_spl = 20.0 * np.log10(np.maximum(np.abs(rom_field), np.finfo(float).tiny) / 20e-6)
            row = {
                "frequency_hz": float(frequency),
                "field_complex_relative_error": _relative_error(exact_field[index], rom_field),
                "field_spl_rms_error_db": float(np.sqrt(np.mean((rom_spl - exact_spl) ** 2))),
                "field_spl_max_error_db": float(np.max(np.abs(rom_spl - exact_spl))),
                "diaphragm_velocity_relative_error": _relative_error(exact_velocity[index], rom_velocity),
                "voice_coil_current_relative_error": _relative_error(exact_current[index], rom_current),
                "schur_gmres_iterations": int(diagnostics["schur_gmres_iterations"]),
                "schur_gmres_relative_residual": float(diagnostics["schur_gmres_relative_residual"]),
                "rom_worker_wall_s": wall_seconds,
            }
            rows.append(row)
            print(
                f"ROM {index + 1}/{len(frequencies)}: {frequency:.6g} Hz, "
                f"SPL RMS={row['field_spl_rms_error_db']:.4f} dB, "
                f"field={100.0 * row['field_complex_relative_error']:.3f}%, "
                f"wall={wall_seconds:.3f}s",
                flush=True,
            )
            _write_json(output, {"status": "running", "per_frequency": rows})
    finally:
        worker.terminate()

    metric_names = (
        "field_complex_relative_error",
        "field_spl_rms_error_db",
        "field_spl_max_error_db",
        "diaphragm_velocity_relative_error",
        "voice_coil_current_relative_error",
        "schur_gmres_relative_residual",
        "rom_worker_wall_s",
    )
    report = {
        "status": "completed",
        "exact_package": str(exact_package),
        "rom_package": str(rom_package),
        "cabinet_count": args.count,
        "frequency_count": len(frequencies),
        "frequency_band_hz": [float(frequencies[0]), float(frequencies[-1])],
        "summary": {name: _metric_summary(rows, name) for name in metric_names},
        "schur_gmres_iterations": {
            "median": float(np.median([row["schur_gmres_iterations"] for row in rows])),
            "maximum": int(max(row["schur_gmres_iterations"] for row in rows)),
        },
        "per_frequency": rows,
    }
    _write_json(output, report)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
