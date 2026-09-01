"""Compare a parity-ROM Deploy solve with the exact Level 3 oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from blab.deploy_solve import (
    DeploySolveCache,
    prepare_deploy_coupled_request,
    prepare_deploy_rom_request,
)
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


def _submit(worker: BeatEngineWorkerProcess, request_path: Path) -> dict[str, object]:
    result = None
    for event in worker.submit(request_path):
        if event.get("type") == "result" and isinstance(event.get("result"), dict):
            result = event["result"]
        elif event.get("type") == "failed":
            raise RuntimeError(str(event.get("error", "Deploy validation failed.")))
    if not isinstance(result, dict):
        raise RuntimeError("Deploy validation returned no result.")
    return result


def _complex_payload(raw: object) -> np.ndarray:
    if not isinstance(raw, dict):
        raise ValueError("Complex result payload is missing.")
    return np.asarray(raw["real"], dtype=np.float64) + 1j * np.asarray(raw["imag"], dtype=np.float64)


def _relative_error(exact: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - exact) / max(np.linalg.norm(exact), np.finfo(float).eps))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exact_package", type=Path)
    parser.add_argument("rom_package", type=Path)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--frequency", type=float, default=100.0)
    parser.add_argument("--spacing-m", type=float, default=1.5)
    parser.add_argument("--julia", default="julia")
    args = parser.parse_args()

    cache = DeploySolveCache()
    work_root = ROOT / "runs" / ".deploy_rom_validation" / f"{args.count}-cabinets"
    exact_dir = work_root / "exact"
    rom_dir = work_root / "rom"
    exact_dir.mkdir(parents=True, exist_ok=True)
    rom_dir.mkdir(parents=True, exist_ok=True)
    exact_request_path, exact_request = prepare_deploy_coupled_request(
        _payload(args.exact_package, args.count, args.frequency, args.spacing_m),
        exact_dir,
        cache=cache,
    )
    exact_request["outputs"].extend(
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
    exact_request_path.write_text(json.dumps(exact_request, separators=(",", ":")), encoding="utf-8")
    rom_request_path, _rom_request = prepare_deploy_rom_request(
        _payload(args.rom_package, args.count, args.frequency, args.spacing_m),
        rom_dir,
        cache=cache,
    )

    exact_worker = BeatEngineWorkerProcess(
        julia_executable=args.julia,
        solver_script=DEFAULT_COUPLED_SOLVER_SCRIPT,
        julia_threads="auto",
        julia_project=DEFAULT_BEAT_ENGINE_CUDA_PROJECT,
    )
    rom_worker = BeatEngineWorkerProcess(
        julia_executable=args.julia,
        solver_script=DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT,
        julia_threads="auto",
        julia_project=DEFAULT_BEAT_ENGINE_CUDA_PROJECT,
    )
    try:
        exact_raw = _submit(exact_worker, exact_request_path)
        rom_raw = _submit(rom_worker, rom_request_path)
    finally:
        exact_worker.terminate()
        rom_worker.terminate()

    exact = system_frequency_result_from_dict(exact_raw)
    exact_quantities = {quantity.id: quantity for quantity in exact.quantities}
    exact_field = np.asarray(exact_quantities["deploy:field-pressure"].values).reshape(-1)
    weights = np.asarray(
        [complex(item["real"], item["imag"]) for item in exact_request["outputs"][0]["options"]["excitation_weights"]]
    )
    exact_velocity = weights @ np.asarray(exact_quantities["validation:diaphragm-velocity"].values)
    exact_current = weights @ np.asarray(exact_quantities["validation:voice-coil-current"].values)
    rom_field = _complex_payload(rom_raw["field_pressure"])
    diagnostics = rom_raw["diagnostics"]
    rom_velocity = np.concatenate([_complex_payload(item) for item in diagnostics["transducer_velocity"]])
    rom_current = np.concatenate([_complex_payload(item) for item in diagnostics["transducer_current"]])

    exact_spl = 20 * np.log10(np.maximum(np.abs(exact_field), np.finfo(float).tiny) / 20e-6)
    rom_spl = 20 * np.log10(np.maximum(np.abs(rom_field), np.finfo(float).tiny) / 20e-6)
    report = {
        "cabinet_count": args.count,
        "frequency_hz": args.frequency,
        "field_complex_relative_error": _relative_error(exact_field, rom_field),
        "field_spl_rms_error_db": float(np.sqrt(np.mean((rom_spl - exact_spl) ** 2))),
        "field_spl_max_error_db": float(np.max(np.abs(rom_spl - exact_spl))),
        "diaphragm_velocity_relative_error": _relative_error(exact_velocity, rom_velocity),
        "voice_coil_current_relative_error": _relative_error(exact_current, rom_current),
        "schur_gmres_iterations": diagnostics["schur_gmres_iterations"],
        "schur_gmres_relative_residual": diagnostics["schur_gmres_relative_residual"],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
