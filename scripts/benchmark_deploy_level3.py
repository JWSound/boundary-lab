"""Benchmark exact Level 3 arrays through the Deploy preparation and Julia worker path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from blab.deploy_solve import DeploySolveCache, prepare_deploy_coupled_request
from blab.solvers.beat_engine_backend import DEFAULT_BEAT_ENGINE_CUDA_PROJECT, BeatEngineWorkerProcess
from blab.solvers.coupled_backend import DEFAULT_COUPLED_CPU_PROJECT, DEFAULT_COUPLED_SOLVER_SCRIPT

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--counts", default="1,4,8", help="Comma-separated cabinet counts.")
    parser.add_argument("--frequency", type=float, default=100.0)
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--columns", type=int, default=33)
    parser.add_argument("--rows", type=int, default=33)
    parser.add_argument("--spacing-m", type=float, default=1.5)
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--json", action="store_true", help="Emit one JSON object per run.")
    args = parser.parse_args()
    counts = tuple(int(value) for value in args.counts.split(",") if value.strip())
    if not counts or any(value < 1 or value > 8 for value in counts):
        parser.error("--counts must contain values from 1 through 8.")
    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be non-negative and --repeat must be positive.")

    project = DEFAULT_BEAT_ENGINE_CUDA_PROJECT if args.backend == "cuda" else DEFAULT_COUPLED_CPU_PROJECT
    worker = BeatEngineWorkerProcess(
        julia_executable=args.julia,
        solver_script=DEFAULT_COUPLED_SOLVER_SCRIPT,
        julia_threads="auto",
        julia_project=project,
    )
    cache = DeploySolveCache()
    try:
        worker.ensure_started()
        for count in counts:
            for run_index in range(args.warmup + args.repeat):
                phase = "warmup" if run_index < args.warmup else "measured"
                result = _run_once(
                    worker,
                    cache,
                    package=args.package,
                    cabinet_count=count,
                    frequency_hz=args.frequency,
                    backend=args.backend,
                    columns=args.columns,
                    rows=args.rows,
                    spacing_m=args.spacing_m,
                )
                result.update(phase=phase, run=run_index + 1 if phase == "warmup" else run_index - args.warmup + 1)
                if args.json:
                    print(json.dumps(result, separators=(",", ":")), flush=True)
                else:
                    timings = result["timings"]
                    print(
                        f"{count} cabinets {phase} {result['run']}: wall={result['wall_s']:.3f}s "
                        f"prepare={result['prepare_s']:.3f}s assembly={timings.get('assembly_s', 0.0):.3f}s "
                        f"solve={timings.get('solve_s', 0.0):.3f}s field={timings.get('field_s', 0.0):.3f}s "
                        f"order={result.get('solved_system_order', 'unknown')}",
                        flush=True,
                    )
    finally:
        worker.terminate()
    return 0


def _run_once(
    worker: BeatEngineWorkerProcess,
    cache: DeploySolveCache,
    *,
    package: Path,
    cabinet_count: int,
    frequency_hz: float,
    backend: str,
    columns: int,
    rows: int,
    spacing_m: float,
) -> dict[str, object]:
    center = (cabinet_count - 1) / 2.0
    payload = {
        "packagePath": str(package.resolve()),
        "frequencyHz": frequency_hz,
        "backend": backend,
        "fidelity": "coupled",
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
            "columns": columns,
            "rows": rows,
        },
    }
    work_dir = ROOT / "runs" / ".deploy_level3_benchmark" / f"{cabinet_count}-cabinets"
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    request_path, _ = prepare_deploy_coupled_request(payload, work_dir, cache=cache)
    prepared = time.perf_counter()
    raw_result = None
    for event in worker.submit(request_path):
        event_type = str(event.get("type", ""))
        if event_type == "result":
            raw_result = event.get("result")
        elif event_type == "failed":
            raise RuntimeError(str(event.get("error", "Level 3 benchmark failed.")))
    finished = time.perf_counter()
    if not isinstance(raw_result, dict):
        raise RuntimeError("Level 3 benchmark returned no frequency result.")
    diagnostics = dict(raw_result.get("diagnostics", {}))
    return {
        "cabinet_count": cabinet_count,
        "frequency_hz": frequency_hz,
        "wall_s": finished - started,
        "prepare_s": prepared - started,
        "worker_s": finished - prepared,
        "timings": diagnostics.get("timings", {}),
        "full_system_order": diagnostics.get("full_system_order"),
        "solved_system_order": diagnostics.get("solved_system_order"),
        "linear_solver": diagnostics.get("linear_solver"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
