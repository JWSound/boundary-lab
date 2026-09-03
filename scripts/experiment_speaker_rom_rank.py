"""Measure parity-sector ROM rank for an exact Level 3 speaker interior."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from blab.deploy_solve import DeploySolveCache, prepare_deploy_coupled_request
from blab.solvers.beat_engine_backend import DEFAULT_BEAT_ENGINE_CUDA_PROJECT, BeatEngineWorkerProcess
from blab.solvers.coupled_backend import DEFAULT_COUPLED_SOLVER_SCRIPT

ROOT = Path(__file__).resolve().parents[1]


def _positive_csv(value: str) -> list[int]:
    parsed = sorted({int(item) for item in value.split(",") if item.strip()})
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="Exact Level 3 .blabsp package.")
    parser.add_argument("--frequency", type=float, default=100.0)
    parser.add_argument(
        "--package-frequency",
        type=float,
        default=100.0,
        help="In-band frequency used only while staging a single-frequency exploratory package.",
    )
    parser.add_argument("--train", type=int, default=128, help="Training fields per parity sector.")
    parser.add_argument("--test", type=int, default=32, help="Held-out fields per parity sector.")
    parser.add_argument("--ranks", type=_positive_csv, default=_positive_csv("8,16,32,64,96,128"))
    parser.add_argument("--julia", default="julia")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / ".speaker_rom_rank" / "s218bp-rank.json",
    )
    args = parser.parse_args()
    if args.train <= 0 or args.test <= 0:
        parser.error("--train and --test must be positive")

    work_dir = ROOT / "runs" / ".speaker_rom_rank" / "work"
    payload = {
        "packagePath": str(args.package.resolve()),
        # The exploratory exact package may have been exported at a single nominal
        # frequency even though its compiled FEM/BEM description is parametric.
        "frequencyHz": args.package_frequency,
        "backend": "cuda",
        "fidelity": "coupled",
        "sources": [
            {
                "id": "rank-cabinet",
                "positionX": 0.0,
                "positionHeightM": 0.7,
                "positionZ": 0.0,
                "pitchDeg": 0.0,
                "yawDeg": 0.0,
                "rollDeg": 0.0,
                "levelDb": 0.0,
                "delayMs": 0.0,
                "polarity": 1,
            }
        ],
        "rigidObjects": [],
        "observation": {
            "widthM": 1.0,
            "depthM": 1.0,
            "centerXM": 0.0,
            "nearM": 2.0,
            "heightM": 1.2,
            "pitchDeg": 0.0,
            "yawDeg": 0.0,
            "rollDeg": 0.0,
            "columns": 2,
            "rows": 2,
        },
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    request_path, _ = prepare_deploy_coupled_request(payload, work_dir, cache=DeploySolveCache())
    request = json.loads(request_path.read_text(encoding="utf-8"))
    options = request.setdefault("solver_options", {})
    options["speaker_rom_rank_experiment"] = {
        "train_per_sector": args.train,
        "test_per_sector": args.test,
        "ranks": args.ranks,
    }
    request["frequencies_hz"] = [args.frequency]
    request_path.write_text(json.dumps(request, separators=(",", ":")), encoding="utf-8")

    worker = BeatEngineWorkerProcess(
        julia_executable=args.julia,
        solver_script=DEFAULT_COUPLED_SOLVER_SCRIPT,
        julia_threads="auto",
        julia_project=DEFAULT_BEAT_ENGINE_CUDA_PROJECT,
    )
    result: dict[str, object] | None = None
    started = time.perf_counter()
    try:
        for event in worker.submit(request_path):
            event_type = str(event.get("type", ""))
            if event_type == "result":
                raw_result = event.get("result")
                if isinstance(raw_result, dict):
                    result = raw_result
            elif event_type == "failed":
                raise RuntimeError(str(event.get("error", "Speaker ROM rank experiment failed.")))
    finally:
        worker.terminate()
    if result is None:
        raise RuntimeError("Speaker ROM rank experiment returned no result.")

    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise RuntimeError("Speaker ROM rank experiment returned no diagnostics.")
    experiment = diagnostics.get("speaker_rom_rank_experiment")
    if not isinstance(experiment, dict):
        raise RuntimeError("Speaker ROM rank diagnostics are missing.")
    report = {
        "schema_version": 1,
        "package": str(args.package.resolve()),
        "wall_s": time.perf_counter() - started,
        "solve_timings": diagnostics.get("timings", {}),
        "experiment": experiment,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Rank experiment completed in {report['wall_s']:.2f}s; report: {args.output}")
    for sector in experiment["sectors"]:
        print(sector["sector"])
        for point in sector["curve"]:
            print(
                f"  r={point['rank']:>3}  train={point['training_energy_residual']:.3e}  "
                f"test p95={point['test_relative_error_p95']:.3e}  "
                f"max={point['test_relative_error_max']:.3e}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
