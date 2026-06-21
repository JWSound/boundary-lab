"""Benchmark Python-side live-solve GUI pipeline costs.

This script intentionally stays downstream of the solver backend. It can either
generate synthetic frequency results or collect real results from the Julia
local backend, then replay them through the same LiveSolveDataset visualization
preparation path used by the Qt GUI.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blab.config import MeshConfig, RadiatorConfig, SimulationConfig
from blab.live import LiveSolveDataset, build_log_frequencies
from blab.postprocess import PrepConfig
from blab.solvers.base import FrequencyResult, FrequencySolveTimings, SolverDiagnostics, SolveRequest
from blab.solvers.registry import create_backend

DEFAULT_SAMPLE_MESH = ROOT / "src" / "blab" / "solvers" / "julia_local" / "test_meshes" / "sample.msh"
DEFAULT_JULIA_EXE = r"C:\Users\John\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe"
DEFAULT_ISOBAR_ANGLE_SAMPLES = 250
DEFAULT_ISOBAR_FREQ_SAMPLES = 500


@dataclass(frozen=True)
class TimingStats:
    count: int
    total_s: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    max_ms: float


def _stats(samples_s: Iterable[float]) -> TimingStats:
    samples = np.asarray(list(samples_s), dtype=float)
    if samples.size == 0:
        return TimingStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return TimingStats(
        count=int(samples.size),
        total_s=float(np.sum(samples)),
        mean_ms=float(np.mean(samples) * 1000.0),
        median_ms=float(np.median(samples) * 1000.0),
        p95_ms=float(np.percentile(samples, 95) * 1000.0),
        max_ms=float(np.max(samples) * 1000.0),
    )


def _time_call(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def _angles(step_deg: float) -> np.ndarray:
    values = np.asarray(np.arange(-180.0, 180.0 + step_deg * 0.5, step_deg), dtype=np.float32)
    if values[-1] < 180.0:
        values = np.append(values, np.float32(180.0))
    return values.astype(np.float32, copy=False)


def _synthetic_results(
    frequencies: np.ndarray,
    angles: np.ndarray,
    *,
    radiator_count: int,
    sphere_points: int,
) -> list[FrequencyResult]:
    angle_rad = np.deg2rad(angles.astype(float))
    results: list[FrequencyResult] = []
    sphere_template = None
    if sphere_points > 0:
        phase = np.linspace(0.0, 2.0 * np.pi, sphere_points, endpoint=False, dtype=np.float32)
        sphere_template = -12.0 + 6.0 * np.cos(phase)

    for index, freq in enumerate(frequencies):
        directivity = -8.0 * (1.0 - np.cos(angle_rad))
        ripple = 1.5 * np.sin(angle_rad * 3.0 + index * 0.3)
        raw = 92.0 + 3.0 * np.sin(np.log(float(freq)) * 1.7) + directivity + ripple
        horizontal = raw.astype(np.float32)
        vertical = (raw - 1.5 * np.sin(angle_rad * 2.0)).astype(np.float32)
        reference = horizontal[int(np.argmin(np.abs(angles)))]
        impedance = np.asarray(
            [
                [6.0 + 0.25 * radiator + 0.1 * np.sin(index), 0.5 * np.cos(index * 0.2 + radiator)]
                for radiator in range(radiator_count)
            ],
            dtype=np.float32,
        )
        sphere = None
        if sphere_template is not None:
            sphere = (sphere_template + 0.3 * np.sin(index * 0.4)).astype(np.float32)
        results.append(
            FrequencyResult(
                freq_hz=float(freq),
                horizontal_spl_norm_db=(horizontal - reference).astype(np.float32),
                vertical_spl_norm_db=(vertical - reference).astype(np.float32),
                impedance=impedance,
                horizontal_spl_db=horizontal,
                vertical_spl_db=vertical,
                sphere_spl_norm_db=sphere,
                timings=FrequencySolveTimings(assembly_s=0.2, solve_s=0.02, field_s=0.03),
                diagnostics=SolverDiagnostics(convergence_info=0, message="synthetic"),
            )
        )
    return results


def _collect_julia_results(
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray] | None, list[FrequencyResult], dict]:
    frequencies = build_log_frequencies(args.freq_min, args.freq_max, args.freq_count)
    config = SimulationConfig(
        mesh_file=str(args.mesh),
        meshes=(MeshConfig(name="sample", file=str(args.mesh), scale_factor=args.scale_factor),),
        radiators=(RadiatorConfig(name="sample", tag=args.tag, mesh="sample"),),
        tag_throat=args.tag,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        freq_count=args.freq_count,
        step_size=args.angle_step,
        use_burton_miller=not args.no_burton_miller,
        flat_target_normalization_enabled=not args.no_flat_target,
        spherical_sampling_enabled=args.sphere_points > 0,
        spherical_sampling_points=max(1, args.sphere_points),
    )
    backend = create_backend(
        "julia_local",
        julia_executable=args.julia_executable,
        julia_threads=args.julia_threads,
        persistent_worker=not args.no_persistent_worker,
    )
    session = backend.create_session(SolveRequest(config, frequencies))
    metadata = session.metadata

    results: list[FrequencyResult] = []
    yield_intervals: list[float] = []
    result_parse_and_wait: list[float] = []
    started = time.perf_counter()
    last = started
    for result in session.solve_stream():
        now = time.perf_counter()
        yield_intervals.append(now - last)
        result_parse_and_wait.append(now - started if not results else now - last)
        last = now
        results.append(result)

    extra = {
        "solver_wall_s": time.perf_counter() - started,
        "yield_interval": asdict(_stats(yield_intervals)),
        "result_wait": asdict(_stats(result_parse_and_wait)),
    }
    return (
        frequencies,
        np.asarray(metadata.polar_angle_deg, dtype=np.float32),
        np.asarray(metadata.radiator_names),
        metadata.sphere_metadata,
        results,
        extra,
    )


def _sphere_metadata(point_count: int) -> dict[str, np.ndarray] | None:
    if point_count <= 0:
        return None
    theta = np.linspace(0.0, np.pi, point_count, dtype=np.float32)
    phi = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False, dtype=np.float32)
    return {
        "r_distance_m": np.full(point_count, 2.0, dtype=np.float32),
        "theta_polar_rad": theta,
        "phi_azimuth_rad": phi,
    }


def _dataset_for(
    angles: np.ndarray,
    radiator_names: np.ndarray,
    sphere_metadata: dict[str, np.ndarray] | None,
) -> LiveSolveDataset:
    sphere_metadata = sphere_metadata or {}
    return LiveSolveDataset(
        polar_angle_deg=np.asarray(angles, dtype=np.float32),
        radiator_names=np.asarray(radiator_names),
        sphere_r_distance_m=sphere_metadata.get("r_distance_m"),
        sphere_theta_polar_rad=sphere_metadata.get("theta_polar_rad"),
        sphere_phi_azimuth_rad=sphere_metadata.get("phi_azimuth_rad"),
    )


def _prep_config(args) -> PrepConfig:
    return PrepConfig(
        angle_samples=args.isobar_angle_samples,
        freq_samples=args.isobar_freq_samples,
        octave_smoothing=None if args.no_smoothing else args.octave_smoothing,
        hor_ref_angle=args.horizontal_ref_angle,
        vert_ref_angle=args.vertical_ref_angle,
        min_db=args.spl_min_db,
        max_db=args.spl_max_db,
        normalize_polar=True,
        auto_db_span=False,
    )


def _benchmark_replay(
    results: list[FrequencyResult],
    angles: np.ndarray,
    radiator_names: np.ndarray,
    sphere_metadata: dict[str, np.ndarray] | None,
    cfg: PrepConfig,
    *,
    refresh_every: int,
) -> dict:
    dataset = _dataset_for(angles, radiator_names, sphere_metadata)
    add_times: list[float] = []
    prep_every_times: list[float] = []

    for result in results:
        _, elapsed = _time_call(lambda result=result: dataset.add(result))
        add_times.append(elapsed)
        _, elapsed = _time_call(lambda: dataset.as_visualization_dataset(cfg))
        prep_every_times.append(elapsed)

    throttled_dataset = _dataset_for(angles, radiator_names, sphere_metadata)
    throttled_times: list[float] = []
    for index, result in enumerate(results, start=1):
        throttled_dataset.add(result)
        if index % refresh_every == 0 or index == len(results):
            _, elapsed = _time_call(lambda: throttled_dataset.as_visualization_dataset(cfg))
            throttled_times.append(elapsed)

    final_dataset = _dataset_for(angles, radiator_names, sphere_metadata)
    for result in results:
        final_dataset.add(result)
    _, final_elapsed = _time_call(lambda: final_dataset.as_visualization_dataset(cfg))

    return {
        "add_result": asdict(_stats(add_times)),
        "prep_every_result": asdict(_stats(prep_every_times)),
        "prep_throttled": asdict(_stats(throttled_times)),
        "final_prep_s": final_elapsed,
        "refresh_every": refresh_every,
    }


def _snapshot_dataset(dataset: LiveSolveDataset) -> LiveSolveDataset:
    snapshot = _dataset_for(
        dataset.polar_angle_deg,
        dataset.radiator_names,
        {
            "r_distance_m": dataset.sphere_r_distance_m,
            "theta_polar_rad": dataset.sphere_theta_polar_rad,
            "phi_azimuth_rad": dataset.sphere_phi_azimuth_rad,
        },
    )
    snapshot.results = dict(dataset.results)
    return snapshot


def _benchmark_latest_wins_thread(
    results: list[FrequencyResult],
    angles: np.ndarray,
    radiator_names: np.ndarray,
    sphere_metadata: dict[str, np.ndarray] | None,
    cfg: PrepConfig,
) -> dict:
    dataset = _dataset_for(angles, radiator_names, sphere_metadata)
    submit_times: list[float] = []
    prep_times: list[float] = []
    skipped_submissions = 0

    def prepare(snapshot: LiveSolveDataset) -> float:
        _, elapsed = _time_call(lambda: snapshot.as_visualization_dataset(cfg))
        return elapsed

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        submitted_count = 0
        for result in results:
            dataset.add(result)
            if pending is not None and not pending.done():
                skipped_submissions += 1
                continue
            if pending is not None:
                prep_times.append(float(pending.result()))
            pending, elapsed = _time_call(lambda: pool.submit(prepare, _snapshot_dataset(dataset)))
            submit_times.append(elapsed)
            submitted_count = dataset.solved_count
        if pending is not None:
            prep_times.append(float(pending.result()))
        if submitted_count < dataset.solved_count:
            pending, elapsed = _time_call(lambda: pool.submit(prepare, _snapshot_dataset(dataset)))
            submit_times.append(elapsed)
            prep_times.append(float(pending.result()))

    return {
        "submit_snapshot": asdict(_stats(submit_times)),
        "background_prep": asdict(_stats(prep_times)),
        "completed_preps": len(prep_times),
        "skipped_submissions": skipped_submissions,
    }


def _profile_final_prep(
    results: list[FrequencyResult],
    angles: np.ndarray,
    radiator_names: np.ndarray,
    sphere_metadata: dict[str, np.ndarray] | None,
    cfg: PrepConfig,
    *,
    limit: int,
) -> str:
    dataset = _dataset_for(angles, radiator_names, sphere_metadata)
    for result in results:
        dataset.add(result)

    profile = cProfile.Profile()
    profile.enable()
    dataset.as_visualization_dataset(cfg)
    profile.disable()

    import io

    output = io.StringIO()
    pstats.Stats(profile, stream=output).strip_dirs().sort_stats("cumtime").print_stats(limit)
    return output.getvalue()


def _estimate_result_payload_bytes(results: list[FrequencyResult]) -> int:
    total = 0
    for result in results:
        arrays = (
            result.horizontal_spl_norm_db,
            result.vertical_spl_norm_db,
            result.impedance,
            result.horizontal_spl_db,
            result.vertical_spl_db,
            result.sphere_spl_norm_db,
        )
        total += sum(0 if array is None else int(np.asarray(array).nbytes) for array in arrays)
    return total


def _sphere_point_count(sphere_metadata: dict[str, np.ndarray] | None) -> int:
    if not sphere_metadata:
        return 0
    values = sphere_metadata.get("r_distance_m")
    return 0 if values is None else int(len(values))


def _print_stats(name: str, stats: dict) -> None:
    print(
        f"{name:22s} count={stats['count']:4d} "
        f"mean={stats['mean_ms']:8.3f} ms  "
        f"p95={stats['p95_ms']:8.3f} ms  "
        f"max={stats['max_ms']:8.3f} ms  "
        f"total={stats['total_s']:8.3f} s"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "julia"), default="synthetic")
    parser.add_argument("--freq-min", type=float, default=200.0)
    parser.add_argument("--freq-max", type=float, default=20000.0)
    parser.add_argument("--freq-count", type=int, default=41)
    parser.add_argument("--angle-step", type=float, default=5.0)
    parser.add_argument("--radiators", type=int, default=1)
    parser.add_argument("--sphere-points", type=int, default=0)
    parser.add_argument("--isobar-angle-samples", type=int, default=DEFAULT_ISOBAR_ANGLE_SAMPLES)
    parser.add_argument("--isobar-freq-samples", type=int, default=DEFAULT_ISOBAR_FREQ_SAMPLES)
    parser.add_argument("--octave-smoothing", type=float, default=24.0)
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--horizontal-ref-angle", type=float, default=10.0)
    parser.add_argument("--vertical-ref-angle", type=float, default=10.0)
    parser.add_argument("--spl-min-db", type=float, default=-30.0)
    parser.add_argument("--spl-max-db", type=float, default=0.0)
    parser.add_argument("--refresh-every", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-limit", type=int, default=20)
    parser.add_argument("--json-out", type=Path)

    parser.add_argument("--mesh", type=Path, default=DEFAULT_SAMPLE_MESH)
    parser.add_argument("--tag", type=int, default=2)
    parser.add_argument("--scale-factor", type=float, default=0.001)
    parser.add_argument("--julia-executable", default=DEFAULT_JULIA_EXE)
    parser.add_argument("--julia-threads", default="auto")
    parser.add_argument("--no-persistent-worker", action="store_true")
    parser.add_argument("--no-burton-miller", action="store_true")
    parser.add_argument("--no-flat-target", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    refresh_every = max(1, int(args.refresh_every))
    cfg = _prep_config(args)

    if args.mode == "julia":
        frequencies, angles, radiator_names, sphere_metadata, results, collection = _collect_julia_results(args)
    else:
        frequencies = build_log_frequencies(args.freq_min, args.freq_max, args.freq_count)
        angles = _angles(args.angle_step)
        radiator_names = np.asarray([f"Radiator {index + 1}" for index in range(args.radiators)])
        sphere_metadata = _sphere_metadata(args.sphere_points)
        results = _synthetic_results(
            frequencies,
            angles,
            radiator_count=args.radiators,
            sphere_points=args.sphere_points,
        )
        collection = {}

    replay = _benchmark_replay(
        results,
        angles,
        radiator_names,
        sphere_metadata,
        cfg,
        refresh_every=refresh_every,
    )
    latest_wins = _benchmark_latest_wins_thread(results, angles, radiator_names, sphere_metadata, cfg)
    payload_bytes = _estimate_result_payload_bytes(results)

    report = {
        "mode": args.mode,
        "frequency_count": len(results),
        "angle_count": int(len(angles)),
        "radiator_count": int(len(radiator_names)),
        "sphere_points": _sphere_point_count(sphere_metadata),
        "payload_bytes_total": payload_bytes,
        "payload_bytes_per_frequency": payload_bytes / max(1, len(results)),
        "prep_config": {
            "angle_samples": cfg.angle_samples,
            "freq_samples": cfg.freq_samples,
            "octave_smoothing": cfg.octave_smoothing,
        },
        "collection": collection,
        "replay": replay,
        "latest_wins_thread": latest_wins,
    }

    print("Boundary Lab GUI pipeline benchmark")
    print(f"mode={args.mode} frequencies={len(results)} angles={len(angles)} radiators={len(radiator_names)}")
    print(
        f"payload={payload_bytes / 1024.0:.1f} KiB total ({payload_bytes / max(1, len(results)) / 1024.0:.1f} KiB/frequency)"
    )
    if collection:
        print(f"solver_wall={collection['solver_wall_s']:.3f} s")
        _print_stats("solver yield interval", collection["yield_interval"])
    print()
    _print_stats("add result", replay["add_result"])
    _print_stats("prep every result", replay["prep_every_result"])
    _print_stats("prep throttled", replay["prep_throttled"])
    print(f"final prep             {replay['final_prep_s'] * 1000.0:8.3f} ms")
    print()
    _print_stats("thread submit", latest_wins["submit_snapshot"])
    _print_stats("thread prep", latest_wins["background_prep"])
    print(
        f"latest-wins completed_preps={latest_wins['completed_preps']} "
        f"skipped_submissions={latest_wins['skipped_submissions']}"
    )

    if args.profile:
        print()
        print(_profile_final_prep(results, angles, radiator_names, sphere_metadata, cfg, limit=args.profile_limit))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
