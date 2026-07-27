"""Benchmark the coupled Julia backend through its application wire path."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from blab.physical_model import (
    AcousticInterface,
    AcousticRegion,
    AcousticRegionKind,
    Boundary,
    BoundaryKind,
    ComponentKind,
    ExcitationPort,
    ExcitationPortKind,
    MeshPurpose,
    MeshResource,
    PhysicalComponent,
    PhysicalGroupRef,
    PhysicalSystem,
    physical_system_from_dict,
)
from blab.solvers.coupled_backend import (
    CoupledProductionBackend,
    CoupledReferenceBackend,
)
from blab.ui.project_io import read_project_file
from blab.ui.system_solve import prepare_coupled_ui_solve

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, help="Boundary Lab project containing a physical_system.")
    parser.add_argument("--julia", default="julia", help="Julia executable or absolute path.")
    parser.add_argument("--julia-threads", default="1", help="Julia worker thread count or 'auto'.")
    parser.add_argument("--frequencies", default="500,1000", help="Comma-separated frequencies in Hz.")
    parser.add_argument("--angle-step", type=float, default=90.0, help="Polar observation angle step.")
    parser.add_argument("--quadrature-order", type=int, default=2)
    parser.add_argument("--singular-order", type=int, default=2)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--bem-backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--symmetry", choices=("off", "x", "xy"), default="off")
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="Reuse the Julia process across benchmark modes, matching application behavior.",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Number of runs per mode.")
    parser.add_argument(
        "--mode",
        choices=("interactive", "reference", "uncached", "all"),
        default="all",
        help="Interactive skips replay diagnostics; uncached also disables invariant caching.",
    )
    args = parser.parse_args()
    frequencies = tuple(float(value.strip()) for value in args.frequencies.split(",") if value.strip())
    if not frequencies:
        parser.error("--frequencies must contain at least one value.")
    system = _fixture_system() if args.project is None else _system_from_project(args.project)
    modes = ("interactive", "reference", "uncached") if args.mode == "all" else (args.mode,)
    for mode in modes:
        for run_index in range(max(1, args.repeat)):
            _run_mode(
                system,
                mode=mode,
                run_index=run_index + 1,
                frequencies=frequencies,
                angle_step=args.angle_step,
                julia_executable=args.julia,
                julia_threads=args.julia_threads,
                quadrature_order=args.quadrature_order,
                singular_order=args.singular_order,
                persistent_worker=args.persistent,
                precision=args.precision,
                bem_backend=args.bem_backend,
                symmetry_mode=args.symmetry,
            )
    return 0


def _run_mode(
    system: PhysicalSystem,
    *,
    mode: str,
    run_index: int,
    frequencies: tuple[float, ...],
    angle_step: float,
    julia_executable: str,
    julia_threads: str,
    quadrature_order: int,
    singular_order: int,
    persistent_worker: bool,
    precision: str,
    bem_backend: str,
    symmetry_mode: str,
) -> None:
    prepared = prepare_coupled_ui_solve(
        system,
        freq_min_hz=min(frequencies),
        freq_max_hz=max(frequencies),
        freq_count=len(frequencies),
        observation_distance_m=2.0,
        polar_angle_step_deg=angle_step,
        symmetry_mode=symmetry_mode,
    )
    options = dict(prepared.request.solver_options)
    options["quadrature_order"] = int(quadrature_order)
    options["singular_order"] = int(singular_order)
    options["validation_diagnostics"] = mode != "interactive"
    options["cache_frequency_invariant"] = mode != "uncached"
    request = replace(
        prepared.request,
        frequencies_hz=frequencies,
        solver_options=options,
    )

    if precision == "float64":
        if bem_backend != "cpu":
            raise ValueError("Float64 is reserved for the CPU coupled reference backend.")
        backend = CoupledReferenceBackend(
            julia_executable=julia_executable,
            julia_threads=julia_threads,
            persistent_worker=persistent_worker,
        )
    else:
        backend = CoupledProductionBackend(
            bem_backend=bem_backend,
            julia_executable=julia_executable,
            julia_threads=julia_threads,
            persistent_worker=persistent_worker,
        )
    started = time.perf_counter()
    results = list(backend.create_system_session(request).solve_stream())
    wall_s = time.perf_counter() - started
    measured_s = 0.0
    detail_keys = (
        "fem_system_s",
        "bem_operator_s",
        "bem_matrix_s",
        "block_assembly_s",
        "coupled_factorization_s",
        "replay_factorization_s",
    )
    detail_totals = {key: 0.0 for key in detail_keys}
    print(
        f"\n{mode.upper()} run={run_index}  precision={precision}  bem={bem_backend}"
        f"  symmetry={symmetry_mode}  frequencies={len(results)}  wall={wall_s:.3f}s"
    )
    print("frequency       assembly       solve       field       setup")
    for result in results:
        timings = result.diagnostics.get("timings", {})
        assembly_s = float(timings.get("assembly_s", 0.0))
        solve_s = float(timings.get("solve_s", 0.0))
        field_s = float(timings.get("field_s", 0.0))
        setup_s = float(timings.get("mesh_setup_s", 0.0)) + float(timings.get("cache_setup_s", 0.0))
        for key in detail_keys:
            detail_totals[key] += float(timings.get(key, 0.0))
        measured_s += assembly_s + solve_s + field_s + setup_s
        print(
            f"{result.freq_hz:8.1f} Hz"
            f"  {assembly_s:10.3f}s"
            f"  {solve_s:8.3f}s"
            f"  {field_s:8.3f}s"
            f"  {setup_s:8.3f}s"
        )
    print(f"reported stages={measured_s:.3f}s  process startup/JIT/serialization={wall_s - measured_s:.3f}s")
    print(
        "assembly detail: "
        + "  ".join(f"{key.removesuffix('_s')}={value:.3f}s" for key, value in detail_totals.items())
    )


def _system_from_project(path: Path) -> PhysicalSystem:
    payload = read_project_file(path)
    raw = payload.get("physical_system")
    if not isinstance(raw, dict):
        raise ValueError(f"Project does not contain a physical_system: {path}")
    return physical_system_from_dict(raw)


def _fixture_system() -> PhysicalSystem:
    fem_path = FIXTURE_ROOT / "femvolume.msh"
    bem_path = FIXTURE_ROOT / "exterior_conforming.msh"
    return PhysicalSystem(
        id="system:benchmark",
        name="Coupled reference benchmark",
        meshes=(
            MeshResource(
                id="mesh:fem",
                name="Interior",
                file=str(fem_path),
                purpose=MeshPurpose.FEM_VOLUME,
                scale_to_m=0.001,
            ),
            MeshResource(
                id="mesh:bem",
                name="Exterior",
                file=str(bem_path),
                purpose=MeshPurpose.BEM_SURFACE,
                scale_to_m=0.001,
            ),
        ),
        regions=(
            AcousticRegion(
                id="region:interior",
                name="Interior Air",
                kind=AcousticRegionKind.BOUNDED_AIR,
                mesh_ids=("mesh:fem",),
                volume_groups=(PhysicalGroupRef("mesh:fem", 3, name="Volume"),),
            ),
            AcousticRegion(
                id="region:exterior",
                name="Exterior Air",
                kind=AcousticRegionKind.UNBOUNDED_AIR,
                mesh_ids=("mesh:bem",),
            ),
        ),
        boundaries=(
            Boundary(
                "boundary:radiator",
                "Radiator",
                "region:interior",
                PhysicalGroupRef("mesh:fem", 2, name="Radiator"),
                BoundaryKind.MOVING,
            ),
            Boundary(
                "boundary:wall",
                "Walls",
                "region:interior",
                PhysicalGroupRef("mesh:fem", 2, name="Volume_boundary"),
                BoundaryKind.RIGID,
            ),
            Boundary(
                "boundary:fem-interface",
                "Interface",
                "region:interior",
                PhysicalGroupRef("mesh:fem", 2, name="Interface"),
                BoundaryKind.INTERFACE,
            ),
            Boundary(
                "boundary:exterior",
                "Exterior",
                "region:exterior",
                PhysicalGroupRef("mesh:bem", 2, name="ExteriorBox"),
                BoundaryKind.RIGID,
            ),
            Boundary(
                "boundary:bem-interface",
                "Interface",
                "region:exterior",
                PhysicalGroupRef("mesh:bem", 2, name="Interface"),
                BoundaryKind.INTERFACE,
            ),
        ),
        interfaces=(
            AcousticInterface(
                "interface:port",
                "Port",
                "boundary:fem-interface",
                "boundary:bem-interface",
            ),
        ),
        components=(
            PhysicalComponent(
                "component:radiator",
                "Radiator",
                ComponentKind.IDEAL_VELOCITY_SOURCE,
                ("boundary:radiator",),
                {"motion_profile": "uniform"},
            ),
        ),
        excitation_ports=(
            ExcitationPort(
                "excitation:radiator",
                "Radiator unit normal velocity",
                "component:radiator",
                ExcitationPortKind.NORMAL_VELOCITY,
            ),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
