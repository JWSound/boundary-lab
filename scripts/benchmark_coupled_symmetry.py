"""Benchmark FP32 coupled full/X/XY SAWMOD solves and compare exterior pressure."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np

from blab.interface_conform import conform_interface_mesh_files
from blab.physical_compiler import PhysicalSystemCompiler
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
)
from blab.solvers.coupled_backend import CoupledProductionBackend
from blab.system_contract import OutputRequest, SystemSolveRequest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "SAWMOD"


@dataclass(frozen=True)
class ModeSpec:
    symmetry: str
    suffix: str
    bem_interface: str
    bem_exterior: str
    volume: str
    wall: str
    symmetry_surface: str | None
    bem_suffix: str | None = None


MODE_SPECS = {
    "off": ModeSpec(
        "off",
        "full",
        "BEMInterface",
        "BEMExt",
        "SAWMODFEM",
        "SAWMODFEM_boundary",
        None,
    ),
    "x": ModeSpec(
        "x",
        "reduced_x",
        "BEMInterface (1)",
        "BEMExt (1)",
        "SAWMODFEM (1)",
        "SAWMODFEM (1)_boundary",
        "SYMMETRY",
    ),
    "xy": ModeSpec(
        "xy",
        "reduced_xy",
        "BEMInterface (1) (1)",
        "BEMExt (1) (1)",
        "SAWMODFEM (1) (1)",
        "SAWMODFEM (1) (1)_boundary",
        "SYMMETRY",
    ),
    "xy-detailed": ModeSpec(
        "xy",
        "reduced_xy_detailed",
        "BEMInterface (1) (1)",
        "BEMExt (1) (1)",
        "SAWMODFEM (1) (1)",
        "SAWMODFEM (1) (1)_boundary",
        None,
        bem_suffix="reduced_xy",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        default="off,x,xy",
        help="Comma-separated subset of off,x,xy,xy-detailed.",
    )
    parser.add_argument("--frequencies", default="500", help="Comma-separated frequencies in Hz.")
    parser.add_argument(
        "--bem-backend",
        choices=("cpu", "cuda", "rocm", "both", "cpu-cuda", "cpu-rocm"),
        default="cpu",
    )
    parser.add_argument("--julia", default="julia")
    parser.add_argument(
        "--julia-threads",
        help="Julia worker thread count or 'auto' (defaults by backend).",
    )
    parser.add_argument(
        "--solver-script",
        type=Path,
        help="Optional Julia coupled-worker entry point, primarily for profiler wrappers.",
    )
    parser.add_argument("--quadrature-order", type=int, default=2)
    parser.add_argument("--singular-order", type=int, default=2)
    parser.add_argument("--validation-diagnostics", action="store_true")
    parser.add_argument(
        "--static-condensation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable exact FEM interface condensation "
            "(CUDA and ROCm default to enabled outside validation diagnostics)."
        ),
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Conform and compile the selected meshes, then report topology and dense-storage estimates without solving.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "diagnostics" / "coupled_symmetry",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    modes = tuple(value.strip().lower() for value in args.modes.split(",") if value.strip())
    if not modes or any(mode not in MODE_SPECS for mode in modes):
        parser.error("--modes must contain off, x, xy, and/or xy-detailed.")
    frequencies = tuple(float(value) for value in args.frequencies.split(",") if value.strip())
    if not frequencies or any(value <= 0.0 for value in frequencies):
        parser.error("--frequencies must contain positive values.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.inspect_only:
        inspections = {}
        for mode in modes:
            spec = MODE_SPECS[mode]
            system = _system_for_mode(spec, output_dir)
            compiled = PhysicalSystemCompiler().compile(
                system,
                symmetry_mode=spec.symmetry,
            )
            inspections[mode] = _topology_report(mode, spec.symmetry, system, compiled)
            print(_format_topology_report(inspections[mode]))
        payload = {
            "schema_version": 1,
            "precision": "float32",
            "inspections": inspections,
        }
        _write_payload(args.output_json, payload)
        return 0

    points = _observation_points()
    backend_names = {
        "both": ("cpu", "cuda"),
        "cpu-cuda": ("cpu", "cuda"),
        "cpu-rocm": ("cpu", "rocm"),
    }.get(args.bem_backend, (args.bem_backend,))
    reports_by_backend: dict[str, dict[str, dict[str, object]]] = {}
    pressure_by_backend: dict[str, dict[str, np.ndarray]] = {}

    for backend_name in backend_names:
        backend_options = {
            "bem_backend": backend_name,
            "julia_executable": args.julia,
            "julia_threads": args.julia_threads,
        }
        if args.solver_script is not None:
            backend_options["solver_script"] = args.solver_script.resolve()
        backend = CoupledProductionBackend(
            **backend_options,
        )
        reports: dict[str, dict[str, object]] = {}
        pressure_by_mode: dict[str, np.ndarray] = {}
        for mode in modes:
            spec = MODE_SPECS[mode]
            system = _system_for_mode(spec, output_dir)
            compiled = PhysicalSystemCompiler().compile(
                system,
                symmetry_mode=spec.symmetry,
            )
            solver_options = {
                "quadrature_order": args.quadrature_order,
                "singular_order": args.singular_order,
                "validation_diagnostics": args.validation_diagnostics,
                "cache_frequency_invariant": True,
                "symmetry": spec.symmetry,
                "static_condensation": backend_name in {"cuda", "rocm"} and not args.validation_diagnostics,
            }
            if args.static_condensation is not None:
                solver_options["static_condensation"] = args.static_condensation
            request = SystemSolveRequest(
                compiled_system=compiled,
                frequencies_hz=frequencies,
                excitation_port_ids=tuple(port.id for port in compiled.excitation_ports),
                outputs=(
                    OutputRequest(
                        id="field",
                        quantity="exterior_pressure",
                        options={"points_m": points.tolist()},
                    ),
                ),
                solver_options=solver_options,
            )
            started = time.perf_counter()
            results = list(backend.create_system_session(request).solve_stream())
            wall_s = time.perf_counter() - started
            pressure = np.stack(
                [
                    next(quantity for quantity in result.quantities if quantity.id == "field").values
                    for result in results
                ]
            )
            pressure_by_mode[mode] = pressure
            reports[mode] = _mode_report(mode, spec.symmetry, compiled, results, wall_s)
            print(f"{backend_name.upper():>4} {_format_mode_report(reports[mode])}")

        reference_mode = "off" if "off" in pressure_by_mode else modes[0]
        reference = pressure_by_mode[reference_mode]
        for mode in modes:
            comparison = _pressure_comparison(reference, pressure_by_mode[mode])
            reports[mode]["comparison_to"] = reference_mode
            reports[mode]["pressure_comparison"] = comparison
            print(
                f"{backend_name.upper():>4} {mode.upper():>3} vs {reference_mode.upper():>3}: "
                f"relative L2={comparison['relative_l2']:.4e}, "
                f"relative max={comparison['relative_max']:.4e}, "
                f"max |dB|={comparison['max_abs_db']:.3f}"
            )
        reports_by_backend[backend_name] = reports
        pressure_by_backend[backend_name] = pressure_by_mode

    if len(backend_names) == 2:
        comparison_backend = backend_names[1]
        for mode in modes:
            comparison = _pressure_comparison(
                pressure_by_backend["cpu"][mode],
                pressure_by_backend[comparison_backend][mode],
            )
            reports_by_backend[comparison_backend][mode]["cpu_comparison"] = comparison
            print(
                f"{comparison_backend.upper()} {mode.upper():>3} vs CPU: "
                f"relative L2={comparison['relative_l2']:.4e}, "
                f"relative max={comparison['relative_max']:.4e}, "
                f"max |dB|={comparison['max_abs_db']:.3f}"
            )

    payload = {
        "schema_version": 1,
        "precision": "float32",
        "bem_backend": args.bem_backend,
        "frequencies_hz": list(frequencies),
        "observation_points_m": points.tolist(),
        "backends": {backend_name: {"modes": reports_by_backend[backend_name]} for backend_name in backend_names},
    }
    if len(backend_names) == 1:
        payload["modes"] = reports_by_backend[backend_names[0]]
    _write_payload(args.output_json, payload)
    return 0


def _system_for_mode(spec: ModeSpec, output_dir: Path) -> PhysicalSystem:
    fem_path = FIXTURE_ROOT / f"SMfemvolume_{spec.suffix}.msh"
    raw_bem_path = FIXTURE_ROOT / f"SMexterior_{spec.bem_suffix or spec.suffix}.msh"
    bem_path = output_dir / f"SMexterior_{spec.suffix}_conforming.msh"
    conform_interface_mesh_files(
        fem_path,
        raw_bem_path,
        bem_path,
        fem_interface_name="INTERFACE",
        bem_interface_name=spec.bem_interface,
        symmetry_mode=spec.symmetry,
    )
    boundaries = [
        Boundary(
            "boundary:bem-exterior",
            spec.bem_exterior,
            "region:exterior",
            PhysicalGroupRef("mesh:bem", 2, name=spec.bem_exterior),
            BoundaryKind.RIGID,
        ),
        Boundary(
            "boundary:bem-interface",
            spec.bem_interface,
            "region:exterior",
            PhysicalGroupRef("mesh:bem", 2, name=spec.bem_interface),
            BoundaryKind.INTERFACE,
        ),
        Boundary(
            "boundary:hf",
            "HF",
            "region:interior",
            PhysicalGroupRef("mesh:fem", 2, name="HF"),
            BoundaryKind.MOVING,
        ),
        Boundary(
            "boundary:mf",
            "MF",
            "region:interior",
            PhysicalGroupRef("mesh:fem", 2, name="MF"),
            BoundaryKind.MOVING,
        ),
        Boundary(
            "boundary:fem-interface",
            "INTERFACE",
            "region:interior",
            PhysicalGroupRef("mesh:fem", 2, name="INTERFACE"),
            BoundaryKind.INTERFACE,
        ),
        Boundary(
            "boundary:wall",
            spec.wall,
            "region:interior",
            PhysicalGroupRef("mesh:fem", 2, name=spec.wall),
            BoundaryKind.RIGID,
        ),
    ]
    if spec.symmetry_surface is not None:
        boundaries.append(
            Boundary(
                "boundary:symmetry",
                spec.symmetry_surface,
                "region:interior",
                PhysicalGroupRef("mesh:fem", 2, name=spec.symmetry_surface),
                BoundaryKind.RIGID,
            )
        )
    return PhysicalSystem(
        id=f"system:sawmod-{spec.symmetry}",
        name=f"SAWMOD {spec.symmetry.upper()}",
        meshes=(
            MeshResource(
                "mesh:fem",
                f"SAWMOD FEM {spec.symmetry}",
                str(fem_path),
                MeshPurpose.FEM_VOLUME,
                scale_to_m=0.001,
            ),
            MeshResource(
                "mesh:bem",
                f"SAWMOD BEM {spec.symmetry}",
                str(bem_path),
                MeshPurpose.BEM_SURFACE,
                scale_to_m=0.001,
            ),
        ),
        regions=(
            AcousticRegion(
                "region:interior",
                "Interior Air",
                AcousticRegionKind.BOUNDED_AIR,
                ("mesh:fem",),
                volume_groups=(PhysicalGroupRef("mesh:fem", 3, name=spec.volume),),
            ),
            AcousticRegion(
                "region:exterior",
                "Exterior Air",
                AcousticRegionKind.UNBOUNDED_AIR,
                ("mesh:bem",),
            ),
        ),
        boundaries=tuple(boundaries),
        interfaces=(
            AcousticInterface(
                "interface:port",
                "FEM / BEM",
                "boundary:fem-interface",
                "boundary:bem-interface",
            ),
        ),
        components=(
            PhysicalComponent(
                "component:hf",
                "HF",
                ComponentKind.IDEAL_VELOCITY_SOURCE,
                ("boundary:hf",),
                {"motion_profile": "uniform"},
            ),
            PhysicalComponent(
                "component:mf",
                "MF",
                ComponentKind.IDEAL_VELOCITY_SOURCE,
                ("boundary:mf",),
                {"motion_profile": "uniform"},
            ),
        ),
        excitation_ports=(
            ExcitationPort(
                id="excitation:hf",
                name="HF unit normal velocity",
                component_id="component:hf",
                kind=ExcitationPortKind.NORMAL_VELOCITY,
            ),
            ExcitationPort(
                id="excitation:mf",
                name="MF unit normal velocity",
                component_id="component:mf",
                kind=ExcitationPortKind.NORMAL_VELOCITY,
            ),
        ),
    )


def _observation_points() -> np.ndarray:
    radius = 5.0
    angles = np.deg2rad(np.asarray([-90.0, -45.0, 0.0, 45.0, 90.0]))
    horizontal = np.column_stack((radius * np.sin(angles), np.zeros_like(angles), radius * np.cos(angles)))
    vertical = np.column_stack((np.zeros_like(angles), radius * np.sin(angles), radius * np.cos(angles)))
    return np.vstack((horizontal, vertical))


def _topology_report(mode, symmetry, system, compiled) -> dict[str, object]:
    fem_resource = next(mesh for mesh in system.meshes if mesh.purpose == MeshPurpose.FEM_VOLUME)
    bem_resource = next(mesh for mesh in system.meshes if mesh.purpose == MeshPurpose.BEM_SURFACE)
    fem_mesh = meshio.read(fem_resource.file)
    bem_mesh = meshio.read(bem_resource.file)
    fem_nodes = len(fem_mesh.points)
    fem_tetrahedra = sum(len(block.data) for block in fem_mesh.cells if block.type in {"tetra", "tetra4"})
    bem_nodes = len(bem_mesh.points)
    bem_faces = sum(len(block.data) for block in bem_mesh.cells if block.type in {"triangle", "triangle3"})
    topology = compiled.interfaces[0].topology
    interface_nodes = len(topology.fem_vertex_indices)
    interface_faces = len(topology.fem_face_indices)
    system_order = fem_nodes + bem_nodes + interface_nodes
    bytes_per_complex = np.dtype(np.complex64).itemsize
    return {
        "mode": mode,
        "symmetry": symmetry,
        "fem_nodes": fem_nodes,
        "fem_tetrahedra": fem_tetrahedra,
        "bem_nodes": bem_nodes,
        "bem_faces": bem_faces,
        "interface_nodes": interface_nodes,
        "interface_faces": interface_faces,
        "coupled_order": system_order,
        "fem_dense_gib": fem_nodes * fem_nodes * bytes_per_complex / 2**30,
        "coupled_dense_gib": system_order * system_order * bytes_per_complex / 2**30,
    }


def _format_topology_report(report: dict[str, object]) -> str:
    return (
        f"{str(report['mode']).upper():>11}: "
        f"FEM={report['fem_nodes']} nodes/{report['fem_tetrahedra']} tets, "
        f"BEM={report['bem_nodes']} nodes/{report['bem_faces']} faces, "
        f"interface={report['interface_nodes']} nodes/{report['interface_faces']} faces, "
        f"coupled order={report['coupled_order']}, "
        f"dense={report['coupled_dense_gib']:.2f} GiB"
    )


def _write_payload(output_json: Path | None, payload: dict[str, object]) -> None:
    if output_json is None:
        return
    resolved = output_json.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {resolved}")


def _mode_report(
    mode,
    symmetry,
    compiled,
    results,
    wall_s: float,
) -> dict[str, object]:
    timing_keys = (
        "mesh_setup_s",
        "cache_setup_s",
        "fem_matrix_cache_s",
        "interface_operator_cache_s",
        "bem_space_cache_s",
        "bem_singular_cache_s",
        "bem_cpu_assembly_cache_s",
        "bem_device_regular_cache_s",
        "bem_device_singular_cache_s",
        "bem_device_image_cache_s",
        "bem_identity_cache_s",
        "device_block_cache_s",
        "field_cache_s",
        "assembly_s",
        "fem_system_s",
        "bem_operator_s",
        "bem_matrix_s",
        "fem_condensation_s",
        "fem_condensation_analysis_s",
        "fem_condensation_partition_s",
        "fem_condensation_factorization_s",
        "fem_schur_extraction_s",
        "fem_schur_upload_s",
        "fem_rhs_condensation_s",
        "fem_reconstruction_s",
        "block_assembly_s",
        "coupled_factorization_s",
        "solve_s",
        "field_s",
    )
    timings = {
        key: float(sum(float(result.diagnostics.get("timings", {}).get(key, 0.0)) for result in results))
        for key in timing_keys
    }
    frequency_timings = [
        {
            "frequency_hz": float(result.freq_hz),
            **{key: float(result.diagnostics.get("timings", {}).get(key, 0.0)) for key in timing_keys},
        }
        for result in results
    ]
    topology = compiled.interfaces[0].topology
    return {
        "mode": mode,
        "symmetry": symmetry,
        "bem_backend": str(results[0].diagnostics["bem_backend"]),
        "linear_backend": str(results[0].diagnostics["linear_backend"]),
        "linear_solver": str(results[0].diagnostics["linear_solver"]),
        "formulation": str(results[0].diagnostics["formulation"]),
        "fem_condensation_backend": results[0].diagnostics.get("fem_condensation_backend"),
        "fem_schur_block_size": int(results[0].diagnostics.get("fem_schur_block_size", 0)),
        "fem_schur_thread_count": int(results[0].diagnostics.get("fem_schur_thread_count", 0)),
        "static_condensation_requested": bool(results[0].diagnostics.get("static_condensation_requested", False)),
        "static_condensation_active": bool(results[0].diagnostics.get("static_condensation_active", False)),
        "full_system_order": int(results[0].diagnostics["full_system_order"]),
        "solved_system_order": int(results[0].diagnostics["solved_system_order"]),
        "wall_s": float(wall_s),
        "frequency_count": len(results),
        "interface_vertices": len(topology.fem_vertex_indices),
        "interface_faces": len(topology.fem_face_indices),
        "bem_open_symmetry_edges": topology.bem_boundary_edges,
        "timings_s": timings,
        "frequency_timings_s": frequency_timings,
        "max_pressure_continuity_error": max(
            float(result.diagnostics["pressure_continuity_error"]) for result in results
        ),
        "max_flux_conservation_error": max(float(result.diagnostics["flux_conservation_error"]) for result in results),
    }


def _format_mode_report(report: dict[str, object]) -> str:
    timings = report["timings_s"]
    assert isinstance(timings, dict)
    return (
        f"{str(report['mode']).upper():>11}: wall={report['wall_s']:.3f}s, "
        f"interface={report['interface_vertices']} nodes/{report['interface_faces']} faces, "
        f"order={report['solved_system_order']}/{report['full_system_order']}, "
        f"formulation={report['formulation']}, "
        f"assembly={timings['assembly_s']:.3f}s, solve={timings['solve_s']:.3f}s, "
        f"field={timings['field_s']:.3f}s"
    )


def _pressure_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    scale = max(float(np.linalg.norm(reference)), np.finfo(np.float32).tiny)
    peak = max(float(np.max(np.abs(reference), initial=0.0)), np.finfo(np.float32).tiny)
    magnitudes_reference = np.maximum(np.abs(reference), peak * 1e-8)
    magnitudes_candidate = np.maximum(np.abs(candidate), peak * 1e-8)
    return {
        "relative_l2": float(np.linalg.norm(candidate - reference) / scale),
        "relative_max": float(np.max(np.abs(candidate - reference), initial=0.0) / peak),
        "max_abs_db": float(np.max(np.abs(20.0 * np.log10(magnitudes_candidate / magnitudes_reference)), initial=0.0)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
