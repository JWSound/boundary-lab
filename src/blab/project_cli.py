"""Command-line entry points for validating and solving Boundary Lab projects."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from blab.fem_validation import (
    FEMConvergenceGates,
    FEMValidationGates,
    compare_fem_validation_reports,
    evaluate_fem_run,
    write_fem_validation_report,
)
from blab.headless import (
    HEADLESS_BACKEND_AUTO,
    HEADLESS_BACKEND_IDS,
    default_result_path,
    load_headless_project,
    load_headless_solve_spec,
    prepare_headless_solve,
    resolve_headless_backend,
    run_headless_solve,
    validation_summary,
)
from blab.speaker_package import (
    SpeakerPackageConfig,
    SpeakerPackageCoupledRepresentation,
    SpeakerPackageFidelity,
    export_speaker_package,
    prepare_speaker_package_solve,
    solve_speaker_package_system,
)
from blab.speaker_preflight import estimate_level_three_package
from blab.speaker_symmetry import (
    expand_speaker_system_for_export,
    preferred_full_meshes_from_project_payload,
)


def _build_arg_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Validate or solve a .blab.json physical-system project without opening the GUI.",
    )
    commands = parser.add_subparsers(dest="project_command", required=True)

    validate = commands.add_parser("validate", help="Load, compile, and validate a project and solve request")
    validate.add_argument("project_file", type=Path, help="Path to the .blab.json project")
    validate.add_argument("--request", type=Path, help="Optional headless solve-request JSON overlay")
    validate.add_argument("--backend", choices=HEADLESS_BACKEND_IDS, default=HEADLESS_BACKEND_AUTO)
    validate.add_argument(
        "--julia-executable",
        default=os.environ.get("BLAB_JULIA_EXE", "julia"),
        help="Julia executable used to probe automatic CUDA selection",
    )
    validate.add_argument("--json", action="store_true", help="Print the validation summary as JSON")

    solve = commands.add_parser("solve", help="Run a physical-system project solve")
    solve.add_argument("project_file", type=Path, help="Path to the .blab.json project")
    solve.add_argument("--request", type=Path, help="Optional headless solve-request JSON overlay")
    solve.add_argument("--backend", choices=HEADLESS_BACKEND_IDS, default=HEADLESS_BACKEND_AUTO)
    solve.add_argument("--output", type=Path, help="New result directory; defaults below the project runs directory")
    solve.add_argument(
        "--events",
        choices=("text", "ndjson"),
        default="text",
        help="Progress event format",
    )
    solve.add_argument(
        "--julia-executable",
        default=os.environ.get("BLAB_JULIA_EXE", "julia"),
        help="Julia executable used by BEAT Engine",
    )
    solve.add_argument("--julia-threads", default=None, help="Julia thread count, or auto")

    evaluate_fem = commands.add_parser(
        "evaluate-fem",
        help="Evaluate phase coherence and plane-mode purity on tagged FEM exit surfaces",
    )
    evaluate_fem.add_argument("run_dir", type=Path, help="Completed headless result directory")
    evaluate_fem.add_argument(
        "--surface-pattern",
        action="append",
        dest="surface_patterns",
        help="Physical surface glob; repeatable (default: exit_*)",
    )
    evaluate_fem.add_argument("--output", type=Path, help="Report path (default: RUN_DIR/fem-validation.json)")
    evaluate_fem.add_argument("--max-within-phase-rms-deg", type=float, default=10.0)
    evaluate_fem.add_argument("--max-inter-phase-rms-deg", type=float, default=5.0)
    evaluate_fem.add_argument("--max-inter-phase-deg", type=float, default=10.0)
    evaluate_fem.add_argument("--min-plane-mode-fraction", type=float, default=0.95)
    evaluate_fem.add_argument("--min-points-per-wavelength-p95", type=float, default=8.0)
    evaluate_fem.add_argument("--min-points-per-wavelength-maximum-edge", type=float, default=4.0)
    evaluate_fem.add_argument(
        "--split-surface-entities",
        action="store_true",
        help="Evaluate each Gmsh geometrical entity within a physical surface separately",
    )

    compare_fem = commands.add_parser(
        "compare-fem",
        help="Compare tagged-surface fields from coarse and fine FEM validation reports",
    )
    compare_fem.add_argument("coarse_report", type=Path)
    compare_fem.add_argument("fine_report", type=Path)
    compare_fem.add_argument("--output", type=Path)
    compare_fem.add_argument("--max-phase-rms-delta-deg", type=float, default=1.0)
    compare_fem.add_argument("--max-phase-delta-deg", type=float, default=2.0)
    compare_fem.add_argument("--max-normalized-amplitude-rms-delta", type=float, default=0.02)
    compare_fem.add_argument("--max-plane-mode-fraction-delta", type=float, default=0.01)

    export_speaker = commands.add_parser(
        "export-speaker",
        help="Solve a project and export a level 1, 2, or 3 .blabsp speaker package",
    )
    export_speaker.add_argument("project_file", type=Path, help="Path to the .blab.json project")
    export_speaker.add_argument("--output", type=Path, required=True, help="Output .blabsp package path")
    export_speaker.add_argument("--name", help="Package display name; defaults to the physical-system name")
    export_speaker.add_argument("--fidelity", choices=("pattern", "fixed", "coupled"), default="pattern")
    export_speaker.add_argument("--speaker-rom-rank", type=int, default=32)
    export_speaker.add_argument("--speaker-rom-training-count", type=int, default=96)
    export_speaker.add_argument("--speaker-rom-validation-count", type=int, default=24)
    export_speaker.add_argument("--request", type=Path, help="Optional headless solve-request JSON overlay")
    export_speaker.add_argument("--backend", choices=HEADLESS_BACKEND_IDS, default=HEADLESS_BACKEND_AUTO)
    export_speaker.add_argument("--events", choices=("text", "ndjson"), default="text")
    export_speaker.add_argument(
        "--julia-executable",
        default=os.environ.get("BLAB_JULIA_EXE", "julia"),
        help="Julia executable used by BEAT Engine",
    )
    export_speaker.add_argument("--julia-threads", default=None, help="Julia thread count, or auto")

    speaker_preflight = commands.add_parser(
        "speaker-preflight",
        help="Estimate Level-3 package storage and per-frequency interior working sets",
    )
    speaker_preflight.add_argument("project_file", type=Path, help="Path to the .blab.json project")
    speaker_preflight.add_argument("--request", type=Path, help="Optional headless solve-request JSON overlay")
    speaker_preflight.add_argument("--rom-rank", type=int, default=32, help="Parity ROM rank per sector")
    speaker_preflight.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default="float32",
        help="Complex matrix precision used for storage estimates",
    )
    speaker_preflight.add_argument("--json", action="store_true", help="Print the complete estimate as JSON")
    return parser


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> None:
    parser = _build_arg_parser(prog)
    args = parser.parse_args(argv)
    try:
        if args.project_command == "validate":
            _validate(args)
        elif args.project_command == "solve":
            _solve(args)
        elif args.project_command == "evaluate-fem":
            _evaluate_fem(args)
        elif args.project_command == "compare-fem":
            _compare_fem(args)
        elif args.project_command == "speaker-preflight":
            _speaker_preflight(args)
        else:
            _export_speaker(args)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        if getattr(args, "json", False) or getattr(args, "events", None) == "ndjson":
            print(
                json.dumps(
                    {
                        "event": "error",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        else:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _validate(args: argparse.Namespace) -> None:
    project = load_headless_project(args.project_file)
    spec = load_headless_solve_spec(args.request)
    backend_id = resolve_headless_backend(args.backend, julia_executable=args.julia_executable)
    prepared = prepare_headless_solve(project, spec, backend_id=backend_id)
    summary = validation_summary(project, prepared, backend_id=backend_id)
    summary["backend_requested"] = args.backend
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(
        f"Valid {summary['solve_kind']} system '{summary['system_name']}' using {summary['backend_id']}: "
        f"{summary['frequency_count']} frequencies, {len(summary['excitation_port_ids'])} excitations, "
        f"{len(summary['meshes'])} meshes."
    )
    for assumption in summary["assumptions"]:
        print(f"[{assumption['status']}] {assumption['statement']}")


def _solve(args: argparse.Namespace) -> None:
    project = load_headless_project(args.project_file)
    spec = load_headless_solve_spec(args.request)
    backend_id = resolve_headless_backend(args.backend, julia_executable=args.julia_executable)
    prepared = prepare_headless_solve(project, spec, backend_id=backend_id)
    output = args.output or default_result_path(project.path)

    def emit(event: dict[str, Any]) -> None:
        if args.events == "ndjson":
            print(json.dumps(event, separators=(",", ":")), flush=True)
            return
        event_name = event.get("event")
        if event_name == "frequency_completed":
            print(
                f"Solved {event['solved_count']}/{event['frequency_count']}: {event['freq_hz']:.6g} Hz",
                file=sys.stderr,
                flush=True,
            )
        elif event_name in {"status", "failed", "interrupted"}:
            print(str(event.get("message", event_name)), file=sys.stderr, flush=True)

    summary = run_headless_solve(
        project,
        prepared,
        output_dir=output,
        backend_id=backend_id,
        public_request=spec.raw or {"schema_version": 1},
        julia_executable=args.julia_executable,
        julia_threads=args.julia_threads,
        event_callback=emit,
    )
    if args.events == "text":
        print(json.dumps(summary, indent=2, sort_keys=True))


def _evaluate_fem(args: argparse.Namespace) -> None:
    gates = FEMValidationGates(
        maximum_within_surface_phase_rms_deg=args.max_within_phase_rms_deg,
        maximum_inter_surface_phase_rms_deg=args.max_inter_phase_rms_deg,
        maximum_inter_surface_phase_deg=args.max_inter_phase_deg,
        minimum_plane_mode_fraction=args.min_plane_mode_fraction,
        minimum_points_per_wavelength_p95=args.min_points_per_wavelength_p95,
        minimum_points_per_wavelength_maximum_edge=(args.min_points_per_wavelength_maximum_edge),
    )
    report = evaluate_fem_run(
        args.run_dir,
        surface_patterns=tuple(args.surface_patterns or ("exit_*",)),
        split_surface_entities=args.split_surface_entities,
        gates=gates,
    )
    output = args.output or args.run_dir / "fem-validation.json"
    write_fem_validation_report(output, report)
    summary = {
        "output": str(output.resolve()),
        "surface_count": report["surface_count"],
        "frequencies_hz": report["frequencies_hz"],
        "sampled_coherence_ceiling_hz_by_excitation": report["sampled_coherence_ceiling_hz_by_excitation"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _compare_fem(args: argparse.Namespace) -> None:
    gates = FEMConvergenceGates(
        maximum_surface_phase_rms_delta_deg=args.max_phase_rms_delta_deg,
        maximum_surface_phase_delta_deg=args.max_phase_delta_deg,
        maximum_normalized_amplitude_rms_delta=(args.max_normalized_amplitude_rms_delta),
        maximum_plane_mode_fraction_delta=args.max_plane_mode_fraction_delta,
    )
    report = compare_fem_validation_reports(
        args.coarse_report,
        args.fine_report,
        gates=gates,
    )
    output = args.output or args.fine_report.with_name("fem-convergence.json")
    write_fem_validation_report(output, report)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "passed": report["passed"],
                "frequency_count": len(report["comparisons"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _speaker_preflight(args: argparse.Namespace) -> None:
    project = load_headless_project(args.project_file)
    spec = load_headless_solve_spec(args.request)
    frequency_count = len(spec.frequencies_hz) if spec.frequencies_hz is not None else int(project.preferences.freq_count)
    sphere_angle_deg = min(max(float(project.preferences.balloon_angle_precision_deg), 0.5), 15.0)
    sphere_point_count = max(int(round(41253.0 / sphere_angle_deg**2)), 1)
    estimate = estimate_level_three_package(
        project.physical_system,
        symmetry=project.symmetry,
        frequency_count=frequency_count,
        complex_bytes=8 if args.precision == "float32" else 16,
        rom_rank=args.rom_rank,
        sphere_point_count=sphere_point_count,
    )
    payload = estimate.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    sizes = payload["sizes"]
    print(
        f"Level-3 preflight for '{project.physical_system.name}': "
        f"{estimate.frequency_count} frequencies, {estimate.state_count} retained states, "
        f"{estimate.bem_node_count} BEM nodes / {estimate.bem_face_count} faces."
    )
    print(
        f"Rank-{estimate.rom_rank}-per-sector parity ROM package estimate: "
        f"{sizes['parity_rom_package_estimate']['mib']:.1f} MiB"
    )
    print(
        "ROM-training current-frequency Schur working set: "
        f"{sizes['rom_training_schur']['mib']:.1f} MiB"
    )


def _export_speaker(args: argparse.Namespace) -> None:
    project = load_headless_project(args.project_file)
    spec = load_headless_solve_spec(args.request)
    sphere_angle_deg = min(max(float(project.preferences.balloon_angle_precision_deg), 0.5), 15.0)
    sphere_point_count = max(int(round(41253.0 / sphere_angle_deg**2)), 1)
    fidelity = SpeakerPackageFidelity.parse(args.fidelity)
    coupled_representation = SpeakerPackageCoupledRepresentation.PARITY_ROM
    backend_id = resolve_headless_backend(args.backend, julia_executable=args.julia_executable)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        if fidelity >= SpeakerPackageFidelity.COUPLED and project.symmetry != "off":
            temporary = tempfile.TemporaryDirectory(prefix="blab-speaker-full-")
            expanded = expand_speaker_system_for_export(
                project.physical_system,
                symmetry=project.symmetry,
                output_dir=temporary.name,
                preferred_full_mesh_by_name=preferred_full_meshes_from_project_payload(project.payload),
            )
            channel_by_component = {
                component_id: project.component_channel_by_id.get(source_id, "main")
                for component_id, source_id in expanded.component_source_ids.items()
            }
            project = replace(
                project,
                physical_system=expanded.system,
                symmetry="off",
                component_channel_by_id=channel_by_component,
            )
        prepared = prepare_headless_solve(project, spec, backend_id=backend_id)
        prepared = prepare_speaker_package_solve(
            prepared,
            fidelity=fidelity,
            coupled_representation=coupled_representation,
            sphere_point_count=sphere_point_count,
            sphere_radius_m=project.preferences.polar_observation_distance_m,
            speaker_rom_rank=args.speaker_rom_rank,
            speaker_rom_training_count=args.speaker_rom_training_count,
            speaker_rom_validation_count=args.speaker_rom_validation_count,
        )

        def emit(event: dict[str, Any]) -> None:
            if args.events == "ndjson":
                print(json.dumps(event, separators=(",", ":")), flush=True)
            elif event.get("event") == "frequency_completed":
                print(
                    f"Solved {event['solved_count']}/{event['frequency_count']}: {event['freq_hz']:.6g} Hz",
                    file=sys.stderr,
                    flush=True,
                )

        solved = solve_speaker_package_system(
            prepared,
            event_callback=emit,
            julia_executable=args.julia_executable,
            julia_threads=args.julia_threads,
        )
        result = export_speaker_package(
            solved,
            SpeakerPackageConfig(
                output_path=args.output,
                name=args.name or project.physical_system.name,
                fidelity=fidelity,
                coupled_representation=coupled_representation,
            ),
        )
    finally:
        if temporary is not None:
            temporary.cleanup()
    summary = {
        "event": "package_completed",
        "output": str(result.path),
        "fidelity": result.fidelity.cli_name,
        "frequency_count": result.frequency_count,
        "excitation_count": result.excitation_count,
    }
    if args.events == "ndjson":
        print(json.dumps(summary, separators=(",", ":")), flush=True)
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
