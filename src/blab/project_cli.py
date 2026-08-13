"""Command-line entry points for validating and solving Boundary Lab projects."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
    return parser


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> None:
    parser = _build_arg_parser(prog)
    args = parser.parse_args(argv)
    try:
        if args.project_command == "validate":
            _validate(args)
        else:
            _solve(args)
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


if __name__ == "__main__":
    main()
