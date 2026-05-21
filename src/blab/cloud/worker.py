"""Reusable cloud solve worker runner."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from blab.cloud.bundle import load_solve_bundle
from blab.cloud.events import DynamoDbEventStore, EventStoreSink, LocalEventStore
from blab.cloud.protocol import completed_event, failed_event, frequency_result_event, initialized_event, status_event
from blab.cloud.storage import S3BundleStore
from blab.config import SimulationConfig
from blab.live import LiveSolver


class EventSink(Protocol):
    def emit(self, event: dict) -> None:
        """Persist or forward one solve event."""


class CallableEventSink:
    def __init__(self, callback: Callable[[dict], None]):
        self._callback = callback

    def emit(self, event: dict) -> None:
        self._callback(event)


class JsonlEventSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: dict) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, separators=(",", ":")))
            handle.write("\n")


class StdoutEventSink:
    def emit(self, event: dict) -> None:
        print(json.dumps(event, separators=(",", ":")), flush=True)


class CompositeEventSink:
    def __init__(self, sinks: list[EventSink]):
        self.sinks = sinks

    def emit(self, event: dict) -> None:
        for sink in self.sinks:
            sink.emit(event)


def run_solve_job(
    *,
    job_id: str,
    config: SimulationConfig,
    frequencies: np.ndarray,
    sink: EventSink,
    stop_requested: Callable[[], bool] | None = None,
) -> None:
    """Run a solve and emit the same events used by the WebSocket API."""
    stop_requested = stop_requested or (lambda: False)
    try:
        start = time.perf_counter()
        live_solver = LiveSolver(config)
        sink.emit(status_event(job_id, f"Worker initialized in {time.perf_counter() - start:.1f}s"))
        sink.emit(
            initialized_event(
                job_id=job_id,
                polar_angle_deg=live_solver.polar_angle_deg,
                radiator_names=live_solver.radiator_names,
                sphere_metadata=live_solver.sphere_metadata,
            )
        )
        for result in live_solver.solve_stream(frequencies, stop_requested=stop_requested):
            if stop_requested():
                break
            sink.emit(frequency_result_event(job_id, result))
        if stop_requested():
            sink.emit(status_event(job_id, "Cancelled"))
        else:
            sink.emit(completed_event(job_id))
    except Exception as exc:  # pragma: no cover - solver failures are environment dependent
        sink.emit(failed_event(job_id, str(exc)))


def _build_arg_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Run a Boundary Lab cloud solve worker.")
    parser.add_argument("--job-id", required=True, help="Solve job identifier.")
    parser.add_argument("--bundle", type=Path, default=None, help="Path to a .blabsolve.zip bundle.")
    parser.add_argument("--s3-bucket", default=None, help="S3 bucket containing the solve bundle.")
    parser.add_argument("--s3-key", default=None, help="S3 object key for the solve bundle.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Directory for extracting the bundle. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--events-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file for emitted events. Defaults to stdout.",
    )
    parser.add_argument(
        "--event-store",
        choices=("none", "local", "dynamodb"),
        default=None,
        help="Optional durable event store. Defaults to BLAB_EVENT_STORE or none.",
    )
    parser.add_argument("--event-store-root", type=Path, default=None, help="Root directory for local event store.")
    parser.add_argument("--dynamodb-events-table", default=None, help="DynamoDB table name for durable events.")
    return parser


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    args = _build_arg_parser(prog).parse_args(argv)
    workspace = args.workspace or Path(tempfile.mkdtemp(prefix="blab_cloud_worker_"))
    bundle_path = _resolve_bundle_path(args, workspace)
    config, frequencies = load_solve_bundle(bundle_path, workspace)
    sink = _build_event_sink(args)
    run_solve_job(job_id=args.job_id, config=config, frequencies=frequencies, sink=sink)


def _resolve_bundle_path(args: argparse.Namespace, workspace: Path) -> Path:
    if args.bundle is not None:
        return args.bundle
    if args.s3_bucket and args.s3_key:
        return S3BundleStore(args.s3_bucket).download_key(args.s3_key, workspace / "solve.blabsolve.zip")
    raise SystemExit("Provide --bundle or both --s3-bucket and --s3-key.")


def _build_event_sink(args: argparse.Namespace) -> EventSink:
    sinks: list[EventSink] = [JsonlEventSink(args.events_jsonl) if args.events_jsonl else StdoutEventSink()]
    store_type = (args.event_store or os.environ.get("BLAB_EVENT_STORE", "none")).strip().lower()
    if store_type in {"", "none"}:
        return sinks[0]
    if store_type == "local":
        root = args.event_store_root or Path(os.environ.get("BLAB_LOCAL_EVENT_ROOT", "runs/cloud/events"))
        sinks.append(EventStoreSink(LocalEventStore(root), args.job_id))
    elif store_type == "dynamodb":
        table = args.dynamodb_events_table or os.environ.get("BLAB_DYNAMODB_EVENTS_TABLE", "")
        if not table:
            raise SystemExit("BLAB_DYNAMODB_EVENTS_TABLE or --dynamodb-events-table is required.")
        sinks.append(EventStoreSink(DynamoDbEventStore(table), args.job_id))
    else:
        raise SystemExit(f"Unsupported event store: {store_type}")
    return CompositeEventSink(sinks)


if __name__ == "__main__":
    main(sys.argv[1:])
