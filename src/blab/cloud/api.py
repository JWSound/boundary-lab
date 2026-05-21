"""Local cloud-solver prototype API.

This module intentionally starts as a single-process service. The HTTP/WebSocket
contract is the part to keep stable while the backing runner moves to SQS,
Fargate tasks, and object storage.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
try:
    from fastapi import WebSocket
except ImportError:  # pragma: no cover - create_app reports the missing optional extra
    WebSocket = Any  # type: ignore[misc, assignment]

from blab.cloud.bundle import load_solve_bundle, load_solve_bundle_bytes
from blab.cloud.events import DynamoDbEventStore, EventStore, LocalEventStore
from blab.cloud.launch import EcsFargateTaskLauncher
from blab.cloud.protocol import (
    config_from_payload,
    failed_event,
    status_event,
)
from blab.cloud.storage import BundleStore, LocalBundleStore, S3BundleStore
from blab.cloud.worker import CallableEventSink, run_solve_job
from blab.config import SimulationConfig
from blab.live import order_frequencies_for_live_plotting


TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}
TERMINAL_EVENT_TYPES = {"completed", "failed"}


@dataclass
class SolveJob:
    job_id: str
    config: SimulationConfig | None = None
    frequencies: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float32))
    worker_count: int = 1
    created_at: float = field(default_factory=time.time)
    status: str = "created"
    bundle_key: str | None = None
    launch: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    stop_requested: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, SolveJob] = {}
        self._lock = threading.Lock()

    def create_pending(self, *, bundle_key: str | None = None) -> SolveJob:
        job = SolveJob(
            job_id=f"job_{uuid.uuid4().hex}",
            bundle_key=bundle_key,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def create(self, config: SimulationConfig, frequencies: np.ndarray, worker_count: int = 1) -> SolveJob:
        job = SolveJob(
            job_id=f"job_{uuid.uuid4().hex}",
            config=config,
            frequencies=frequencies,
            worker_count=worker_count,
            status="queued",
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> SolveJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job: SolveJob) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "status": job.status,
            "created_at": job.created_at,
            "bundle_key": job.bundle_key,
            "launch": job.launch,
            "frequency_count": int(job.frequencies.size),
            "solved_count": sum(1 for event in job.events if event.get("type") == "frequency_result"),
        }

    def mark_uploaded(self, job: SolveJob) -> None:
        job.status = "uploaded"

    def prepare_to_start(self, job: SolveJob, config: SimulationConfig, frequencies: np.ndarray) -> None:
        job.config = config
        job.frequencies = frequencies
        job.worker_count = config.workers
        job.status = "queued"

    def append_event(self, loop: asyncio.AbstractEventLoop, job: SolveJob, event: dict[str, Any]) -> None:
        job.events.append(event)
        self._apply_status_from_event(job, event)

        for queue in list(job.subscribers):
            loop.call_soon_threadsafe(queue.put_nowait, event)

    def _apply_status_from_event(self, job: SolveJob, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type in {"completed", "failed"}:
            job.status = str(event_type)
        elif event_type == "initialized":
            job.status = "running"

    async def subscribe(self, job: SolveJob):
        queue: asyncio.Queue = asyncio.Queue()
        job.subscribers.append(queue)
        try:
            for event in job.events:
                yield event
            while job.status not in {"completed", "failed", "cancelled"}:
                yield await queue.get()
        finally:
            if queue in job.subscribers:
                job.subscribers.remove(queue)


def create_app(
    bundle_store: BundleStore | None = None,
    task_launcher: EcsFargateTaskLauncher | None = None,
    event_store: EventStore | None = None,
):
    try:
        from fastapi import Body, FastAPI, HTTPException, WebSocketDisconnect
    except ImportError as exc:  # pragma: no cover - exercised by manual launch
        raise SystemExit('Install cloud extras first: python -m pip install -e ".[cloud]"') from exc

    app = FastAPI(title="Boundary Lab Cloud Solver Prototype")
    store = InMemoryJobStore()
    bundle_store = bundle_store or _bundle_store_from_env()
    task_launcher = task_launcher if task_launcher is not None else _task_launcher_from_env()
    event_store = event_store if event_store is not None else _event_store_from_env()

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.post("/v1/solve-jobs")
    async def create_solve_job(payload: dict[str, Any]):
        config = config_from_payload(payload["config"])
        frequencies = np.asarray(payload.get("frequencies") or [], dtype=np.float32)
        if frequencies.size == 0:
            frequencies = np.logspace(
                np.log10(config.freq_min),
                np.log10(config.freq_max),
                config.freq_count,
            ).astype(np.float32)
        if payload.get("live_order", True):
            frequencies = order_frequencies_for_live_plotting(frequencies)

        worker_count = int(payload.get("worker_count", 1))
        job = store.create(config, frequencies, worker_count=worker_count)
        loop = asyncio.get_running_loop()
        _start_job_thread(loop, store, job, event_store)
        return {
            **store.snapshot(job),
            "stream_path": f"/v1/solve-jobs/{job.job_id}/stream",
        }

    @app.post("/v1/solve-jobs/upload-target")
    async def create_solve_job_upload_target():
        job = store.create_pending()
        job.bundle_key = bundle_store.bundle_key(job.job_id)
        target = bundle_store.create_upload_target(job.job_id)
        if isinstance(bundle_store, LocalBundleStore):
            target_payload = {
                "url": f"/v1/solve-jobs/{job.job_id}/bundle",
                "method": "PUT",
                "headers": {"Content-Type": "application/zip"},
                "key": target.key,
            }
        else:
            target_payload = {
                "url": target.url,
                "method": target.method,
                "headers": target.headers,
                "key": target.key,
            }
        return {
            **store.snapshot(job),
            "upload": target_payload,
            "start_path": f"/v1/solve-jobs/{job.job_id}/start",
            "stream_path": f"/v1/solve-jobs/{job.job_id}/stream",
        }

    @app.put("/v1/solve-jobs/{job_id}/bundle")
    async def upload_solve_job_bundle(job_id: str, bundle_bytes: bytes = Body(..., media_type="application/zip")):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown solve job.")
        if not isinstance(bundle_store, LocalBundleStore):
            raise HTTPException(status_code=400, detail="Upload bundles directly to the returned presigned URL.")
        if not bundle_bytes:
            raise HTTPException(status_code=400, detail="Solve bundle body is empty.")
        bundle_store.put_bytes(job.job_id, bundle_bytes)
        job.bundle_key = bundle_store.bundle_key(job.job_id)
        store.mark_uploaded(job)
        return store.snapshot(job)

    @app.post("/v1/solve-jobs/{job_id}/start")
    async def start_uploaded_solve_job(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown solve job.")
        if job.status not in {"created", "uploaded"}:
            raise HTTPException(status_code=409, detail=f"Job cannot be started from status {job.status}.")

        workspace = tempfile.mkdtemp(prefix="blab_solve_")
        bundle_path = Path(workspace) / "solve.blabsolve.zip"
        try:
            bundle_store.download_bundle(job.job_id, bundle_path)
            config, frequencies = load_solve_bundle(bundle_path, Path(workspace) / "extracted")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        store.prepare_to_start(job, config, frequencies)
        loop = asyncio.get_running_loop()
        _launch_started_job(loop, store, job, bundle_store, task_launcher, event_store)
        return {
            **store.snapshot(job),
            "stream_path": f"/v1/solve-jobs/{job.job_id}/stream",
        }

    @app.post("/v1/solve-jobs/bundle")
    async def create_solve_job_from_bundle(bundle_bytes: bytes = Body(..., media_type="application/zip")):
        if not bundle_bytes:
            raise HTTPException(status_code=400, detail="Solve bundle body is empty.")

        workspace = tempfile.mkdtemp(prefix="blab_solve_")
        try:
            config, frequencies = load_solve_bundle_bytes(bundle_bytes, workspace)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        job = store.create(config, frequencies, worker_count=config.workers)
        loop = asyncio.get_running_loop()
        _start_job_thread(loop, store, job, event_store)
        return {
            **store.snapshot(job),
            "stream_path": f"/v1/solve-jobs/{job.job_id}/stream",
        }

    @app.get("/v1/solve-jobs/{job_id}")
    async def get_solve_job(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown solve job.")
        return store.snapshot(job)

    @app.post("/v1/solve-jobs/{job_id}/cancel")
    async def cancel_solve_job(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown solve job.")
        job.stop_requested.set()
        store.append_event(asyncio.get_running_loop(), job, status_event(job.job_id, "Cancellation requested"))
        job.status = "cancelled"
        return store.snapshot(job)

    @app.websocket("/v1/solve-jobs/{job_id}/stream")
    async def stream_solve_job(websocket: WebSocket, job_id: str):
        await websocket.accept()
        job = store.get(job_id)
        if job is None:
            await websocket.send_json(failed_event(job_id, "Unknown solve job."))
            await websocket.close(code=4404)
            return
        try:
            async for event in _subscribe_job_events(store, job, event_store):
                await websocket.send_json(event)
        except WebSocketDisconnect:
            return

    return app


def _start_job_thread(
    loop: asyncio.AbstractEventLoop,
    store: InMemoryJobStore,
    job: SolveJob,
    event_store: EventStore | None = None,
) -> None:
    def run_job() -> None:
        def emit(event: dict) -> None:
            if event_store is not None:
                event_store.append(job.job_id, event)
            store.append_event(loop, job, event)

        sink = CallableEventSink(emit)
        if job.config is None:
            store.append_event(loop, job, failed_event(job.job_id, "Solve job has no loaded config."))
            return
        run_solve_job(
            job_id=job.job_id,
            config=job.config,
            frequencies=job.frequencies,
            sink=sink,
            stop_requested=job.stop_requested.is_set,
        )
        if job.stop_requested.is_set():
            job.status = "cancelled"

    job.thread = threading.Thread(target=run_job, name=f"solve-{job.job_id}", daemon=True)
    job.thread.start()


def _launch_started_job(
    loop: asyncio.AbstractEventLoop,
    store: InMemoryJobStore,
    job: SolveJob,
    bundle_store: BundleStore,
    task_launcher: EcsFargateTaskLauncher | None,
    event_store: EventStore | None,
) -> None:
    if task_launcher is None:
        _start_job_thread(loop, store, job, event_store)
        return
    if not isinstance(bundle_store, S3BundleStore):
        store.append_event(loop, job, failed_event(job.job_id, "ECS launch requires BLAB_BUNDLE_STORE=s3."))
        return
    if job.bundle_key is None:
        store.append_event(loop, job, failed_event(job.job_id, "Solve job has no bundle key."))
        return

    launch = task_launcher.launch_worker(
        job_id=job.job_id,
        s3_bucket=bundle_store.bucket,
        s3_key=job.bundle_key,
    )
    job.status = "submitted"
    job.launch = {
        "type": "ecs",
        "task_arn": launch.task_arn,
        "cluster": launch.cluster,
        "task_definition": launch.task_definition,
    }
    store.append_event(
        loop,
        job,
        status_event(job.job_id, f"ECS task submitted: {launch.task_arn}. Waiting for worker events..."),
    )


async def _subscribe_job_events(
    store: InMemoryJobStore,
    job: SolveJob,
    event_store: EventStore | None,
):
    durable_seq = 0
    seen_memory_count = 0
    while True:
        while seen_memory_count < len(job.events):
            event = job.events[seen_memory_count]
            seen_memory_count += 1
            yield event

        if event_store is not None and job.launch is not None:
            records = event_store.list_after(job.job_id, durable_seq)
            for record in records:
                durable_seq = max(durable_seq, record.seq)
                event = record.event
                store._apply_status_from_event(job, event)
                yield event

        if job.status in TERMINAL_JOB_STATUSES:
            break
        await asyncio.sleep(0.25)


def _bundle_store_from_env() -> BundleStore:
    store_type = os.environ.get("BLAB_BUNDLE_STORE", "local").strip().lower()
    if store_type == "s3":
        bucket = os.environ.get("BLAB_S3_BUCKET", "").strip()
        if not bucket:
            raise SystemExit("BLAB_S3_BUCKET is required when BLAB_BUNDLE_STORE=s3.")
        return S3BundleStore(bucket, prefix=os.environ.get("BLAB_S3_PREFIX", ""))

    root = os.environ.get("BLAB_LOCAL_BUNDLE_ROOT")
    if root is None:
        root = str(Path(tempfile.mkdtemp(prefix="blab_bundles_")))
    return LocalBundleStore(root)


def _task_launcher_from_env() -> EcsFargateTaskLauncher | None:
    launcher = os.environ.get("BLAB_JOB_LAUNCHER", "local").strip().lower()
    if launcher in {"", "local", "thread"}:
        return None
    if launcher == "ecs":
        return EcsFargateTaskLauncher.from_env()
    raise SystemExit(f"Unsupported BLAB_JOB_LAUNCHER: {launcher}")


def _event_store_from_env() -> EventStore | None:
    store_type = os.environ.get("BLAB_EVENT_STORE", "none").strip().lower()
    if store_type in {"", "none"}:
        return None
    if store_type == "local":
        root = os.environ.get("BLAB_LOCAL_EVENT_ROOT", "runs/cloud/events")
        return LocalEventStore(root)
    if store_type == "dynamodb":
        table = os.environ.get("BLAB_DYNAMODB_EVENTS_TABLE", "").strip()
        if not table:
            raise SystemExit("BLAB_DYNAMODB_EVENTS_TABLE is required when BLAB_EVENT_STORE=dynamodb.")
        return DynamoDbEventStore(table)
    raise SystemExit(f"Unsupported BLAB_EVENT_STORE: {store_type}")


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description="Run the Boundary Lab cloud solver prototype API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised by manual launch
        raise SystemExit('Install cloud extras first: python -m pip install -e ".[cloud]"') from exc

    uvicorn.run("blab.cloud.api:create_app", factory=True, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
