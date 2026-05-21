"""Local cloud-solver prototype API.

This module intentionally starts as a single-process service. The HTTP/WebSocket
contract is the part to keep stable while the backing runner moves to SQS,
Fargate tasks, and object storage.
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
try:
    from fastapi import WebSocket
except ImportError:  # pragma: no cover - create_app reports the missing optional extra
    WebSocket = Any  # type: ignore[misc, assignment]

from blab.cloud.bundle import load_solve_bundle_bytes
from blab.cloud.protocol import (
    completed_event,
    config_from_payload,
    failed_event,
    frequency_result_event,
    initialized_event,
    status_event,
)
from blab.config import SimulationConfig
from blab.live import LiveSolver, order_frequencies_for_live_plotting


@dataclass
class SolveJob:
    job_id: str
    config: SimulationConfig
    frequencies: np.ndarray
    worker_count: int = 1
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    stop_requested: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, SolveJob] = {}
        self._lock = threading.Lock()

    def create(self, config: SimulationConfig, frequencies: np.ndarray, worker_count: int = 1) -> SolveJob:
        job = SolveJob(
            job_id=f"job_{uuid.uuid4().hex}",
            config=config,
            frequencies=frequencies,
            worker_count=worker_count,
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
            "frequency_count": int(job.frequencies.size),
            "solved_count": sum(1 for event in job.events if event.get("type") == "frequency_result"),
        }

    def append_event(self, loop: asyncio.AbstractEventLoop, job: SolveJob, event: dict[str, Any]) -> None:
        job.events.append(event)
        if event["type"] in {"completed", "failed"}:
            job.status = event["type"]
        elif event["type"] == "initialized":
            job.status = "running"

        for queue in list(job.subscribers):
            loop.call_soon_threadsafe(queue.put_nowait, event)

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


def create_app():
    try:
        from fastapi import Body, FastAPI, HTTPException, WebSocketDisconnect
    except ImportError as exc:  # pragma: no cover - exercised by manual launch
        raise SystemExit('Install cloud extras first: python -m pip install -e ".[cloud]"') from exc

    app = FastAPI(title="Boundary Lab Cloud Solver Prototype")
    store = InMemoryJobStore()

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
        _start_job_thread(loop, store, job)
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
        _start_job_thread(loop, store, job)
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
            async for event in store.subscribe(job):
                await websocket.send_json(event)
        except WebSocketDisconnect:
            return

    return app


def _start_job_thread(loop: asyncio.AbstractEventLoop, store: InMemoryJobStore, job: SolveJob) -> None:
    def run_job() -> None:
        try:
            start = time.perf_counter()
            live_solver = LiveSolver(job.config)
            store.append_event(
                loop,
                job,
                status_event(job.job_id, f"Worker initialized in {time.perf_counter() - start:.1f}s"),
            )
            store.append_event(
                loop,
                job,
                initialized_event(
                    job_id=job.job_id,
                    polar_angle_deg=live_solver.polar_angle_deg,
                    radiator_names=live_solver.radiator_names,
                    sphere_metadata=live_solver.sphere_metadata,
                ),
            )
            for result in live_solver.solve_stream(
                job.frequencies,
                stop_requested=job.stop_requested.is_set,
            ):
                if job.stop_requested.is_set():
                    break
                store.append_event(loop, job, frequency_result_event(job.job_id, result))
            if job.stop_requested.is_set():
                store.append_event(loop, job, status_event(job.job_id, "Cancelled"))
                job.status = "cancelled"
            else:
                store.append_event(loop, job, completed_event(job.job_id))
        except Exception as exc:  # pragma: no cover - solver failures are environment dependent
            store.append_event(loop, job, failed_event(job.job_id, str(exc)))

    job.thread = threading.Thread(target=run_job, name=f"solve-{job.job_id}", daemon=True)
    job.thread.start()


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
