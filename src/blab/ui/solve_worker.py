"""Qt worker wrapper for streaming live BEM solves."""

from __future__ import annotations

import multiprocessing as mp
import time
from queue import Empty

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from blab.config import SimulationConfig
from blab.live import (
    LiveSolver,
    solve_frequency_worker_process,
    split_frequency_order_for_workers,
)


class SolveWorker(QObject):
    initialized = Signal(object, object, object)
    result_ready = Signal(object)
    status = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, config: SimulationConfig, frequencies: np.ndarray, worker_count: int = 1):
        super().__init__()
        self.config = config
        self.frequencies = frequencies
        self.worker_count = worker_count
        self._stop = False
        self._stop_event = None

    @Slot()
    def run(self) -> None:
        try:
            if self.worker_count > 1:
                self._run_process_workers()
                return

            t_start = time.perf_counter()
            live_solver = LiveSolver(self.config)
            self.status.emit(f"Worker 1 initialized in {time.perf_counter() - t_start:.1f}s")
            self.initialized.emit(live_solver.polar_angle_deg, live_solver.radiator_names, live_solver.sphere_metadata)
            for result in live_solver.solve_stream(
                self.frequencies,
                stop_requested=lambda: self._stop,
            ):
                self.result_ready.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self._stop = True
        if self._stop_event is not None:
            self._stop_event.set()

    def _run_process_workers(self) -> None:
        ctx = mp.get_context("spawn")
        chunks = split_frequency_order_for_workers(self.frequencies, self.worker_count)
        self._stop_event = ctx.Event()
        output_queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=solve_frequency_worker_process,
                args=(self.config, chunk, self._stop_event, output_queue, worker_id),
            )
            for worker_id, chunk in enumerate(chunks)
        ]

        for process in processes:
            process.start()

        initialized = False
        initialized_workers = 0
        completed_workers = 0
        try:
            while completed_workers < len(processes):
                if self._stop:
                    self._stop_event.set()

                try:
                    message, worker_id, payload = output_queue.get(timeout=0.1)
                except Empty:
                    continue

                if message == "initialized":
                    initialized_workers += 1
                    angles, radiator_names, sphere_metadata, elapsed_s = payload
                    self.status.emit(
                        f"Worker {worker_id + 1}/{len(processes)} initialized "
                        f"in {elapsed_s:.1f}s ({initialized_workers}/{len(processes)} ready)"
                    )
                    if not initialized:
                        self.initialized.emit(angles, radiator_names, sphere_metadata)
                        initialized = True
                elif message == "result":
                    self.result_ready.emit(payload)
                elif message == "error":
                    self.failed.emit(f"Worker {worker_id + 1}: {payload}")
                    self._stop_event.set()
                elif message == "done":
                    completed_workers += 1
        finally:
            self._stop_event.set()
            for process in processes:
                process.join(timeout=2.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
