"""Asynchronous, coalescing execution for exterior BEM field requests."""

from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Signal

from blab.solvers.coupled_field import BemFieldEvaluationRequest, evaluate_bem_field


@dataclass(frozen=True)
class ExteriorFieldTask:
    key: tuple[object, ...]
    backend_id: str
    request: BemFieldEvaluationRequest


class ExteriorFieldEvaluationService(QObject):
    """Run one field evaluation at a time and retain only the latest queued task."""

    completed = Signal(object, object)
    failed = Signal(object, str)
    discarded = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="blab-bem-field")
        self._lock = threading.Lock()
        self._running_key: tuple[object, ...] | None = None
        self._queued: ExteriorFieldTask | None = None
        self._closed = False
        self.destroyed.connect(self.close)

    def submit(self, task: ExteriorFieldTask) -> None:
        with self._lock:
            if self._closed:
                self.discarded.emit(task.key)
                return
            if task.key == self._running_key or (self._queued is not None and task.key == self._queued.key):
                return
            if self._running_key is not None:
                discarded = self._queued
                self._queued = task
                if discarded is not None:
                    self.discarded.emit(discarded.key)
                return
            self._start_locked(task)

    def close(self, *_args: object) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            queued = self._queued
            self._queued = None
        if queued is not None:
            self.discarded.emit(queued.key)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _start_locked(self, task: ExteriorFieldTask) -> None:
        self._running_key = task.key
        future = self._executor.submit(
            evaluate_bem_field,
            task.request,
            backend_id=task.backend_id,
            julia_executable=os.environ.get("BLAB_JULIA_EXE", "julia"),
        )
        future.add_done_callback(lambda completed, task=task: self._finished(task, completed))

    def _finished(self, task: ExteriorFieldTask, future: Future[np.ndarray]) -> None:
        try:
            values = future.result()
        except Exception as exc:
            self.failed.emit(task.key, str(exc))
        else:
            self.completed.emit(task.key, values)
        with self._lock:
            self._running_key = None
            if self._closed:
                return
            queued = self._queued
            self._queued = None
            if queued is not None:
                self._start_locked(queued)


__all__ = ["ExteriorFieldEvaluationService", "ExteriorFieldTask"]
