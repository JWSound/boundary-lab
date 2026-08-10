"""Asynchronous, coalescing execution for exterior BEM field requests."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
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
    mask_distance_m: float = 0.0


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
        self._boundary_surfaces: OrderedDict[str, object] = OrderedDict()
        self._boundary_masks: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()
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
            self._evaluate_task,
            task,
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

    def _evaluate_task(self, task: ExteriorFieldTask) -> np.ndarray:
        values = evaluate_bem_field(
            task.request,
            backend_id=task.backend_id,
            julia_executable=os.environ.get("BLAB_JULIA_EXE", "julia"),
        )
        if task.mask_distance_m <= 0.0 or not values.size:
            return values
        mask = self._boundary_mask(task)
        if not np.any(mask):
            return values
        masked = values.copy()
        masked[mask] = np.nan + 1j * np.nan
        return masked

    def _boundary_mask(self, task: ExteriorFieldTask) -> np.ndarray:
        mask_key = (
            task.request.domain_key,
            *task.key[:8],
            float(task.mask_distance_m),
            int(task.request.points_m.shape[0]),
        )
        cached = self._boundary_masks.get(mask_key)
        if cached is not None:
            self._boundary_masks.move_to_end(mask_key)
            return cached
        surface = self._boundary_surface(task.request)
        pv = __import__("pyvista")
        samples = pv.PolyData(np.asarray(task.request.points_m, dtype=float))
        distances = np.abs(np.asarray(samples.compute_implicit_distance(surface)["implicit_distance"]))
        mask = distances <= float(task.mask_distance_m)
        self._boundary_masks[mask_key] = mask
        while len(self._boundary_masks) > 8:
            self._boundary_masks.popitem(last=False)
        return mask

    def _boundary_surface(self, request: BemFieldEvaluationRequest):
        cached = self._boundary_surfaces.get(request.domain_key)
        if cached is not None:
            self._boundary_surfaces.move_to_end(request.domain_key)
            return cached
        pv = __import__("pyvista")
        points, triangles = _expanded_boundary_geometry(
            request.boundary_points_m,
            request.boundary_triangles,
            request.symmetry,
        )
        faces = np.column_stack((np.full(triangles.shape[0], 3, dtype=np.int64), triangles)).reshape(-1)
        surface = pv.PolyData(points, faces)
        self._boundary_surfaces[request.domain_key] = surface
        while len(self._boundary_surfaces) > 2:
            self._boundary_surfaces.popitem(last=False)
        return surface


def _expanded_boundary_geometry(
    points: np.ndarray,
    triangles: np.ndarray,
    symmetry: str,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    signs = [(1.0, 1.0, 1.0)]
    if symmetry in {"x", "xy"}:
        signs.append((-1.0, 1.0, 1.0))
    if symmetry == "xy":
        signs.extend(((1.0, -1.0, 1.0), (-1.0, -1.0, 1.0)))
    return (
        np.vstack([points * np.asarray(image_signs, dtype=float) for image_signs in signs]),
        np.vstack([triangles + index * points.shape[0] for index, _image_signs in enumerate(signs)]),
    )


__all__ = ["ExteriorFieldEvaluationService", "ExteriorFieldTask"]
