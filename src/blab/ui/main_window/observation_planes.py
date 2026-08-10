"""Main-window coordination for observation-plane authoring."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from PySide6.QtCore import QObject, Qt, Slot
from PySide6.QtWidgets import QWidget

from blab.observation_planes import ObservationPlane, new_observation_plane
from blab.ui.exterior_field_service import ExteriorFieldEvaluationService, ExteriorFieldTask
from blab.ui.observation_plane_dialog import ObservationPlanePropertiesDialog
from blab.ui.observation_plane_results import ObservationFieldResults
from blab.ui.project_state import ProjectDocument


class ObservationPlaneController(QObject):
    """Keep persisted plane definitions and their viewport actors synchronized."""

    def __init__(
        self,
        parent: QObject | None,
        *,
        window: QWidget,
        preview,
        project: Callable[[], ProjectDocument],
        show_status: Callable[[str], None],
        field_results: Callable[[], ObservationFieldResults | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._window = window
        self._preview = preview
        self._project = project
        self._show_status = show_status
        self._field_results = field_results or (lambda: None)
        self._dialogs: dict[str, ObservationPlanePropertiesDialog] = {}
        self._active_plane_id: str | None = None
        self._exterior_fields = ExteriorFieldEvaluationService(self)
        self._exterior_fields.completed.connect(self._exterior_field_completed)
        self._exterior_fields.failed.connect(self._exterior_field_failed)
        self._exterior_fields.discarded.connect(self._exterior_field_discarded)
        preview.newObservationPlaneRequested.connect(self.create_plane)
        preview.observationPlaneChanged.connect(self.update_plane)
        preview.observationPlanePropertiesRequested.connect(self.open_properties)
        preview.observationPlaneDeleteRequested.connect(self.delete_plane)
        if hasattr(preview, "observationPlaneExteriorFieldRequested"):
            preview.observationPlaneExteriorFieldRequested.connect(self._request_exterior_field)

    def sync_view(self, *, selected_id: str | None = None) -> None:
        available_ids = {plane.id for plane in self._project().observation_planes}
        if self._active_plane_id not in available_ids:
            self._active_plane_id = None
        results = self._field_results()
        if hasattr(self._preview, "set_observation_plane_results"):
            self._preview.set_observation_plane_results(results)
        frequencies = () if results is None else tuple(float(value) for value in results.frequencies_hz)
        responses = () if results is None else results.response_options
        for dialog in self._dialogs.values():
            dialog.set_solved_results(frequencies, responses)
            dialog.set_active(dialog.plane.id == self._active_plane_id)
        self._preview.set_observation_planes(self._project().observation_planes, selected_id=selected_id)
        if hasattr(self._preview, "set_observation_plane_active"):
            self._preview.set_observation_plane_active(self._active_plane_id)

    @Slot(object)
    def create_plane(self, placement: object) -> None:
        if not isinstance(placement, dict):
            return
        project = self._project()
        used_names = {plane.name for plane in project.observation_planes}
        index = 1
        while f"Observation Plane {index}" in used_names:
            index += 1
        plane = new_observation_plane(
            f"Observation Plane {index}",
            center_m=tuple(placement.get("center_m", (0.0, 0.0, 0.0))),
            orientation_wxyz=tuple(placement.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0))),
            width_m=float(placement.get("width_m", 0.25)),
            height_m=float(placement.get("height_m", placement.get("width_m", 0.25))),
        )
        project.observation_planes = (*project.observation_planes, plane)
        self.sync_view(selected_id=plane.id)
        self._show_status(f"Added {plane.name}")

    @Slot(object)
    def update_plane(self, updated: object) -> None:
        if not isinstance(updated, ObservationPlane):
            return
        updated = updated.validated()
        current = self._find(updated.id)
        if current is None:
            return
        dialog = self._dialogs.get(updated.id)
        if dialog is not None:
            updated = replace(
                current,
                center_m=updated.center_m,
                orientation_wxyz=updated.orientation_wxyz,
                width_m=updated.width_m,
                height_m=updated.height_m,
            )
            dialog.update_geometry(updated)
        self._store_plane(updated)
        self._show_status(f"Updated {updated.name}")

    @Slot(str)
    def open_properties(self, plane_id: str) -> None:
        plane = self._find(plane_id)
        if plane is None:
            return
        existing = self._dialogs.get(plane_id)
        if existing is not None:
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return
        results = self._field_results()
        dialog = ObservationPlanePropertiesDialog(
            plane,
            self._window,
            active=plane.id == self._active_plane_id,
            solved_frequencies_hz=(() if results is None else tuple(float(value) for value in results.frequencies_hz)),
            response_options=(() if results is None else results.response_options),
        )
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.previewChanged.connect(self._preview_properties)
        dialog.activeChanged.connect(self.set_active_plane)
        dialog.accepted.connect(lambda plane_id=plane_id, dialog=dialog: self._accept_properties(plane_id, dialog))
        dialog.rejected.connect(lambda plane_id=plane_id: self._reject_properties(plane_id))
        self._dialogs[plane_id] = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _accept_properties(self, plane_id: str, dialog: ObservationPlanePropertiesDialog) -> None:
        self._dialogs.pop(plane_id, None)
        updated = dialog.plane
        if updated.id == plane_id and self._store_plane(updated):
            self._show_status(f"Updated {updated.name}")
        self._stop_animation()
        self.sync_view(selected_id=None if plane_id == self._active_plane_id else plane_id)

    def _reject_properties(self, plane_id: str) -> None:
        self._dialogs.pop(plane_id, None)
        self._stop_animation()
        self.sync_view(selected_id=None if plane_id == self._active_plane_id else plane_id)

    @Slot(object, bool)
    def _preview_properties(self, updated: object, animate: bool) -> None:
        if not isinstance(updated, ObservationPlane):
            return
        project = self._project()
        preview_planes = tuple(updated if plane.id == updated.id else plane for plane in project.observation_planes)
        selected_id = None if updated.id == self._active_plane_id else updated.id
        self._preview.set_observation_planes(preview_planes, selected_id=selected_id)
        if hasattr(self._preview, "set_observation_plane_animation"):
            self._preview.set_observation_plane_animation(updated.id, animate)

    @Slot(str, bool)
    def set_active_plane(self, plane_id: str, active: bool) -> None:
        if active and self._find(plane_id) is not None:
            self._active_plane_id = plane_id
        elif not active and self._active_plane_id == plane_id:
            self._active_plane_id = None
        for candidate_id, dialog in self._dialogs.items():
            dialog.set_active(candidate_id == self._active_plane_id)
        if hasattr(self._preview, "set_observation_plane_active"):
            self._preview.set_observation_plane_active(self._active_plane_id)

    @Slot(str)
    def delete_plane(self, plane_id: str) -> None:
        plane = self._find(plane_id)
        if plane is None:
            return
        project = self._project()
        project.observation_planes = tuple(item for item in project.observation_planes if item.id != plane_id)
        if self._active_plane_id == plane_id:
            self._active_plane_id = None
        dialog = self._dialogs.get(plane_id)
        if dialog is not None:
            dialog.reject()
        self.sync_view()
        self._show_status(f"Deleted {plane.name}")

    def _store_plane(self, updated: ObservationPlane) -> bool:
        updated = updated.validated()
        project = self._project()
        if not any(plane.id == updated.id for plane in project.observation_planes):
            return False
        project.observation_planes = tuple(
            updated if plane.id == updated.id else plane for plane in project.observation_planes
        )
        return True

    def _stop_animation(self) -> None:
        if hasattr(self._preview, "set_observation_plane_animation"):
            self._preview.set_observation_plane_animation(None, False)

    @Slot(object)
    def _request_exterior_field(self, task: object) -> None:
        if isinstance(task, ExteriorFieldTask):
            self._exterior_fields.submit(task)

    @Slot(object, object)
    def _exterior_field_completed(self, key: object, pressure: object) -> None:
        if hasattr(self._preview, "set_observation_plane_exterior_field"):
            self._preview.set_observation_plane_exterior_field(key, pressure)

    @Slot(object, str)
    def _exterior_field_failed(self, key: object, message: str) -> None:
        if hasattr(self._preview, "set_observation_plane_exterior_field_failed"):
            self._preview.set_observation_plane_exterior_field_failed(key, message)

    @Slot(object)
    def _exterior_field_discarded(self, key: object) -> None:
        if hasattr(self._preview, "discard_observation_plane_exterior_field_request"):
            self._preview.discard_observation_plane_exterior_field_request(key)

    def _find(self, plane_id: str) -> ObservationPlane | None:
        return next((plane for plane in self._project().observation_planes if plane.id == plane_id), None)
