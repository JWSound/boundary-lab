"""Main-window coordination for observation-plane authoring."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QDialog, QWidget

from blab.observation_planes import ObservationPlane, new_observation_plane
from blab.ui.observation_plane_dialog import ObservationPlanePropertiesDialog
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
    ) -> None:
        super().__init__(parent)
        self._window = window
        self._preview = preview
        self._project = project
        self._show_status = show_status
        preview.newObservationPlaneRequested.connect(self.create_plane)
        preview.observationPlaneChanged.connect(self.update_plane)
        preview.observationPlanePropertiesRequested.connect(self.open_properties)
        preview.observationPlaneDeleteRequested.connect(self.delete_plane)

    def sync_view(self, *, selected_id: str | None = None) -> None:
        self._preview.set_observation_planes(self._project().observation_planes, selected_id=selected_id)

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
        project = self._project()
        if not any(plane.id == updated.id for plane in project.observation_planes):
            return
        project.observation_planes = tuple(
            updated if plane.id == updated.id else plane for plane in project.observation_planes
        )
        self._show_status(f"Updated {updated.name}")

    @Slot(str)
    def open_properties(self, plane_id: str) -> None:
        plane = self._find(plane_id)
        if plane is None:
            return
        dialog = ObservationPlanePropertiesDialog(plane, self._window)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        updated = dialog.plane
        self.update_plane(updated)
        self.sync_view(selected_id=updated.id)

    @Slot(str)
    def delete_plane(self, plane_id: str) -> None:
        plane = self._find(plane_id)
        if plane is None:
            return
        project = self._project()
        project.observation_planes = tuple(item for item in project.observation_planes if item.id != plane_id)
        self.sync_view()
        self._show_status(f"Deleted {plane.name}")

    def _find(self, plane_id: str) -> ObservationPlane | None:
        return next((plane for plane in self._project().observation_planes if plane.id == plane_id), None)
