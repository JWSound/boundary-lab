"""Geometry generation lifecycle and Ath runtime configuration.

A controller, not a mixin: it reaches the UI through :class:`WorkflowView` and
:class:`PlotPresenter`, and the domain through :class:`GeometryInputs`. It
imports no Qt widgets — even the wait cursor goes through the view seam.

Follows the shape of :mod:`blab.ui.main_window.backend_health`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from blab.generators.ath import ATH_PROVIDER_ID
from blab.generators.base import GenerationCompleted, GenerationRequest
from blab.generators.registry import generator_info
from blab.project.model import generator_mesh_name
from blab.ui.application_state import OperationPhase
from blab.ui.main_window.constants import (
    ATH_BUNDLE_DIR,
    GENERATED_GEOMETRY_ROOT,
)
from blab.ui.main_window.workflow_view import GeometryInputs, PlotPresenter, WorkflowView
from blab.ui.operation_controllers import GeometryController, SolveController

#: Stop stays hidden this long, so a short generation cannot be interrupted
#: part-way through writing its output.
CANCEL_DELAY_MS = 3000


class GeometryWorkflowController(QObject):
    """Geometry generation lifecycle and Ath runtime configuration."""

    #: Emitted when generated geometry changes what the mesh preview shows.
    mesh_state_changed = Signal(str)

    #: Emitted when starting a generation invalidates existing solve results.
    solve_results_invalidated = Signal(str)
    generation_accepted = Signal(object)

    def __init__(
        self,
        parent: QObject | None,
        *,
        view: WorkflowView,
        plots: PlotPresenter,
        inputs: GeometryInputs,
        geometry_controller: GeometryController,
        solve_controller: SolveController,
    ) -> None:
        super().__init__(parent)
        self._view = view
        self._plots = plots
        self._inputs = inputs
        self._geometry_controller = geometry_controller
        self._solve_controller = solve_controller
        self._pending_request_id: str | None = None

    # -- Ath runtime --------------------------------------------------------

    def _find_ath_exe(self) -> Path:
        bundled = ATH_BUNDLE_DIR / "ath202608.exe"
        if bundled.exists():
            return bundled
        for root in (Path.cwd(), Path.cwd().parent):
            candidate = root / "ath202608.exe"
            if candidate.exists():
                return candidate
        return bundled

    # -- generation ---------------------------------------------------------

    @Slot()
    def generate_geometry(self) -> None:
        if self._geometry_controller.active or self._solve_controller.active:
            return
        document = self._inputs.active_generator_document()
        if document is None:
            self._view.warn("No waveguide design", "Add a waveguide design before generating.")
            return
        mesh_name = generator_mesh_name(document)
        case_name = f"{mesh_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{document.id}"
        run_root = GENERATED_GEOMETRY_ROOT
        provider_options = {}
        try:
            provider = generator_info(document.provider_id)
            if document.provider_id == ATH_PROVIDER_ID:
                provider_options["ath_exe"] = str(self._find_ath_exe())
        except Exception as exc:
            self._view.show_status("Generate failed")
            self._view.show_error("Geometry generation failed", str(exc))
            return

        revision, configuration = self._inputs.generation_context()
        request = GenerationRequest(
            provider_id=document.provider_id,
            document_id=document.id,
            mesh_name=mesh_name,
            source=document.source,
            run_root=run_root,
            case_name=case_name,
            provider_options=provider_options,
            project_revision=revision,
            configuration=configuration,
        )
        self.solve_results_invalidated.emit("geometry_generation_started")
        self._view.show_status(f"Generating {document.name} with {provider.label}...")
        # Stop is withheld until _enable_geometry_cancel_if_active fires.
        self._view.set_workflow_phase(OperationPhase.RUNNING, cancel_available=False)
        self._view.set_plot_exports_available(False)
        self._plots.refresh_contour_controls()
        self._view.set_busy_cursor(True)

        self._pending_request_id = request.request_id
        self._geometry_controller.start(request)
        QTimer.singleShot(CANCEL_DELAY_MS, self._enable_geometry_cancel_if_active)

    @Slot()
    def _enable_geometry_cancel_if_active(self) -> None:
        if self._geometry_controller.state.phase == OperationPhase.RUNNING:
            # Re-applying RUNNING only turns Cancel on; the other controls are
            # already disabled for the duration of the generation.
            self._view.set_workflow_phase(OperationPhase.RUNNING)

    # -- results ------------------------------------------------------------

    @Slot(object)
    def _on_geometry_generated(self, completed: GenerationCompleted) -> None:
        if completed.request.request_id != self._pending_request_id:
            return
        self._pending_request_id = None
        try:
            result = self._inputs.accept_generation(completed)
        except Exception as exc:
            self._on_geometry_generation_failed(str(exc))
            return
        self.mesh_state_changed.emit("geometry_generated")
        self.generation_accepted.emit(completed)
        self._view.show_status(f"Generated and cleaned {result.output_dir}")
        self._view.show_mesh_quality_warning(result)

    @Slot(str)
    def _on_geometry_generation_failed(self, message: str) -> None:
        self._view.show_status("Generate failed")
        self._view.show_error("Geometry generation failed", message)

    @Slot()
    def _on_geometry_generation_cancelled(self) -> None:
        self._view.show_status("Geometry generation stopped")

    @Slot()
    def _on_geometry_generation_finished(self) -> None:
        self._pending_request_id = None
        self._view.set_busy_cursor(False)
        self._view.set_workflow_phase(OperationPhase.IDLE)
