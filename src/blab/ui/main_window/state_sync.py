"""Cross-cutting state signals and the solve-invalidation confirmation flow."""

from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QMessageBox,
)

from blab.ui.application_state import solve_invalidation_policy


class StateSyncMixin:
    """Cross-cutting state signals and the solve-invalidation confirmation flow.

    Mixed into :class:`~blab.ui.main_window.window.MainWindow`.
    """

    def _connect_state_events(self) -> None:
        self.mesh_state_changed.connect(self._on_mesh_state_changed)
        self.project_state_changed.connect(self._on_project_state_changed)
        self.solve_results_invalidated.connect(self._on_solve_results_invalidated)
        self.visualization_settings_changed.connect(self._on_visualization_settings_changed)

    def _connect_operation_controllers(self) -> None:
        self.geometry_controller.completed.connect(self.geometry_workflow._on_geometry_generated)
        self.geometry_controller.status.connect(self.show_status)
        self.geometry_controller.failed.connect(self.geometry_workflow._on_geometry_generation_failed)
        self.geometry_controller.cancelled.connect(self.geometry_workflow._on_geometry_generation_cancelled)
        self.geometry_controller.finished.connect(self.geometry_workflow._on_geometry_generation_finished)
        self.solve_controller.initialized.connect(self.solve_workflow._on_solver_initialized)
        self.solve_controller.result_ready.connect(self.solve_workflow._on_frequency_result)
        self.solve_controller.system_result_ready.connect(self.solve_workflow._on_system_frequency_result)
        self.solve_controller.status.connect(self.show_status)
        self.solve_controller.failed.connect(self.solve_workflow._on_solve_failed)
        self.solve_controller.finished.connect(self.solve_workflow._on_solve_finished)

    @Slot(str)
    def _on_mesh_state_changed(self, reason: str) -> None:
        self._refresh_mesh_preview()
        if reason in {"mesh_config_changed", "imported_mesh_files_reloaded"}:
            self._record_imported_mesh_source_fingerprints()
        self.set_system_config_available(self.has_solver_meshes())

    @Slot(str)
    def _on_project_state_changed(self, _reason: str) -> None:
        self._refresh_mesh_preview()
        self._record_imported_mesh_source_fingerprints()
        self.set_system_config_available(self.has_solver_meshes())
        self.set_max_spl_available(bool(self.max_spl_channel_names()))

    @Slot(str)
    def _on_solve_results_invalidated(self, reason: str) -> None:
        policy = solve_invalidation_policy(reason)
        if policy.clear_solve_results:
            self.clear_plots()
            self.set_max_spl_available(bool(self.max_spl_channel_names()))
        if policy.clear_comparison_history:
            self.clear_comparison_history()

    def _has_solved_data(self) -> bool:
        return self._solve_session().has_solved_data()

    def _confirm_clear_solved_data(self) -> bool:
        if not self._has_solved_data():
            return True
        message = QMessageBox(
            QMessageBox.Warning,
            "Clear solved data?",
            "Applying this action will clear solved data",
            QMessageBox.NoButton,
            self,
        )
        continue_button = message.addButton("Continue", QMessageBox.AcceptRole)
        cancel_button = message.addButton("Cancel", QMessageBox.RejectRole)
        message.setDefaultButton(cancel_button)
        message.exec()
        return message.clickedButton() is continue_button

    @Slot(str)
    def _on_visualization_settings_changed(self, _reason: str) -> None:
        self.refresh_plots()

    def reconcile_symmetry_with_backend(self) -> bool:
        """Reconcile the project's symmetry with what the backend can honour.

        The capability question belongs to the backend health controller; the
        state mutation belongs here, because symmetry is project state.
        Returns True when the project was downgraded.
        """
        effective_symmetry = self.backend_health.effective_symmetry(self.symmetry, self.preferences)
        if effective_symmetry == self.symmetry:
            return False
        self.symmetry = effective_symmetry
        return True
