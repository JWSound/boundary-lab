"""Menu-driven dialogs: balloon, diagnostics, help, donate, mesh and system config."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt, QUrl, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QMessageBox,
)

from blab.solvers.registry import backend_info
from blab.ui.diagnostics import DiagnosticsDialog
from blab.ui.dialogs import (
    DonateDialog,
    MeshConfigDialog,
)
from blab.ui.main_window.constants import (
    HELP_GUIDE_PDF,
)
from blab.ui.main_window.helpers import (
    _mesh_entries_with_file_overrides,
)
from blab.ui.physical_system_migration import AUTO_SEEDED_EXTERIOR_KEY
from blab.ui.system_config import (
    SystemConfigDialog,
    inspect_system_meshes,
    sync_physical_system_meshes,
)


class DialogActionsMixin:
    """Menu-driven dialogs: balloon, diagnostics, help, donate, mesh and system config.

    Mixed into :class:`~blab.ui.main_window.window.MainWindow`.
    """

    @Slot()
    def open_balloon_plot(self) -> None:
        if self.live_dataset is None or self.live_dataset.solved_count == 0:
            QMessageBox.warning(self, "No balloon data", "Run a solve before opening the balloon plot.")
            return

        balloon_window = self.balloon_window
        if balloon_window is not None and balloon_window.isVisible():
            refresh_balloon = getattr(balloon_window, "refresh_from_latest_results", None)
            if callable(refresh_balloon):
                refresh_balloon()
            balloon_window.raise_()
            balloon_window.activateWindow()
            return

        self.live_dataset.set_channel_synthesis(
            self.channel_configs(),
            flat_target_reference_angle_deg=self.preferences.horizontal_normalization_angle,
        )
        raw_balloon = self.live_dataset.as_balloon_raw_bundle()
        if raw_balloon is None:
            QMessageBox.warning(
                self,
                "No balloon data",
                "Enable spherical sampling in Preferences before running a solve.",
            )
            return

        try:
            from blab.ui.balloon import BalloonPlotWindow

            self.balloon_window = BalloonPlotWindow(
                raw_balloon,
                min_db=self.preferences.spl_min_db,
                max_db=self.preferences.spl_max_db,
                polar_smoothing=self.preferences.polar_smoothing,
                raw_balloon_data_provider=lambda: (
                    None if self.live_dataset is None else self.live_dataset.as_balloon_raw_bundle()
                ),
                file_dialog_service=self.file_dialogs,
                parent=self,
            )
            self.balloon_window.show()
            self.balloon_window.raise_()
        except Exception as exc:
            QMessageBox.critical(self, "Balloon plot failed", str(exc))

    @Slot()
    def open_diagnostics(self) -> None:
        dialog = DiagnosticsDialog(
            self.preferences,
            self,
            context_provider=self._diagnostic_context,
        )
        dialog.exec()

    def _diagnostic_context(self) -> dict[str, object]:
        backend = backend_info(self.preferences.solve_backend)
        backend_details: dict[str, object] = {
            "id": backend.backend_id,
            "label": backend.label,
            "remote": backend.capabilities.is_remote,
            "supports symmetry": backend.capabilities.supports_symmetry,
            "supports spherical sampling": backend.capabilities.supports_spherical_sampling,
            "supports channel resynthesis": backend.capabilities.supports_channel_resynthesis,
        }
        if backend.capabilities.is_remote:
            backend_details["server health"] = (
                "reachable (cached)" if self.backend_health.matches_preferences() else "not confirmed"
            )
            if self.backend_health.matches_preferences() and self.backend_health.payload is not None:
                backend_details["server solver"] = (
                    self.backend_health.payload.get("solver_label")
                    or self.backend_health.payload.get("solver")
                    or "unknown"
                )

        geometry_state = self.geometry_controller.state
        solve_state = self.solve_controller.state
        operations: dict[str, object] = {
            "geometry": {
                "phase": geometry_state.phase.value,
                "message": geometry_state.message or "none",
                "last error": self.geometry_controller.last_error or "none",
            },
            "solve": {
                "phase": solve_state.phase.value,
                "message": solve_state.message or "none",
                "last error": self.solve_controller.last_error or "none",
            },
        }
        completion = self.solve_controller.last_completion
        if completion is not None:
            solve_details = operations["solve"]
            assert isinstance(solve_details, dict)
            solve_details.update(
                {
                    "solved frequencies": completion.solved_count,
                    "requested frequencies": completion.expected_count,
                    "elapsed seconds": round(completion.elapsed_s, 3),
                }
            )

        enabled_generated_meshes = sum(
            1
            for script in self.generator_documents
            if script.mesh_enabled and script.id in self.generated_geometry_by_document_id
        )
        enabled_imported_meshes = sum(1 for mesh in self.imported_meshes if mesh.enabled)
        result_details: dict[str, object] = {
            "solved frequencies": 0 if self.live_dataset is None else self.live_dataset.solved_count,
        }
        frequencies = self.frequency_range()
        context: dict[str, object] = {
            "backend": backend_details,
            "project": {
                "file": self.project_path.name if self.project_path is not None else "unsaved",
                "modified": self._has_unsaved_project_changes(),
                "waveguide designs": len(self.generator_documents),
                "enabled meshes": enabled_generated_meshes + enabled_imported_meshes,
                "imported meshes": len(self.imported_meshes),
                "radiators": len(self.all_radiators()),
                "channels": len(self.channel_configs()),
                "symmetry": self.symmetry,
                "stitch exterior meshes": self.stitch_imported_meshes,
                "frequency minimum Hz": frequencies.min_hz,
                "frequency maximum Hz": frequencies.max_hz,
                "frequency count": frequencies.count,
            },
            "operations": operations,
            "results": result_details,
        }

        if self.live_dataset is not None and self.live_dataset.results:
            latest_frequency = next(reversed(self.live_dataset.results))
            latest_result = self.live_dataset.results[latest_frequency]
            result_details.update(
                {
                    "latest frequency Hz": round(float(latest_result.freq_hz), 3),
                    "latest assembly seconds": round(float(latest_result.timings.assembly_s), 3),
                    "latest solve seconds": round(float(latest_result.timings.solve_s), 3),
                    "latest field seconds": round(float(latest_result.timings.field_s), 3),
                }
            )
            if latest_result.diagnostics is not None:
                result_details["latest convergence info"] = latest_result.diagnostics.convergence_info
                result_details["latest solver message"] = latest_result.diagnostics.message or "none"

        return context

    @Slot()
    def open_donate(self) -> None:
        dialog = DonateDialog(self)
        dialog.exec()

    @Slot()
    def open_help(self) -> None:
        if not HELP_GUIDE_PDF.exists():
            QMessageBox.warning(
                self,
                "Help guide missing",
                f"The Boundary Lab guide PDF could not be found:\n{HELP_GUIDE_PDF}",
            )
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(HELP_GUIDE_PDF))):
            QMessageBox.warning(
                self,
                "Help guide failed",
                "Unable to open the Boundary Lab guide PDF in the default viewer.",
            )

    @Slot()
    def open_mesh_config(self) -> None:
        self.reconcile_symmetry_with_backend()
        symmetry_enabled = self.backend_health.selected_backend_supports_symmetry()
        dialog = MeshConfigDialog(
            self._mesh_config_dialog_entries(),
            stitch_imported_meshes=self.stitch_imported_meshes,
            symmetry=self.symmetry,
            symmetry_enabled=symmetry_enabled,
            file_dialog_service=self.file_dialogs,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return

        meshes = dialog.meshes()
        replaced_mesh_names = dialog.replaced_mesh_names()
        stitch_imported_meshes = dialog.stitch_imported_meshes()
        symmetry = dialog.symmetry() if symmetry_enabled else self.symmetry
        config_changed = (
            meshes != self._mesh_config_dialog_entries()
            or stitch_imported_meshes != self.stitch_imported_meshes
            or symmetry != self.symmetry
        )
        if not config_changed:
            self.status_label.setText("Mesh config unchanged")
            return
        if not self._confirm_clear_solved_data():
            return

        try:
            self.status_label.setText("Cleaning imported meshes...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self._apply_mesh_config_dialog_entries(meshes)
            self.stitch_imported_meshes = stitch_imported_meshes
            if symmetry_enabled:
                self.symmetry = symmetry
            self.imported_meshes = self._clean_imported_meshes(self.imported_meshes)
            if replaced_mesh_names:
                if self.project.physical_system is not None:
                    inspected_meshes = inspect_system_meshes(self._mesh_config_dialog_entries())
                    self.project.physical_system = sync_physical_system_meshes(
                        self.project.physical_system,
                        inspected_meshes,
                    )
                self.apply_saved_imported_source_config(self.surface_tags_for_meshes())
            self.mesh_state_changed.emit("mesh_config_changed")
            self.solve_results_invalidated.emit("mesh_config_changed")
            self.status_label.setText(
                f"Mesh config updated: {len(self._active_imported_meshes())}/{len(self.imported_meshes)} meshes enabled"
            )
        except Exception as exc:
            self.status_label.setText("Mesh config failed")
            QMessageBox.critical(self, "Mesh config failed", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    @Slot()
    def open_system_config(self) -> None:
        try:
            meshes = inspect_system_meshes(self._mesh_config_dialog_entries())
            symmetry_analysis_meshes = inspect_system_meshes(self.mesh_entries_for_symmetry(self.symmetry))
        except Exception as exc:
            QMessageBox.critical(self, "System", f"Could not inspect the enabled meshes:\n{exc}")
            return
        if not meshes:
            QMessageBox.warning(self, "System", "Enable at least one mesh before configuring the system.")
            return
        self.ensure_seeded_exterior_system()
        system = self.project.physical_system
        if system is not None:
            system = sync_physical_system_meshes(system, meshes)
        dialog = SystemConfigDialog(
            meshes,
            system,
            tuple(channel.name for channel in self.channel_configs()),
            self.project.component_channel_by_id,
            self,
            stitch_exterior_meshes=self.stitch_imported_meshes,
            interface_output_root=self.mesh_service().output_root,
            symmetry_mode=self.symmetry,
            symmetry_analysis_meshes=symmetry_analysis_meshes,
        )
        dialog.systemApplied.connect(self._apply_system_config)
        dialog.exec()

    @Slot(object)
    def _apply_system_config(self, configuration) -> None:
        mesh_file_overrides = dict(getattr(configuration, "mesh_file_overrides_by_name", {}))
        updated_imported_meshes = _mesh_entries_with_file_overrides(
            self.imported_meshes,
            mesh_file_overrides,
        )
        if (
            configuration.system == self.project.physical_system
            and configuration.component_channel_by_id == self.project.component_channel_by_id
            and configuration.stitch_exterior_meshes == self.stitch_imported_meshes
            and updated_imported_meshes == self.imported_meshes
        ):
            self.status_label.setText("System unchanged")
            return
        self.imported_meshes = updated_imported_meshes
        metadata = dict(configuration.system.metadata)
        metadata.pop(AUTO_SEEDED_EXTERIOR_KEY, None)
        self.project.physical_system = replace(configuration.system, metadata=metadata)
        self.project.component_channel_by_id = dict(configuration.component_channel_by_id)
        self.project.source_config_by_name = {}
        self.stitch_imported_meshes = bool(configuration.stitch_exterior_meshes)
        reason = "system_interface_mesh_built" if mesh_file_overrides else "system_config_changed"
        self.project_state_changed.emit(reason)
        self.solve_results_invalidated.emit("system_config_changed")
        self.status_label.setText("System updated")
