"""Solve orchestration: start, cancel, and per-frequency result handling.

This is a controller, not a mixin. It reaches the UI only through the three
declared protocols — :class:`WorkflowView`, :class:`PlotPresenter` and
:class:`SolveInputs` — reads live results from a :class:`SolveSession`, and
takes everything else as a constructor argument. It imports no Qt widgets, so
a UI revamp replaces its collaborators without touching it.

Follows the shape of :mod:`blab.ui.main_window.backend_health`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from blab.live import (
    FrequencyResult,
    LiveSolveDataset,
    build_log_frequencies,
)
from blab.physical_model import (
    PhysicalSolveKind,
    infer_physical_solve_kind,
)
from blab.solve_results import (
    SolvedSystemBuilder,
    SolveProvenance,
    legacy_result_domains,
    legacy_result_to_system_result,
)
from blab.symmetry import SymmetryValidationError
from blab.system_contract import SystemFrequencyResult
from blab.ui.application_state import OperationPhase, SolveCompletion
from blab.ui.exterior_system import exterior_bem_inputs
from blab.ui.main_window.solve_session import SolveSession
from blab.ui.main_window.workflow_view import PlotPresenter, SolveInputs, WorkflowView
from blab.ui.main_window_widgets import (
    format_frequency_solve_timings,
)
from blab.ui.operation_controllers import (
    GeometryController,
    SolveController,
    SolveRequest,
)
from blab.ui.plots import (
    FINAL_ISOBAR_ANGLE_SAMPLES,
    FINAL_ISOBAR_FREQ_SAMPLES,
)
from blab.ui.project_state import ProjectDocument
from blab.ui.server_tokens import load_server_access_token
from blab.ui.settings import (
    GuiPreferences,
    balloon_sampling_points,
)
from blab.ui.simulation_assembler import SimulationAssembler, SimulationParameters
from blab.ui.system_config import (
    inspect_system_meshes,
    sync_physical_system_meshes,
)
from blab.ui.system_solve import prepare_coupled_ui_solve


class SolveWorkflowController(QObject):
    """Solve orchestration: start, cancel, and per-frequency result handling."""

    #: Emitted when a solve changes something the mesh preview derives from,
    #: so the host can invalidate anything downstream of it.
    mesh_state_changed = Signal(str)

    def __init__(
        self,
        parent: QObject | None,
        *,
        view: WorkflowView,
        plots: PlotPresenter,
        inputs: SolveInputs,
        session: SolveSession,
        project: Callable[[], ProjectDocument],
        preferences: Callable[[], GuiPreferences],
        assembler: SimulationAssembler,
        geometry_controller: GeometryController,
        solve_controller: SolveController,
    ) -> None:
        super().__init__(parent)
        self._view = view
        self._plots = plots
        self._inputs = inputs
        self._session = session
        self._project = project
        self._read_preferences = preferences
        self._assembler = assembler
        self._geometry_controller = geometry_controller
        self._solve_controller = solve_controller

    # -- starting a run -----------------------------------------------------

    def _begin_run(self, status: str) -> None:
        """Clear the previous run and lock the inputs for a starting solve.

        The same eleven lines opened all three solve paths; ``clear_plots``
        already resets the session, so the resets that used to bracket it were
        redundant.
        """
        self._plots.clear_plots()
        self._plots.apply_last_completed_comparison()
        self._view.set_balloon_plot_available(False)
        self._view.set_workflow_phase(OperationPhase.RUNNING)
        self._view.set_plot_exports_available(False)
        self._view.set_polar_export_available(False)
        self._view.set_on_axis_export_available(False)
        self._plots.refresh_contour_controls()
        self._view.show_status(status)

    def _solve_request(self, config, ordered_frequencies) -> SolveRequest:
        preferences = self._read_preferences()
        return SolveRequest(
            config=config,
            ordered_frequencies=ordered_frequencies,
            worker_count=1,
            backend_id=preferences.solve_backend,
            server_url=preferences.solve_server_url,
            server_access_token=load_server_access_token(preferences.solve_server_url),
        )

    def _simulation_parameters(self, frequencies, preferences: GuiPreferences) -> SimulationParameters:
        return SimulationParameters(
            freq_min_hz=float(frequencies.min_hz),
            freq_max_hz=float(frequencies.max_hz),
            freq_count=frequencies.count,
            observation_distance_m=preferences.polar_observation_distance_m,
            polar_angle_step_deg=preferences.polar_angle_step_deg,
            use_burton_miller=preferences.use_burton_miller,
            gmres_tolerance=preferences.gmres_tolerance,
            normalized_channel_correction=preferences.normalized_channel_correction,
            horizontal_normalization_angle_deg=preferences.horizontal_normalization_angle,
            spherical_sampling_enabled=preferences.spherical_sampling_enabled,
            spherical_sampling_points=balloon_sampling_points(preferences.balloon_angle_precision_deg),
            symmetry=self._project().symmetry,
        )

    @Slot()
    def start_solve(self) -> None:
        if self._geometry_controller.active or self._solve_controller.active:
            return
        if not self._inputs.has_solver_meshes():
            self._view.warn("No mesh", "Enable at least one generated or imported mesh before solving.")
            return
        self._inputs.ensure_seeded_exterior_system()
        project = self._project()
        if project.physical_system is not None:
            try:
                solve_kind = infer_physical_solve_kind(project.physical_system)
            except ValueError as exc:
                self._view.warn("System solve", str(exc))
                return
            if solve_kind == PhysicalSolveKind.EXTERIOR_BEM:
                self._start_exterior_system_solve()
            else:
                self._start_coupled_system_solve()
            return
        radiators = self._inputs.all_radiators()
        if not radiators:
            self._view.warn(
                "No driven surfaces",
                "Open System and add a prescribed-velocity component to a moving boundary.",
            )
            return
        if self._inputs.reconcile_symmetry_with_backend():
            self.mesh_state_changed.emit("symmetry_disabled_for_backend")

        try:
            assembly = self._inputs.prepare_mesh_assembly(radiators)
            mesh_configs = assembly.mesh_configs
            radiators = assembly.radiators
        except Exception as exc:
            self._view.show_stitch_or_generic_error("Imported mesh preparation failed", exc)
            return
        frequencies = self._view.frequency_range().normalized()
        channels = self._inputs.solver_channel_configs(radiators)
        try:
            prepared_simulation = self._assembler.prepare(
                mesh_configs=mesh_configs,
                radiators=radiators,
                channels=channels,
                parameters=self._simulation_parameters(frequencies, self._read_preferences()),
            )
        except SymmetryValidationError as exc:
            self._view.warn("Symmetry validation failed", str(exc))
            return

        self._begin_run("Initializing Solver...")
        self._solve_controller.start(
            self._solve_request(prepared_simulation.config, prepared_simulation.ordered_frequencies)
        )

    def _start_exterior_system_solve(self) -> None:
        if self._inputs.reconcile_symmetry_with_backend():
            self.mesh_state_changed.emit("symmetry_disabled_for_backend")
        project = self._project()
        preferences = self._read_preferences()
        symmetry = project.symmetry
        try:
            meshes = inspect_system_meshes(self._inputs.mesh_entries_for_symmetry(symmetry))
            system = sync_physical_system_meshes(project.physical_system, meshes)
            project.physical_system = system
            inputs = exterior_bem_inputs(
                system,
                component_channel_by_id=project.component_channel_by_id,
                symmetry_mode=symmetry,
            )
            mesh_configs, radiators = self._inputs.mesh_service().prepare_mesh_configs(
                inputs.mesh_configs,
                inputs.radiators,
                stitch_meshes_enabled=project.stitch_imported_meshes,
                stitch_tolerance_mm=preferences.stitch_tolerance_mm,
                symmetry=symmetry,
            )
            prepared_simulation = self._assembler.prepare(
                mesh_configs=mesh_configs,
                radiators=radiators,
                channels=self._inputs.solver_channel_configs(radiators),
                parameters=self._simulation_parameters(self._view.frequency_range(), preferences),
            )
        except (ValueError, OSError, SymmetryValidationError) as exc:
            self._view.show_stitch_or_generic_error("Exterior system preparation failed", exc)
            return

        self._begin_run("Initializing exterior solver...")
        self._solve_controller.start(
            self._solve_request(prepared_simulation.config, prepared_simulation.ordered_frequencies)
        )

    def _start_coupled_system_solve(self) -> None:
        project = self._project()
        preferences = self._read_preferences()
        try:
            meshes = inspect_system_meshes(self._inputs.mesh_entries_for_symmetry(project.symmetry))
            system = sync_physical_system_meshes(project.physical_system, meshes)
            project.physical_system = system
            coupled_frequencies = self._view.frequency_range()
            prepared = prepare_coupled_ui_solve(
                system,
                freq_min_hz=float(coupled_frequencies.min_hz),
                freq_max_hz=float(coupled_frequencies.max_hz),
                freq_count=coupled_frequencies.count,
                observation_distance_m=preferences.polar_observation_distance_m,
                polar_angle_step_deg=preferences.polar_angle_step_deg,
                spherical_sampling_enabled=preferences.spherical_sampling_enabled,
                spherical_sampling_points=balloon_sampling_points(preferences.balloon_angle_precision_deg),
                component_channel_by_id=project.component_channel_by_id,
                backend_id=preferences.solve_backend,
                symmetry_mode=project.symmetry,
                observation_planes=project.observation_planes,
            )
        except Exception as exc:
            self._view.warn("Coupled solve", str(exc))
            return

        self._begin_run("Initializing coupled solver...")
        self._session.result_builder = SolvedSystemBuilder(
            frequencies_hz=prepared.request.frequencies_hz,
            excitation_ids=prepared.request.excitation_port_ids,
            provenance=SolveProvenance(
                backend_id=prepared.backend_id,
                solve_kind="coupled_bem_fem",
                solver_options=dict(prepared.request.solver_options),
            ),
            domains=prepared.result_domains,
            compiled_system=prepared.request.compiled_system,
        )
        self._solve_controller.start(prepared)

    # -- cancelling ---------------------------------------------------------

    @Slot()
    def cancel_current_operation(self) -> None:
        if self._geometry_controller.active:
            self.cancel_geometry_generation()
            return
        self.cancel_solve()

    @Slot()
    def cancel_geometry_generation(self) -> None:
        self._view.set_workflow_phase(OperationPhase.CANCELLING)
        if self._geometry_controller.active:
            self._geometry_controller.cancel()
            self._view.show_status("Stop requested; ending geometry generation...")

    @Slot()
    def cancel_solve(self) -> None:
        if self._solve_controller.active:
            self._solve_controller.cancel()
            self._view.show_status("Stop requested; waiting for current frequency...")

    # -- results ------------------------------------------------------------

    @Slot(object, object)
    def _on_solver_initialized(
        self,
        angles: np.ndarray,
        radiator_names: np.ndarray,
        sphere_metadata: dict[str, np.ndarray] | None,
    ) -> None:
        sphere_metadata = sphere_metadata or {}
        preferences = self._read_preferences()
        self._session.live_dataset = LiveSolveDataset(
            polar_angle_deg=np.asarray(angles, dtype=np.float32),
            radiator_names=np.asarray(radiator_names),
            channel_configs=self._inputs.channel_configs(),
            flat_target_normalization_enabled=preferences.normalized_channel_correction,
            flat_target_reference_angle_deg=preferences.horizontal_normalization_angle,
            sphere_r_distance_m=sphere_metadata.get("r_distance_m"),
            sphere_theta_polar_rad=sphere_metadata.get("theta_polar_rad"),
            sphere_phi_azimuth_rad=sphere_metadata.get("phi_azimuth_rad"),
        )
        self._view.show_status("Solving...")

    @Slot(object)
    def _on_frequency_result(self, result: FrequencyResult) -> None:
        live_dataset = self._session.live_dataset
        if live_dataset is None:
            return
        live_dataset.add(result)
        if self._session.result_builder is None:
            canonical = legacy_result_to_system_result(result)
            frequencies = self._view.frequency_range().normalized()
            self._session.result_builder = SolvedSystemBuilder(
                frequencies_hz=build_log_frequencies(
                    float(frequencies.min_hz),
                    float(frequencies.max_hz),
                    int(frequencies.count),
                ),
                excitation_ids=canonical.excitation_port_ids,
                provenance=SolveProvenance(
                    backend_id=self._read_preferences().solve_backend,
                    solve_kind="exterior_bem",
                ),
                domains=legacy_result_domains(live_dataset),
            )
            self._session.result_builder.add(canonical)
        elif self._session.result_builder.compiled_system is None:
            self._session.result_builder.add(legacy_result_to_system_result(result))
        self._view.show_status(
            f"Solved {live_dataset.solved_count}/{self._view.frequency_range().count} "
            f"({result.freq_hz:.1f} Hz) | {format_frequency_solve_timings(result)}"
        )
        if not self._read_preferences().live_plot_streaming:
            return
        self._plots.request_live_refresh()

    @Slot(object)
    def _on_system_frequency_result(self, result: SystemFrequencyResult) -> None:
        builder = self._session.result_builder
        if builder is None:
            raise RuntimeError("Received a physical-system result before its result builder was initialized.")
        builder.add(result)

    @Slot(str)
    def _on_solve_failed(self, message: str) -> None:
        self._view.show_error("Solve failed", message)
        self._view.show_status("Solve failed")

    @Slot(object)
    def _on_solve_finished(self, completion: SolveCompletion) -> None:
        self._plots.cancel_live_refresh()
        self._view.set_workflow_phase(OperationPhase.IDLE)
        session = self._session
        session.finalize_results(status=completion.phase.value)
        observation_planes = getattr(self._view, "observation_plane_controller", None)
        if observation_planes is not None:
            observation_planes.sync_view()
        elapsed_s = completion.elapsed_s
        if session.has_solved_data():
            solved_count = session.solved_count
            solve_completed = completion.completed
            session.use_final_isobar_resolution = solve_completed
            if solve_completed:
                self._view.show_status("Rendering final high-resolution plots...")
            refreshed_dataset = None
            if self._read_preferences().live_plot_streaming or solve_completed:
                refreshed_dataset = self._plots.refresh_plots()
            if solve_completed:
                if refreshed_dataset is None:
                    refreshed_dataset = self._plots.prepared_live_dataset(
                        angle_samples=FINAL_ISOBAR_ANGLE_SAMPLES,
                        freq_samples=FINAL_ISOBAR_FREQ_SAMPLES,
                    )
                if refreshed_dataset is not None:
                    session.last_completed_visualization = refreshed_dataset.snapshot()
            session.final_isobar_plots_rendered = solve_completed and bool(self._plots.visible_isobar_plots())
            self._view.set_plot_exports_available(True)
            self._view.set_polar_export_available(True)
            self._view.set_on_axis_export_available(session.live_dataset.supports_channel_resynthesis)
            self._view.set_balloon_plot_available(session.live_dataset.has_balloon_data)
            self._plots.refresh_contour_controls()
            elapsed_text = f" in {elapsed_s:.1f} s"
            if completion.phase == OperationPhase.CANCELLED:
                self._view.show_status(f"Solve stopped: {session.solved_count} frequencies{elapsed_text}")
                return
            if completion.phase == OperationPhase.FAILED:
                self._view.show_status(f"Solve failed after {solved_count} frequencies{elapsed_text}")
                return
            self._view.show_status(f"Solve complete: {solved_count} frequencies{elapsed_text}")
        elif completion.phase == OperationPhase.CANCELLED:
            self._view.show_status("Solve stopped")
        self._plots.refresh_contour_controls()
