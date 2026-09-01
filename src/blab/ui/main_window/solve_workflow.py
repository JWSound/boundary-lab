"""Solve orchestration: start, cancel, and per-frequency result handling.

This is a controller, not a mixin. It reaches the UI only through the three
declared protocols — :class:`WorkflowView`, :class:`PlotPresenter` and
:class:`SolveInputs` — reads live results from a :class:`SolveSession`, and
takes everything else as a constructor argument. It imports no Qt widgets, so
a UI revamp replaces its collaborators without touching it.

Follows the shape of :mod:`blab.ui.main_window.backend_health`.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from blab.acoustic_impedance import (
    ACOUSTIC_AREA_MISMATCH_WARNING_THRESHOLD,
    normalization_records,
)
from blab.config import MeshConfig
from blab.live import (
    AcousticLoadImpedanceDataset,
    ElectricalImpedanceDataset,
    FrequencyResult,
    LiveSolveDataset,
    TransducerMotionDataset,
    build_log_frequencies,
)
from blab.max_spl import max_spl_limits_from_payload, transducer_rated_resistance_ohm
from blab.mesh_topology import analyze_exterior_mesh_topology
from blab.physical_model import (
    AcousticRegionKind,
    ComponentKind,
    ExcitationPortKind,
    PhysicalSolveKind,
    infer_physical_solve_kind,
)
from blab.solve_results import (
    SolvedSystemBuilder,
    SolveProvenance,
    legacy_result_domains,
    legacy_result_to_system_result,
)
from blab.solvers.registry import supports_physical_system_solves
from blab.speaker_package import (
    SpeakerPackageConfig,
    SpeakerPackageFidelity,
    export_speaker_package,
    prepare_speaker_package_solve,
)
from blab.speaker_symmetry import expand_speaker_system_for_export
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
)
from blab.ui.physical_system_migration import (
    PhysicalSystemMigrationError,
    seed_exterior_system_from_solver_inputs,
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
from blab.ui.system_solve import (
    prepare_system_ui_solve,
    with_exterior_compatibility,
)


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
        self._pending_speaker_package: SpeakerPackageConfig | None = None
        self._pending_speaker_package_temp_dir: tempfile.TemporaryDirectory[str] | None = None

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
        self._view.set_max_spl_available(False)
        self._view.set_max_spl_export_available(False)
        self._plots.refresh_contour_controls()
        self._view.show_status(status)

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
        try:
            self._inputs.ensure_seeded_exterior_system(required=True)
        except PhysicalSystemMigrationError as exc:
            self._view.warn("Physical system migration", str(exc))
            return
        project = self._project()
        system = project.physical_system
        if system is None:
            self._view.warn(
                "Physical system migration",
                "This project has no physical system and could not be migrated automatically.",
            )
            return
        if not system.excitation_ports:
            self._view.warn(
                "No excitation ports",
                "Open System and add an excitation port to a physical component.",
            )
            return
        try:
            solve_kind = infer_physical_solve_kind(system)
        except ValueError as exc:
            self._view.warn("System solve", str(exc))
            return
        if solve_kind == PhysicalSolveKind.EXTERIOR_BEM:
            self._start_exterior_system_solve()
        else:
            self._start_coupled_system_solve()

    def start_speaker_package_solve(self, config: SpeakerPackageConfig) -> bool:
        """Prepare the requested package outputs, run once, then export on completion."""

        if self._geometry_controller.active or self._solve_controller.active:
            return False
        if not self._inputs.has_solver_meshes():
            self._view.warn(
                "No mesh", "Enable at least one generated or imported mesh before exporting a speaker package."
            )
            return False
        try:
            normalized = config.normalized()
        except ValueError as exc:
            self._view.warn("Speaker package", str(exc))
            return False
        self._inputs.ensure_seeded_exterior_system()
        project = self._project()
        if project.physical_system is None:
            self._view.warn(
                "Speaker package",
                "Open System and configure a physical system before exporting a speaker package.",
            )
            return False
        preferences = self._read_preferences()
        temporary: tempfile.TemporaryDirectory[str] | None = None
        try:
            meshes = inspect_system_meshes(self._inputs.mesh_entries_for_symmetry(project.symmetry))
            system = sync_physical_system_meshes(project.physical_system, meshes)
            project.physical_system = system
            solve_symmetry = project.symmetry
            component_channels = project.component_channel_by_id
            if normalized.fidelity >= SpeakerPackageFidelity.COUPLED and project.symmetry != "off":
                temporary = tempfile.TemporaryDirectory(prefix="blab-speaker-full-")
                preferred_full_meshes = {
                    entry.name: entry.source_file
                    for entry in self._inputs.mesh_entries_for_symmetry("off")
                    if entry.locked
                }
                expanded = expand_speaker_system_for_export(
                    system,
                    symmetry=project.symmetry,
                    output_dir=temporary.name,
                    preferred_full_mesh_by_name=preferred_full_meshes,
                )
                system = expanded.system
                solve_symmetry = "off"
                component_channels = {
                    component_id: project.component_channel_by_id.get(source_id, "main")
                    for component_id, source_id in expanded.component_source_ids.items()
                }
            frequencies = self._view.frequency_range()
            prepared = prepare_system_ui_solve(
                system,
                freq_min_hz=float(frequencies.min_hz),
                freq_max_hz=float(frequencies.max_hz),
                freq_count=frequencies.count,
                observation_distance_m=preferences.polar_observation_distance_m,
                polar_angle_step_deg=preferences.polar_angle_step_deg,
                spherical_sampling_enabled=False,
                spherical_sampling_points=0,
                component_channel_by_id=component_channels,
                backend_id=preferences.solve_backend,
                symmetry_mode=solve_symmetry,
                observation_planes=(),
            )
            prepared = prepare_speaker_package_solve(
                prepared,
                fidelity=normalized.fidelity,
                coupled_representation=normalized.coupled_representation,
                sphere_point_count=balloon_sampling_points(preferences.balloon_angle_precision_deg),
                sphere_radius_m=preferences.polar_observation_distance_m,
            )
        except Exception as exc:
            if temporary is not None:
                temporary.cleanup()
            self._view.show_stitch_or_generic_error("Speaker package preparation failed", exc)
            return False

        self._pending_speaker_package = normalized
        self._pending_speaker_package_temp_dir = temporary
        if not self._start_prepared_system_solve(prepared, "Initializing speaker package solve..."):
            self._pending_speaker_package = None
            self._pending_speaker_package_temp_dir = None
            if temporary is not None:
                temporary.cleanup()
            return False
        return True

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
            compatibility_required = not supports_physical_system_solves(preferences.solve_backend)
            solver_system = system
            component_channels = project.component_channel_by_id
            prepared_simulation = None
            if compatibility_required or project.stitch_imported_meshes:
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
                if project.stitch_imported_meshes:
                    solver_system, component_channels = seed_exterior_system_from_solver_inputs(
                        mesh_configs,
                        radiators,
                    )

            frequencies = self._view.frequency_range()
            prepared = prepare_system_ui_solve(
                solver_system,
                freq_min_hz=float(frequencies.min_hz),
                freq_max_hz=float(frequencies.max_hz),
                freq_count=frequencies.count,
                observation_distance_m=preferences.polar_observation_distance_m,
                polar_angle_step_deg=preferences.polar_angle_step_deg,
                spherical_sampling_enabled=preferences.spherical_sampling_enabled,
                spherical_sampling_points=balloon_sampling_points(preferences.balloon_angle_precision_deg),
                component_channel_by_id=component_channels,
                backend_id=preferences.solve_backend,
                symmetry_mode=symmetry,
                observation_planes=() if compatibility_required else project.observation_planes,
                allow_exterior_compatibility=compatibility_required,
            )
            if compatibility_required:
                assert prepared_simulation is not None
                prepared = with_exterior_compatibility(
                    prepared,
                    config=prepared_simulation.config,
                    server_url=preferences.solve_server_url,
                    server_access_token=load_server_access_token(preferences.solve_server_url),
                )
        except (ValueError, OSError, SymmetryValidationError) as exc:
            self._view.show_stitch_or_generic_error("Exterior system preparation failed", exc)
            return
        self._start_prepared_system_solve(prepared, "Initializing exterior solver...")

    def _start_coupled_system_solve(self) -> None:
        project = self._project()
        preferences = self._read_preferences()
        try:
            meshes = inspect_system_meshes(self._inputs.mesh_entries_for_symmetry(project.symmetry))
            system = sync_physical_system_meshes(project.physical_system, meshes)
            project.physical_system = system
            coupled_frequencies = self._view.frequency_range()
            prepared = prepare_system_ui_solve(
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
            self._view.warn("FEM system solve", str(exc))
            return

        status = (
            "Initializing interior FEM solver..."
            if prepared.solve_kind == PhysicalSolveKind.INTERIOR_FEM
            else "Initializing coupled solver..."
        )
        self._start_prepared_system_solve(prepared, status)

    def _start_prepared_system_solve(self, prepared, status: str) -> bool:
        if prepared.solve_kind == PhysicalSolveKind.EXTERIOR_BEM:
            if not self._confirm_exterior_mesh_topology(
                self._prepared_exterior_mesh_configs(prepared),
                symmetry=str(prepared.request.solver_options.get("symmetry", "off")),
            ):
                return False
        compiled_system = prepared.request.compiled_system
        impedance_normalization = normalization_records(getattr(compiled_system, "metadata", {}))
        mismatched = [
            record
            for record in impedance_normalization.values()
            if record.relative_side_mismatch is not None
            and record.relative_side_mismatch > ACOUSTIC_AREA_MISMATCH_WARNING_THRESHOLD
        ]
        if mismatched:
            details = "\n".join(
                f"• {record.component_name}: {record.positive_side_area_m2 * 10_000.0:.2f} cm² versus "
                f"{record.negative_side_area_m2 * 10_000.0:.2f} cm² "
                f"({record.relative_side_mismatch:.1%})"
                for record in mismatched
            )
            self._view.warn(
                "Diaphragm area mismatch",
                "Front and rear driven areas differ by more than 10%. "
                "Normalized acoustic impedance will use their average:\n\n" + details,
            )
        self._begin_run(status)
        regions = tuple(getattr(compiled_system, "regions", ()))
        reference_region = regions[0] if regions else None
        self._session.acoustic_impedance_density_kg_per_m3 = float(
            getattr(reference_region, "density_kg_per_m3", 1.21)
        )
        self._session.acoustic_impedance_sound_speed_m_per_s = float(
            getattr(reference_region, "sound_speed_m_per_s", 343.0)
        )
        channel_names = [str(value) for value in prepared.excitation_channel_names.tolist()]
        ports_by_id = {port.id: port for port in prepared.request.compiled_system.excitation_ports}
        excitation_ports = [ports_by_id[port_id] for port_id in prepared.request.excitation_port_ids]
        excitation_component_ids = [port.component_id for port in excitation_ports]
        if prepared.solve_kind == PhysicalSolveKind.EXTERIOR_BEM and all(
            component_id in impedance_normalization for component_id in excitation_component_ids
        ):
            self._session.acoustic_impedance_effective_areas_m2 = tuple(
                impedance_normalization[component_id].effective_area_m2
                for component_id in excitation_component_ids
            )
        voltage_channels = {
            channel_name
            for channel_name, port in zip(
                channel_names,
                excitation_ports,
                strict=True,
            )
            if port.kind == ExcitationPortKind.VOLTAGE
        }
        prescribed_velocity_channels = {
            channel_name
            for channel_name, port in zip(
                channel_names,
                excitation_ports,
                strict=True,
            )
            if port.kind == ExcitationPortKind.NORMAL_VELOCITY
        }
        self._session.voltage_channel_names = frozenset(voltage_channels - prescribed_velocity_channels)
        transducers = [
            component
            for component in prepared.request.compiled_system.components
            if component.kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER
        ]
        transducer_names = np.asarray([component.name for component in transducers])
        if transducer_names.size:
            channel_by_component_id = {
                port.component_id: channel_name
                for channel_name, port in zip(channel_names, excitation_ports, strict=True)
            }
            self._session.transducer_motion = TransducerMotionDataset(
                excitation_channel_names=np.asarray(prepared.excitation_channel_names).copy(),
                transducer_names=transducer_names,
                transducer_channel_names=np.asarray(
                    [channel_by_component_id.get(component.id, "main") for component in transducers]
                ),
                transducer_resistance_ohm=np.asarray(
                    [transducer_rated_resistance_ohm(component.parameters) for component in transducers],
                    dtype=np.float64,
                ),
            )
            if prepared.solve_kind != PhysicalSolveKind.INTERIOR_FEM:
                voltage_channel_names = np.asarray(
                    list(dict.fromkeys(name for name in channel_names if name in self._session.voltage_channel_names))
                )
                self._session.electrical_impedance = ElectricalImpedanceDataset(
                    excitation_port_ids=tuple(prepared.request.excitation_port_ids),
                    excitation_channel_names=np.asarray(prepared.excitation_channel_names).copy(),
                    excitation_component_ids=np.asarray([port.component_id for port in excitation_ports]),
                    transducer_component_ids=np.asarray([component.id for component in transducers]),
                    physical_driver_orbit_counts=np.asarray(
                        [int(component.parameters.get("physical_driver_orbit_count", 1)) for component in transducers],
                        dtype=np.int64,
                    ),
                    channel_names=voltage_channel_names,
                )
                if prepared.solve_kind == PhysicalSolveKind.COUPLED_BEM_FEM:
                    effective_areas = [
                        impedance_normalization[component.id].effective_area_m2
                        for component in transducers
                        if component.id in impedance_normalization
                    ]
                    self._session.acoustic_load_impedance = AcousticLoadImpedanceDataset(
                        excitation_port_ids=tuple(prepared.request.excitation_port_ids),
                        excitation_port_kinds=np.asarray([port.kind.value for port in excitation_ports]),
                        excitation_component_ids=np.asarray([port.component_id for port in excitation_ports]),
                        transducer_component_ids=np.asarray([component.id for component in transducers]),
                        transducer_names=np.asarray([component.name for component in transducers]),
                        bl_n_per_a=np.asarray(
                            [component.parameters["bl_n_per_a"] for component in transducers],
                            dtype=np.float64,
                        ),
                        mmd_kg=np.asarray(
                            [component.parameters["mmd_kg"] for component in transducers],
                            dtype=np.float64,
                        ),
                        cms_m_per_n=np.asarray(
                            [component.parameters["cms_m_per_n"] for component in transducers],
                            dtype=np.float64,
                        ),
                        rms_n_s_per_m=np.asarray(
                            [component.parameters["rms_n_s_per_m"] for component in transducers],
                            dtype=np.float64,
                        ),
                        effective_area_m2=(
                            np.asarray(effective_areas, dtype=np.float64)
                            if len(effective_areas) == len(transducers)
                            else None
                        ),
                        density_kg_per_m3=self._session.acoustic_impedance_density_kg_per_m3,
                        sound_speed_m_per_s=self._session.acoustic_impedance_sound_speed_m_per_s,
                    )
        self._session.result_builder = SolvedSystemBuilder(
            frequencies_hz=prepared.request.frequencies_hz,
            excitation_ids=prepared.request.excitation_port_ids,
            provenance=SolveProvenance(
                backend_id=prepared.backend_id,
                solve_kind=prepared.solve_kind.value,
                solver_options=dict(prepared.request.solver_options),
            ),
            domains=prepared.result_domains,
            compiled_system=prepared.request.compiled_system,
        )
        self._solve_controller.start(prepared)
        return True

    def _confirm_exterior_mesh_topology(
        self,
        mesh_configs: tuple[MeshConfig, ...],
        *,
        symmetry: str,
    ) -> bool:
        try:
            report = analyze_exterior_mesh_topology(mesh_configs, symmetry=symmetry)
        except (OSError, ValueError) as exc:
            self._view.warn("Mesh topology validation failed", str(exc))
            return False
        self._view.show_mesh_topology_issues(report)
        if not report.has_warnings:
            return True
        return self._view.confirm_mesh_topology_warning(report)

    @staticmethod
    def _prepared_exterior_mesh_configs(prepared) -> tuple[MeshConfig, ...]:
        compiled = prepared.request.compiled_system
        exterior = next(region for region in compiled.regions if region.kind == AcousticRegionKind.UNBOUNDED_AIR)
        meshes_by_id = {mesh.id: mesh for mesh in compiled.meshes}
        return tuple(
            MeshConfig(
                name=meshes_by_id[mesh_id].name,
                file=meshes_by_id[mesh_id].file,
                scale_factor=meshes_by_id[mesh_id].scale_to_m,
                translation_m=meshes_by_id[mesh_id].translation_m,
            )
            for mesh_id in exterior.mesh_ids
        )

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
        system = self._project().physical_system
        exterior_sound_speed = next(
            (
                region.sound_speed_m_per_s
                for region in (() if system is None else system.regions)
                if region.kind == AcousticRegionKind.UNBOUNDED_AIR
            ),
            343.0,
        )
        self._session.live_dataset = LiveSolveDataset(
            polar_angle_deg=np.asarray(angles, dtype=np.float32),
            radiator_names=np.asarray(radiator_names),
            channel_configs=self._inputs.channel_configs(),
            flat_target_normalization_enabled=preferences.normalized_channel_correction,
            flat_target_reference_angle_deg=preferences.horizontal_normalization_angle,
            polar_observation_distance_m=preferences.polar_observation_distance_m,
            exterior_sound_speed_m_per_s=exterior_sound_speed,
            acoustic_impedance_effective_area_m2=(
                np.asarray(self._session.acoustic_impedance_effective_areas_m2, dtype=np.float64)
                if self._session.acoustic_impedance_effective_areas_m2 is not None
                and len(self._session.acoustic_impedance_effective_areas_m2) == len(radiator_names)
                else None
            ),
            acoustic_impedance_density_kg_per_m3=self._session.acoustic_impedance_density_kg_per_m3,
            sphere_r_distance_m=sphere_metadata.get("r_distance_m"),
            sphere_theta_polar_rad=sphere_metadata.get("theta_polar_rad"),
            sphere_phi_azimuth_rad=sphere_metadata.get("phi_azimuth_rad"),
            voltage_channel_names=self._session.voltage_channel_names,
        )
        self._view.show_status("Solving...")

    @Slot(object)
    def _on_frequency_result(self, result: FrequencyResult) -> None:
        live_dataset = self._session.live_dataset
        if live_dataset is None:
            return
        live_dataset.add(result)
        self._plots.set_spherical_spin_available(live_dataset.has_balloon_data)
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
        if (
            self._session.result_builder is not None
            and self._session.result_builder.provenance.solve_kind == "interior_fem"
        ):
            return
        self._plots.request_live_refresh()

    @Slot(object)
    def _on_system_frequency_result(self, result: SystemFrequencyResult) -> None:
        builder = self._session.result_builder
        if builder is None:
            raise RuntimeError("Received a physical-system result before its result builder was initialized.")
        builder.add(result)
        motion = self._session.transducer_motion
        if motion is not None:
            motion.add(result)
        electrical_impedance = self._session.electrical_impedance
        if electrical_impedance is not None:
            electrical_impedance.add(result)
        acoustic_load_impedance = self._session.acoustic_load_impedance
        if acoustic_load_impedance is not None:
            acoustic_load_impedance.add(result)

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
        pending_package = self._pending_speaker_package
        self._pending_speaker_package = None
        pending_package_temp_dir = self._pending_speaker_package_temp_dir
        self._pending_speaker_package_temp_dir = None
        package_result = None
        try:
            if pending_package is not None and completion.completed:
                solved = session.solved_system
                if solved is not None and solved.complete:
                    try:
                        package_result = export_speaker_package(solved, pending_package)
                    except Exception as exc:
                        self._view.show_error("Speaker package export failed", str(exc))
        finally:
            if pending_package_temp_dir is not None:
                pending_package_temp_dir.cleanup()
        observation_planes = getattr(self._view, "observation_plane_controller", None)
        if observation_planes is not None:
            observation_planes.sync_view()
        elapsed_s = completion.elapsed_s
        if session.has_solved_data():
            solved_count = session.solved_count
            solve_completed = completion.completed
            interior_fem = (
                session.solved_system is not None and session.solved_system.provenance.solve_kind == "interior_fem"
            )
            eligible_max_spl_channels = (
                ()
                if session.transducer_motion is None
                else session.transducer_motion.eligible_max_spl_channel_names(session.voltage_channel_names)
            )
            configured_max_spl_limits = max_spl_limits_from_payload(self._project().max_spl_limits_by_channel)
            session.max_spl_requested = (
                solve_completed
                and not interior_fem
                and any(
                    configured_max_spl_limits.get(name) is not None and configured_max_spl_limits[name].enabled
                    for name in eligible_max_spl_channels
                )
            )
            session.use_final_isobar_resolution = solve_completed
            if solve_completed and not interior_fem:
                self._view.show_status("Rendering final high-resolution plots...")
            refreshed_dataset = None
            if self._read_preferences().live_plot_streaming or solve_completed:
                if not interior_fem:
                    refreshed_dataset = self._plots.refresh_plots()
            if solve_completed and not interior_fem:
                if refreshed_dataset is None:
                    refreshed_dataset = self._plots.prepared_live_dataset(
                        angle_samples=FINAL_ISOBAR_ANGLE_SAMPLES,
                        freq_samples=FINAL_ISOBAR_FREQ_SAMPLES,
                    )
                if refreshed_dataset is not None:
                    session.last_completed_visualization = refreshed_dataset.snapshot()
            session.final_isobar_plots_rendered = (
                solve_completed and not interior_fem and bool(self._plots.visible_isobar_plots())
            )
            self._view.set_plot_exports_available(not interior_fem)
            self._view.set_balloon_plot_available(not interior_fem and session.live_dataset.has_balloon_data)
            self._view.set_max_spl_available(solve_completed and not interior_fem and bool(eligible_max_spl_channels))
            self._view.set_max_spl_export_available(
                solve_completed
                and not interior_fem
                and refreshed_dataset is not None
                and refreshed_dataset.max_spl is not None
            )
            self._plots.refresh_contour_controls()
            elapsed_text = f" in {elapsed_s:.1f} s"
            if completion.phase == OperationPhase.CANCELLED:
                self._view.show_status(f"Solve stopped: {session.solved_count} frequencies{elapsed_text}")
                return
            if completion.phase == OperationPhase.FAILED:
                self._view.show_status(f"Solve failed after {solved_count} frequencies{elapsed_text}")
                return
            if package_result is not None:
                self._view.show_status(f"Exported speaker package to {package_result.path}")
            else:
                self._view.show_status(f"Solve complete: {solved_count} frequencies{elapsed_text}")
        elif completion.phase == OperationPhase.CANCELLED:
            self._view.show_status("Solve stopped")
        else:
            self._view.set_max_spl_available(False)
            self._view.set_max_spl_export_available(False)
        self._plots.refresh_contour_controls()
