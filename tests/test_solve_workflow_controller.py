"""SolveWorkflowController driven with no MainWindow at all.

This is the payoff of the seam work: the solve workflow depends on WorkflowView,
PlotPresenter and SolveInputs, so it can be exercised against plain recording
fakes. A UI revamp replaces those three implementations and this suite still
describes the behaviour that must survive.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import blab.ui.main_window.solve_workflow as solve_workflow_module  # noqa: E402
from blab.config import MeshConfig  # noqa: E402
from blab.physical_model import (  # noqa: E402
    AcousticRegionKind,
    ComponentKind,
    ExcitationPortKind,
    PhysicalSolveKind,
)
from blab.ui.application_state import OperationPhase, SolveCompletion  # noqa: E402
from blab.ui.main_window.solve_session import SolveSession  # noqa: E402
from blab.ui.main_window.solve_workflow import SolveWorkflowController  # noqa: E402
from blab.ui.main_window.workflow_view import FrequencyRange  # noqa: E402
from blab.ui.physical_system_migration import PhysicalSystemMigrationError  # noqa: E402
from blab.ui.project_state import new_project_document  # noqa: E402
from blab.ui.settings import GuiPreferences  # noqa: E402


class FakeView:
    """Records what the workflow asked the UI to do."""

    def __init__(self) -> None:
        self.status: list[str] = []
        self.warnings: list[tuple[str, str]] = []
        self.errors: list[tuple[str, str]] = []
        self.stitch_errors: list[tuple[str, Exception]] = []
        self.phases: list[tuple[OperationPhase, bool]] = []
        self.plot_exports: list[bool] = []
        self.balloon: list[bool] = []
        self.max_spl: list[bool] = []
        self.max_spl_exports: list[bool] = []
        self.topology_reports: list[object] = []
        self.topology_confirmation = False

    def show_status(self, message):
        self.status.append(message)

    def warn(self, title, message):
        self.warnings.append((title, message))

    def show_error(self, title, message):
        self.errors.append((title, message))

    def show_stitch_or_generic_error(self, title, exc):
        self.stitch_errors.append((title, exc))

    def show_mesh_topology_issues(self, report):
        self.topology_reports.append(report)

    def confirm_mesh_topology_warning(self, _report):
        return self.topology_confirmation

    def set_workflow_phase(self, phase, *, cancel_available=True):
        self.phases.append((phase, cancel_available))

    def frequency_range(self):
        return FrequencyRange(min_hz=200, max_hz=20_000, count=41)

    def set_plot_exports_available(self, available):
        self.plot_exports.append(available)

    def set_balloon_plot_available(self, available):
        self.balloon.append(available)

    def set_max_spl_available(self, available):
        self.max_spl.append(available)

    def set_max_spl_export_available(self, available):
        self.max_spl_exports.append(available)


class FakePlots:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name):
        def record(*_args, **_kwargs):
            self.calls.append(name)
            return None

        return record


class FakeInputs:
    def __init__(self, *, has_meshes=True, radiators=()) -> None:
        self._has_meshes = has_meshes
        self._radiators = radiators
        self.seeded = 0
        self.migration_error: PhysicalSystemMigrationError | None = None

    def has_solver_meshes(self):
        return self._has_meshes

    def ensure_seeded_exterior_system(self, *, required=False):
        self.seeded += 1
        if required and self.migration_error is not None:
            raise self.migration_error
        return False

    def all_radiators(self):
        return self._radiators

    def reconcile_symmetry_with_backend(self):
        return False

    def channel_configs(self):
        return ()


class FakeOperation:
    def __init__(self, *, active=False) -> None:
        self.active = active
        self.started: list[object] = []
        self.cancelled = 0

    def start(self, request):
        self.started.append(request)
        return True

    def cancel(self):
        self.cancelled += 1


@pytest.fixture
def controller(qapp):
    """A controller built entirely from fakes — no window, no widgets."""
    view, plots, inputs = FakeView(), FakePlots(), FakeInputs()
    session = SolveSession()
    geometry, solve = FakeOperation(), FakeOperation()
    project = new_project_document()
    built = SolveWorkflowController(
        None,
        view=view,
        plots=plots,
        inputs=inputs,
        session=session,
        project=lambda: project,
        preferences=lambda: GuiPreferences(),
        assembler=None,
        geometry_controller=geometry,
        solve_controller=solve,
    )
    built.view, built.plots, built.inputs = view, plots, inputs
    built.session, built.geometry, built.solve = session, geometry, solve
    built.project = project
    return built


def test_the_controller_needs_no_main_window(controller) -> None:
    """The whole point: constructed from protocols, not from a QMainWindow."""
    assert isinstance(controller, SolveWorkflowController)


def test_beginning_a_solve_withdraws_every_export_entry_point(controller) -> None:
    controller._begin_run("Initializing solver...")

    assert controller.view.plot_exports == [False]
    assert controller.view.balloon == [False]
    assert controller.view.max_spl == [False]
    assert controller.view.max_spl_exports == [False]


@pytest.mark.parametrize(
    ("solve_kind", "expects_electrical_impedance", "expects_acoustic_load"),
    [
        (PhysicalSolveKind.COUPLED_BEM_FEM, True, True),
        (PhysicalSolveKind.INTERIOR_FEM, False, False),
    ],
)
def test_electrical_impedance_live_cache_remains_disabled_for_interior_fem(
    controller,
    solve_kind,
    expects_electrical_impedance,
    expects_acoustic_load,
) -> None:
    component = SimpleNamespace(
        id="component:woofer",
        name="Woofer",
        kind=ComponentKind.ELECTRODYNAMIC_TRANSDUCER,
        parameters={
            "re_ohm": 6.0,
            "physical_driver_orbit_count": 2,
            "bl_n_per_a": 7.0,
            "mmd_kg": 0.015,
            "cms_m_per_n": 5.0e-4,
            "rms_n_s_per_m": 1.0,
        },
    )
    port = SimpleNamespace(
        id="port:woofer",
        component_id=component.id,
        kind=ExcitationPortKind.VOLTAGE,
    )
    prepared = SimpleNamespace(
        excitation_channel_names=np.asarray(["main"]),
        request=SimpleNamespace(
            compiled_system=SimpleNamespace(
                excitation_ports=(port,),
                components=(component,),
            ),
            excitation_port_ids=(port.id,),
            frequencies_hz=(100.0,),
            solver_options={},
        ),
        backend_id="beat_cpu",
        solve_kind=solve_kind,
        result_domains=(),
    )

    controller._start_prepared_system_solve(prepared, "Starting")

    assert (controller.session.electrical_impedance is not None) is expects_electrical_impedance
    assert (controller.session.acoustic_load_impedance is not None) is expects_acoustic_load
    if expects_electrical_impedance:
        assert controller.session.electrical_impedance.channel_names.tolist() == ["main"]
        assert controller.session.electrical_impedance.physical_driver_orbit_counts.tolist() == [2]
    if expects_acoustic_load:
        assert controller.session.acoustic_load_impedance.transducer_names.tolist() == ["Woofer"]
        assert controller.session.acoustic_load_impedance.bl_n_per_a.tolist() == [7.0]


def test_solve_warns_when_opposing_diaphragm_areas_differ_by_more_than_ten_percent(
    controller,
) -> None:
    component = SimpleNamespace(
        id="component:woofer",
        name="Woofer",
        kind=ComponentKind.ELECTRODYNAMIC_TRANSDUCER,
        parameters={
            "re_ohm": 6.0,
            "physical_driver_orbit_count": 1,
            "bl_n_per_a": 7.0,
            "mmd_kg": 0.015,
            "cms_m_per_n": 5.0e-4,
            "rms_n_s_per_m": 1.0,
        },
    )
    port = SimpleNamespace(
        id="port:woofer",
        component_id=component.id,
        kind=ExcitationPortKind.VOLTAGE,
    )
    prepared = SimpleNamespace(
        excitation_channel_names=np.asarray(["main"]),
        request=SimpleNamespace(
            compiled_system=SimpleNamespace(
                excitation_ports=(port,),
                components=(component,),
                regions=(SimpleNamespace(density_kg_per_m3=1.2, sound_speed_m_per_s=340.0),),
                metadata={
                    "acoustic_impedance_normalization": {
                        component.id: {
                            "component_name": component.name,
                            "effective_area_m2": 0.009,
                            "area_kind": "projected_rigid_translation",
                            "positive_side_area_m2": 0.010,
                            "negative_side_area_m2": 0.008,
                            "relative_side_mismatch": 0.2,
                        },
                        "component:exact-threshold": {
                            "component_name": "At threshold",
                            "effective_area_m2": 0.0095,
                            "area_kind": "projected_rigid_translation",
                            "positive_side_area_m2": 0.010,
                            "negative_side_area_m2": 0.009,
                            "relative_side_mismatch": 0.1,
                        },
                    }
                },
            ),
            excitation_port_ids=(port.id,),
            frequencies_hz=(100.0,),
            solver_options={},
        ),
        backend_id="beat_cpu",
        solve_kind=PhysicalSolveKind.INTERIOR_FEM,
        result_domains=(),
    )

    controller._start_prepared_system_solve(prepared, "Starting")

    assert controller.view.warnings == [
        (
            "Diaphragm area mismatch",
            "Front and rear driven areas differ by more than 10%. "
            "Normalized acoustic impedance will use their average:\n\n"
            "• Woofer: 100.00 cm² versus 80.00 cm² (20.0%)",
        )
    ]
    assert controller.session.acoustic_impedance_effective_areas_m2 is None
    assert controller.solve.started == [prepared]


def test_live_impedance_areas_follow_solver_radiator_order_even_with_duplicate_names(
    controller,
) -> None:
    controller.session.acoustic_impedance_effective_areas_m2 = (0.01, 0.02)
    controller.session.acoustic_impedance_density_kg_per_m3 = 1.2

    controller._on_solver_initialized(
        np.asarray((-90.0, 0.0, 90.0)),
        np.asarray(("Radiator", "Radiator")),
        None,
    )

    dataset = controller.session.live_dataset
    assert dataset is not None
    np.testing.assert_allclose(dataset.acoustic_impedance_effective_area_m2, [0.01, 0.02])
    assert dataset.acoustic_impedance_density_kg_per_m3 == pytest.approx(1.2)


def test_solving_without_a_mesh_warns_instead_of_starting(controller) -> None:
    controller.inputs._has_meshes = False

    controller.start_solve()

    assert controller.view.warnings == [("No mesh", "Enable at least one generated or imported mesh before solving.")]
    assert controller.solve.started == []


def test_a_busy_geometry_run_blocks_a_second_start(controller) -> None:
    controller.geometry.active = True

    controller.start_solve()

    assert controller.solve.started == []
    assert controller.view.warnings == [], "a busy run is not a user error"


def test_solving_stops_when_legacy_project_migration_produces_no_system(controller) -> None:
    controller.start_solve()

    assert controller.view.warnings == [
        (
            "Physical system migration",
            "This project has no physical system and could not be migrated automatically.",
        )
    ]
    assert controller.inputs.seeded == 1
    assert controller.solve.started == []


def test_solving_reports_required_physical_system_migration_failure(controller) -> None:
    controller.inputs.migration_error = PhysicalSystemMigrationError("mesh groups are ambiguous")

    controller.start_solve()

    assert controller.view.warnings == [("Physical system migration", "mesh groups are ambiguous")]
    assert controller.solve.started == []


def test_physical_system_with_no_excitation_explains_what_to_add(controller) -> None:
    controller.project.physical_system = SimpleNamespace(excitation_ports=())

    controller.start_solve()

    assert controller.view.warnings == [
        (
            "No excitation ports",
            "Open System and add an excitation port to a physical component.",
        )
    ]


def test_exterior_solve_dispatch_requires_a_physical_system(controller, monkeypatch) -> None:
    controller.project.physical_system = SimpleNamespace(
        regions=(SimpleNamespace(kind=AcousticRegionKind.UNBOUNDED_AIR),),
        interfaces=(),
        excitation_ports=(SimpleNamespace(id="excitation:source"),),
    )
    dispatched = []
    monkeypatch.setattr(controller, "_start_exterior_system_solve", lambda: dispatched.append("exterior"))

    controller.start_solve()

    assert dispatched == ["exterior"]
    assert controller.inputs.seeded == 1


def test_local_exterior_backend_is_attached_to_a_system_request(controller, monkeypatch) -> None:
    system = SimpleNamespace()
    controller.project.physical_system = system
    controller.project.stitch_imported_meshes = False
    controller.inputs.mesh_entries_for_symmetry = lambda _symmetry: ("mesh-entry",)
    controller.inputs.mesh_service = lambda: SimpleNamespace(
        prepare_mesh_configs=lambda *_args, **_kwargs: (("mesh-config",), ("radiator",))
    )
    controller.inputs.solver_channel_configs = lambda _radiators: ("channel",)
    prepared_simulation = SimpleNamespace(config=SimpleNamespace(name="legacy-config"))
    controller._assembler = SimpleNamespace(prepare=lambda **_kwargs: prepared_simulation)
    monkeypatch.setattr(solve_workflow_module, "inspect_system_meshes", lambda _entries: ("mesh",))
    monkeypatch.setattr(solve_workflow_module, "sync_physical_system_meshes", lambda value, _meshes: value)
    monkeypatch.setattr(
        solve_workflow_module,
        "exterior_bem_inputs",
        lambda *_args, **_kwargs: SimpleNamespace(mesh_configs=("raw-mesh",), radiators=("raw-radiator",)),
    )
    prepared_calls = []
    monkeypatch.setattr(
        solve_workflow_module,
        "prepare_system_ui_solve",
        lambda value, **kwargs: prepared_calls.append((value, kwargs)) or "canonical",
    )
    compatibility_calls = []
    monkeypatch.setattr(
        solve_workflow_module,
        "with_exterior_compatibility",
        lambda prepared, **kwargs: compatibility_calls.append((prepared, kwargs)) or "adapted",
    )
    monkeypatch.setattr(solve_workflow_module, "load_server_access_token", lambda _url: "token")
    dispatched = []
    monkeypatch.setattr(
        controller,
        "_start_prepared_system_solve",
        lambda prepared, status: dispatched.append((prepared, status)),
    )

    controller._start_exterior_system_solve()

    assert prepared_calls[0][0] is system
    assert prepared_calls[0][1]["backend_id"] == "local"
    assert compatibility_calls[0][0] == "canonical"
    assert compatibility_calls[0][1]["config"] is prepared_simulation.config
    assert compatibility_calls[0][1]["server_access_token"] == "token"
    assert dispatched == [("adapted", "Initializing exterior solver...")]


def test_stitched_beat_exterior_is_represented_by_a_derived_physical_system(controller, monkeypatch) -> None:
    source_system = SimpleNamespace()
    derived_system = SimpleNamespace()
    controller.project.physical_system = source_system
    controller.project.stitch_imported_meshes = True
    controller._read_preferences = lambda: GuiPreferences(solve_backend="beat_cpu")
    controller.inputs.mesh_entries_for_symmetry = lambda _symmetry: ("mesh-entry",)
    controller.inputs.mesh_service = lambda: SimpleNamespace(
        prepare_mesh_configs=lambda *_args, **_kwargs: (("stitched-mesh",), ("stitched-radiator",))
    )
    controller.inputs.solver_channel_configs = lambda _radiators: ("channel",)
    controller._assembler = SimpleNamespace(prepare=lambda **_kwargs: SimpleNamespace(config="legacy-config"))
    monkeypatch.setattr(solve_workflow_module, "inspect_system_meshes", lambda _entries: ("mesh",))
    monkeypatch.setattr(solve_workflow_module, "sync_physical_system_meshes", lambda value, _meshes: value)
    monkeypatch.setattr(
        solve_workflow_module,
        "exterior_bem_inputs",
        lambda *_args, **_kwargs: SimpleNamespace(mesh_configs=("raw-mesh",), radiators=("raw-radiator",)),
    )
    monkeypatch.setattr(
        solve_workflow_module,
        "seed_exterior_system_from_solver_inputs",
        lambda meshes, radiators: (
            derived_system,
            {"component:stitched": "main"},
        ),
    )
    prepared_calls = []
    monkeypatch.setattr(
        solve_workflow_module,
        "prepare_system_ui_solve",
        lambda value, **kwargs: prepared_calls.append((value, kwargs)) or "canonical",
    )
    monkeypatch.setattr(
        solve_workflow_module,
        "with_exterior_compatibility",
        lambda *_args, **_kwargs: pytest.fail("BEAT must use the canonical physical-system backend"),
    )
    dispatched = []
    monkeypatch.setattr(
        controller,
        "_start_prepared_system_solve",
        lambda prepared, status: dispatched.append((prepared, status)),
    )

    controller._start_exterior_system_solve()

    assert prepared_calls[0][0] is derived_system
    assert prepared_calls[0][1]["component_channel_by_id"] == {"component:stitched": "main"}
    assert prepared_calls[0][1]["backend_id"] == "beat_cpu"
    assert dispatched == [("canonical", "Initializing exterior solver...")]


@pytest.mark.parametrize(("confirmed", "expected"), [(False, False), (True, True)])
def test_exterior_topology_warning_can_cancel_or_override(
    controller,
    monkeypatch,
    confirmed: bool,
    expected: bool,
) -> None:
    report = SimpleNamespace(has_warnings=True)
    monkeypatch.setattr(solve_workflow_module, "analyze_exterior_mesh_topology", lambda *_args, **_kwargs: report)
    controller.view.topology_confirmation = confirmed

    accepted = controller._confirm_exterior_mesh_topology(
        (MeshConfig("exterior", "unused.msh"),),
        symmetry="off",
    )

    assert accepted is expected
    assert controller.view.topology_reports == [report]


def test_clean_exterior_topology_does_not_show_confirmation(controller, monkeypatch) -> None:
    report = SimpleNamespace(has_warnings=False)
    monkeypatch.setattr(solve_workflow_module, "analyze_exterior_mesh_topology", lambda *_args, **_kwargs: report)

    accepted = controller._confirm_exterior_mesh_topology(
        (MeshConfig("exterior", "unused.msh"),),
        symmetry="off",
    )

    assert accepted is True
    assert controller.view.topology_reports == [report]


def test_cancelling_prefers_the_geometry_run_when_one_is_active(controller) -> None:
    controller.geometry.active = True

    controller.cancel_current_operation()

    assert controller.geometry.cancelled == 1
    assert controller.solve.cancelled == 0
    assert (OperationPhase.CANCELLING, True) in controller.view.phases


def test_cancelling_falls_through_to_the_solve(controller) -> None:
    controller.solve.active = True

    controller.cancel_current_operation()

    assert controller.solve.cancelled == 1
    assert controller.geometry.cancelled == 0


def test_cancelling_an_idle_workflow_does_nothing(controller) -> None:
    controller.cancel_current_operation()

    assert controller.solve.cancelled == 0
    assert controller.geometry.cancelled == 0


def test_a_failed_solve_is_reported_as_an_error_not_a_warning(controller) -> None:
    controller._on_solve_failed("backend exploded")

    assert controller.view.errors == [("Solve failed", "backend exploded")]
    assert controller.view.warnings == []


def test_finishing_with_no_results_restores_idle_without_offering_exports(controller) -> None:
    controller._on_solve_finished(
        SolveCompletion(phase=OperationPhase.COMPLETED, solved_count=0, expected_count=0, elapsed_s=1.0)
    )

    assert (OperationPhase.IDLE, True) in controller.view.phases
    assert controller.view.plot_exports == [], "nothing was solved, so nothing is exportable"


def test_a_cancelled_empty_solve_says_so(controller) -> None:
    controller._on_solve_finished(
        SolveCompletion(phase=OperationPhase.CANCELLED, solved_count=0, expected_count=41, elapsed_s=2.0)
    )

    assert "Solve stopped" in controller.view.status


def test_completed_solve_automatically_requests_configured_max_spl(controller) -> None:
    class CompletedSession:
        live_dataset = SimpleNamespace(
            supports_channel_resynthesis=True,
            has_balloon_data=False,
        )
        transducer_motion = SimpleNamespace(eligible_max_spl_channel_names=lambda _channels: ("main",))
        voltage_channel_names = frozenset({"main"})
        solved_system = SimpleNamespace(provenance=SimpleNamespace(solve_kind="exterior_bem"))
        max_spl_requested = False
        use_final_isobar_resolution = False
        final_isobar_plots_rendered = False
        last_completed_visualization = None
        solved_count = 1

        @staticmethod
        def finalize_results(*, status):
            assert status == "completed"

        @staticmethod
        def has_solved_data():
            return True

    project = new_project_document()
    project.max_spl_limits_by_channel = {"main": {"xmax_mm": 5.0, "pmax_w": 200.0}}
    session = CompletedSession()
    controller._session = session
    controller.session = session
    controller._project = lambda: project

    controller._on_solve_finished(
        SolveCompletion(phase=OperationPhase.COMPLETED, solved_count=1, expected_count=1, elapsed_s=1.0)
    )

    assert session.max_spl_requested is True
    assert controller.view.max_spl[-1] is True
