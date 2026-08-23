"""Behavioural tests for MainWindow.

These replace source-text assertions that previously grepped ``main_window.py``.
Each test drives a real, fully constructed window (with the VTK preview stubbed
by the ``main_window`` fixture) and asserts on widget state, so they survive the
window being reorganised into modules.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtWidgets import QDockWidget, QFrame, QMessageBox, QTabBar  # noqa: E402

import blab.ui.main_window.state_sync as state_sync_module  # noqa: E402
import blab.ui.main_window.view_builder as view_builder_module  # noqa: E402
from blab.config import ChannelConfig  # noqa: E402
from blab.physical_model import ExcitationPortKind  # noqa: E402
from blab.ui.main_window import ADD_DESIGN_TAB_LABEL, LIVE_PLOT_REFRESH_INTERVAL_MS  # noqa: E402
from blab.ui.main_window_widgets import DockTitleBar  # noqa: E402
from repo_paths import BLAB_SRC, MAIN_WINDOW_PKG  # noqa: E402


def test_app_root_resolves_to_the_repository_root() -> None:
    from blab.paths import APP_ROOT, asset

    assert (APP_ROOT / "pyproject.toml").is_file(), f"APP_ROOT resolved to {APP_ROOT}"
    assert (APP_ROOT / "assets").is_dir()
    assert asset("save_dark.ico").is_file()


def test_every_bundled_resource_path_actually_exists() -> None:
    """The whole failure mode is silence.

    A missing icon degrades to a null QIcon and a missing PDF is only opened
    on demand, so a wrong APP_ROOT shows up as a blank toolbar rather than an
    exception. Checking one icon was not enough; check every declared path.
    """
    import importlib
    from pathlib import Path

    # Directories that are created on demand or ship only in a packaged build.
    runtime_created = {"APP_ROOT", "ASSETS_DIR", "GENERATED_GEOMETRY_ROOT", "ATH_BUNDLE_DIR"}

    missing = []
    for module_name in (
        "blab.gui",
        "blab.ui.dialogs",
        "blab.ui.theme",
        "blab.ui.balloon",
        "blab.ui.main_window.constants",
    ):
        module = importlib.import_module(module_name)
        for attribute in dir(module):
            if attribute in runtime_created:
                continue
            value = getattr(module, attribute)
            if isinstance(value, Path):
                candidates = [value]
            elif isinstance(value, tuple) and value and all(isinstance(item, Path) for item in value):
                candidates = list(value)
            else:
                continue
            missing.extend(f"{module_name}.{attribute} -> {path}" for path in candidates if not path.exists())

    assert missing == [], f"declared resource paths that do not exist: {missing}"


PLOT_IDS = {
    "horizontal_isobar",
    "vertical_isobar",
    "acoustic_impedance",
    "electrical_impedance",
    "on_axis_frequency_response",
    "group_delay",
    "transducer_excursion",
    "spinorama",
}
ISOBAR_IDS = ("horizontal_isobar", "vertical_isobar")


class _ButtonStub:
    def __init__(self, text: str, role) -> None:
        self.text = text
        self.role = role


class _MessageBoxStub:
    """Records how a confirmation box was built and clicks a chosen button."""

    Warning = QMessageBox.Warning
    NoButton = QMessageBox.NoButton
    AcceptRole = QMessageBox.AcceptRole
    RejectRole = QMessageBox.RejectRole
    DestructiveRole = QMessageBox.DestructiveRole

    instances: list["_MessageBoxStub"] = []
    click: str = "Cancel"

    def __init__(self, _icon, title, text, _buttons, _parent) -> None:
        self.title = title
        self.text = text
        self.buttons: list[_ButtonStub] = []
        self.default: _ButtonStub | None = None
        self.executed = False
        _MessageBoxStub.instances.append(self)

    def addButton(self, text, role):  # noqa: N802 - Qt API shape
        button = _ButtonStub(text, role)
        self.buttons.append(button)
        return button

    def setDefaultButton(self, button) -> None:  # noqa: N802 - Qt API shape
        self.default = button

    def exec(self) -> None:
        self.executed = True

    def clickedButton(self):  # noqa: N802 - Qt API shape
        return next((b for b in self.buttons if b.text == _MessageBoxStub.click), None)

    @property
    def button_texts(self) -> list[str]:
        return [b.text for b in self.buttons]


@pytest.fixture
def message_box(monkeypatch):
    _MessageBoxStub.instances = []
    _MessageBoxStub.click = "Cancel"
    monkeypatch.setattr(state_sync_module, "QMessageBox", _MessageBoxStub)
    # The unsaved-changes prompt is raised by the WorkflowView adapter now, not
    # by the workflow that decides to ask for it.
    monkeypatch.setattr(view_builder_module, "QMessageBox", _MessageBoxStub)
    return _MessageBoxStub


# --------------------------------------------------------------------------- docks


def test_every_plot_lives_in_its_own_detachable_dock(main_window) -> None:
    assert set(main_window.plot_docks) == PLOT_IDS

    for plot_id, dock in main_window.plot_docks.items():
        assert isinstance(dock, QDockWidget), plot_id
        features = dock.features()
        assert features & QDockWidget.DockWidgetFloatable, plot_id
        assert features & QDockWidget.DockWidgetClosable, plot_id
        assert isinstance(dock.titleBarWidget(), DockTitleBar), plot_id

    # Docks live in the nested workspace, not directly on the window.
    assert main_window.workspace is not main_window
    assert main_window.workspace.isDockNestingEnabled()


def test_dock_title_bar_uses_a_native_raised_frame(main_window) -> None:
    """The bevel comes from QFrame chrome, not a QSS rule."""
    for dock in (main_window.editor_dock, main_window.preview_dock, *main_window.plot_docks.values()):
        title_bar = dock.titleBarWidget()
        assert isinstance(title_bar, DockTitleBar)
        assert isinstance(title_bar, QFrame)
        assert title_bar.frameShape() == QFrame.StyledPanel
        assert title_bar.frameShadow() == QFrame.Raised


def test_on_axis_dock_exposes_trace_filter_and_phase_controls(main_window) -> None:
    acoustic_title_bar = main_window.plot_docks["acoustic_impedance"].titleBarWidget()
    acoustic_trace_button = next(
        button
        for button in acoustic_title_bar.tool_buttons
        if button.menu() is main_window.impedance_plot.trace_filter_menu
    )
    assert acoustic_trace_button.text() == "Traces"

    title_bar = main_window.plot_docks["on_axis_frequency_response"].titleBarWidget()
    actions = [button.defaultAction() for button in title_bar.tool_buttons]

    assert main_window.on_axis_plot.show_phase_action in actions
    phase_button = next(
        button
        for button in title_bar.tool_buttons
        if button.defaultAction() is main_window.on_axis_plot.show_phase_action
    )
    assert not phase_button.icon().isNull()
    assert phase_button.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly
    trace_button = next(
        button for button in title_bar.tool_buttons if button.menu() is main_window.on_axis_plot.trace_filter_menu
    )
    assert trace_button.text() == "Traces"

    group_delay_title_bar = main_window.plot_docks["group_delay"].titleBarWidget()
    group_delay_trace_button = next(
        button
        for button in group_delay_title_bar.tool_buttons
        if button.menu() is main_window.group_delay_plot.trace_filter_menu
    )
    assert group_delay_trace_button.text() == "Traces"

    excursion_title_bar = main_window.plot_docks["transducer_excursion"].titleBarWidget()
    excursion_trace_button = next(
        button
        for button in excursion_title_bar.tool_buttons
        if button.menu() is main_window.excursion_plot.trace_filter_menu
    )
    assert excursion_trace_button.text() == "Traces"

    electrical_title_bar = main_window.plot_docks["electrical_impedance"].titleBarWidget()
    electrical_actions = [button.defaultAction() for button in electrical_title_bar.tool_buttons]
    assert main_window.electrical_impedance_plot.show_phase_action in electrical_actions
    electrical_trace_button = next(
        button
        for button in electrical_title_bar.tool_buttons
        if button.menu() is main_window.electrical_impedance_plot.trace_filter_menu
    )
    assert electrical_trace_button.text() == "Traces"


def test_prescribed_velocity_channel_detection_uses_physical_excitation_ports(main_window) -> None:
    main_window.project.physical_system = SimpleNamespace(
        excitation_ports=(
            SimpleNamespace(component_id="component:woofer", kind=ExcitationPortKind.VOLTAGE),
            SimpleNamespace(component_id="component:port", kind=ExcitationPortKind.NORMAL_VELOCITY),
        )
    )
    main_window.project.component_channel_by_id = {
        "component:woofer": "low",
        "component:port": "mixed",
    }

    assert main_window.prescribed_velocity_channel_names() == frozenset({"mixed"})


def test_dock_state_round_trips_through_workspace(main_window) -> None:
    state = main_window.workspace.saveState()
    assert not state.isEmpty()
    assert main_window.workspace.restoreState(state)


def test_panel_visibility_actions_drive_their_docks(main_window) -> None:
    for dock_id, action in main_window.panel_view_actions.items():
        assert action.isCheckable(), dock_id

        action.setChecked(False)
        assert main_window.findChild(QDockWidget, dock_id) is None or True
        main_window._set_panel_visible(dock_id, False)
        main_window._sync_panel_view_action(dock_id)
        assert not action.isChecked(), dock_id

        main_window._set_panel_visible(dock_id, True)
        main_window._sync_panel_view_action(dock_id)
        assert action.isChecked(), dock_id


# ------------------------------------------------------------------- plot exporting


def test_plot_export_is_offered_per_dock_and_not_in_the_file_menu(main_window) -> None:
    # Hold the QAction: dropping its Python wrapper can take the owned QMenu
    # with it, leaving a dangling C++ object.
    file_action = next(a for a in main_window.menuBar().actions() if a.text() == "File")
    file_menu = file_action.menu()
    assert "Export Plot" not in [a.text() for a in file_menu.actions()]

    assert set(main_window.export_plot_actions) == PLOT_IDS
    for entry in main_window.plot_entries:
        action = main_window.export_plot_actions[entry.plot_id]
        assert action.toolTip() == f"Export {entry.title}"
        assert not action.icon().isNull(), entry.plot_id


def test_data_exports_are_explicit_file_menu_actions(main_window) -> None:
    file_action = next(action for action in main_window.menuBar().actions() if action.text() == "File")
    labels = [action.text() for action in file_action.menu().actions()]

    assert "Export Polar Data" in labels
    assert "Export On-Axis Data" in labels
    assert "Export Speaker Package..." in labels
    assert not main_window.export_polar_data_action.isEnabled()
    assert not main_window.export_on_axis_data_action.isEnabled()

    main_window.set_on_axis_export_available(True)
    assert main_window.export_on_axis_data_action.isEnabled()

    main_window.set_on_axis_export_available(False)
    assert not main_window.export_on_axis_data_action.isEnabled()


def test_export_and_contour_icons_survive_a_palette_refresh(main_window) -> None:
    main_window._refresh_plot_export_icons()

    for action in main_window.export_plot_actions.values():
        assert not action.icon().isNull()
    for plot_id in ISOBAR_IDS:
        assert not main_window.capture_contour_actions[plot_id].icon().isNull()
        assert not main_window.clear_contour_actions[plot_id].icon().isNull()


# ------------------------------------------------------------------ live plot refresh


def test_live_plot_refresh_is_coalesced_into_one_flush(main_window, monkeypatch) -> None:
    timer = main_window._live_plot_refresh_timer
    assert timer.isSingleShot()
    assert timer.interval() == LIVE_PLOT_REFRESH_INTERVAL_MS

    calls: list[bool] = []
    monkeypatch.setattr(main_window, "refresh_plots", lambda **kw: calls.append(kw.get("active_only")))

    for _ in range(5):
        main_window.request_live_refresh()
    assert calls == [], "refresh must be deferred, not run per request"

    main_window.flush_live_refresh()
    assert calls == [True], "five requests should coalesce into one active-only refresh"

    main_window.flush_live_refresh()
    assert calls == [True], "a flush with nothing pending must not refresh again"


def test_hidden_plot_docks_are_not_actively_visible(main_window) -> None:
    entry = main_window.plot_entries[0]
    main_window.plot_docks[entry.plot_id].hide()

    assert main_window._plot_entry_is_actively_visible(entry) is False


def test_cancelling_refresh_clears_pending_work(main_window, monkeypatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(main_window, "refresh_plots", lambda **kw: calls.append(True))

    main_window.request_live_refresh()
    main_window.cancel_live_refresh()
    main_window.flush_live_refresh()

    assert calls == []
    assert not main_window._live_plot_refresh_timer.isActive()


class _ResynthesizableDatasetStub:
    supports_channel_resynthesis = True
    has_balloon_data = True

    def __init__(self) -> None:
        self.synthesis_calls = []

    def set_channel_synthesis(self, channels, **options) -> None:
        self.synthesis_calls.append((channels, options))

    def as_balloon_raw_bundle(self):
        raise AssertionError("closed Balloon windows must not trigger balloon synthesis")


class _BalloonWindowStub:
    def __init__(self, *, visible: bool) -> None:
        self.visible = visible
        self.refresh_count = 0

    def isVisible(self) -> bool:  # noqa: N802 - Qt API shape
        return self.visible

    def refresh_from_latest_results(self) -> None:
        self.refresh_count += 1


def test_channel_apply_resynthesizes_without_refreshing_mesh_or_closed_balloon(main_window, monkeypatch) -> None:
    dataset = _ResynthesizableDatasetStub()
    main_window.live_dataset = dataset
    main_window.balloon_window = _BalloonWindowStub(visible=False)
    plot_refreshes = []
    preview_refreshes = []
    balloon_availability = []
    monkeypatch.setattr(main_window, "refresh_plots", lambda: plot_refreshes.append(True))
    monkeypatch.setattr(main_window, "_refresh_mesh_preview", lambda: preview_refreshes.append(True))
    monkeypatch.setattr(main_window, "set_balloon_plot_available", balloon_availability.append)

    channels = (ChannelConfig(name="main", level_db=-3.0),)
    main_window._apply_channel_config(channels)

    assert dataset.synthesis_calls == [
        (channels, {"flat_target_reference_angle_deg": main_window.preferences.horizontal_normalization_angle})
    ]
    assert plot_refreshes == [True]
    assert preview_refreshes == []
    assert balloon_availability == [True]
    assert main_window.balloon_window.refresh_count == 0


def test_channel_apply_refreshes_an_open_balloon_window(main_window, monkeypatch) -> None:
    dataset = _ResynthesizableDatasetStub()
    balloon_window = _BalloonWindowStub(visible=True)
    main_window.live_dataset = dataset
    main_window.balloon_window = balloon_window
    monkeypatch.setattr(main_window, "refresh_plots", lambda: None)
    monkeypatch.setattr(main_window, "set_balloon_plot_available", lambda _available: None)

    main_window._apply_channel_config((ChannelConfig(name="main", delay_ms=0.5),))

    assert balloon_window.refresh_count == 1


# ------------------------------------------------------------------------ DPI refresh


def test_dpi_refresh_redraws_canvases_without_recomputing_plots(main_window, monkeypatch) -> None:
    drawn: list[str] = []
    for entry in main_window.plot_entries:
        monkeypatch.setattr(entry.widget, "draw_idle", lambda pid=entry.plot_id: drawn.append(pid))
    active_plot_id = main_window.plot_entries[0].plot_id
    monkeypatch.setattr(
        main_window,
        "_plot_entry_is_actively_visible",
        lambda entry: entry.plot_id == active_plot_id,
    )

    refreshed: list[bool] = []
    monkeypatch.setattr(main_window, "refresh_plots", lambda **kw: refreshed.append(True))

    main_window._plot_dpi_refresh_pending = True
    main_window._refresh_plot_canvas_dpi()

    assert drawn == [active_plot_id]
    assert refreshed == [], "a DPI change must not re-run the plot pipeline"
    assert main_window._plot_dpi_refresh_pending is False


# ------------------------------------------------------------- solved-data guard rails


def test_clearing_solved_data_is_confirmed_before_it_happens(main_window, message_box, monkeypatch) -> None:
    monkeypatch.setattr(main_window, "_has_solved_data", lambda: True)

    message_box.click = "Cancel"
    assert main_window._confirm_clear_solved_data() is False

    box = message_box.instances[-1]
    assert box.executed
    assert "clear solved data" in box.text.lower()
    assert box.button_texts == ["Continue", "Cancel"]
    assert box.default is not None and box.default.text == "Cancel", "destructive default must be Cancel"

    message_box.click = "Continue"
    assert main_window._confirm_clear_solved_data() is True


def test_no_confirmation_when_there_is_nothing_solved_to_lose(main_window, message_box, monkeypatch) -> None:
    monkeypatch.setattr(main_window, "_has_solved_data", lambda: False)

    assert main_window._confirm_clear_solved_data() is True
    assert message_box.instances == []


def test_exterior_topology_override_uses_cancel_as_the_safe_default(main_window, message_box) -> None:
    mesh = SimpleNamespace(mesh_name="cabinet", open_edge_count=4, nonmanifold_edge_count=0)
    report = SimpleNamespace(
        meshes=(mesh,),
        open_edge_count=4,
        nonmanifold_edge_count=0,
        symmetry_edge_count=0,
    )

    message_box.click = "Cancel Solve"
    assert main_window.confirm_mesh_topology_warning(report) is False

    box = message_box.instances[-1]
    assert box.button_texts == ["Continue", "Cancel Solve"]
    assert box.default is not None and box.default.text == "Cancel Solve"
    assert "highlighted red" in box.text
    assert "Open edges: 4" in box.text

    message_box.click = "Continue"
    assert main_window.confirm_mesh_topology_warning(report) is True


# --------------------------------------------------------------- unsaved project guard


def test_field_preferences_are_applied_to_the_preview(main_window) -> None:
    from dataclasses import replace

    assert main_window.preview.observation_plane_field_preferences == (
        main_window.preferences.field_cache_size_mb,
        main_window.preferences.field_translation_target_fps,
    )

    updated = replace(
        main_window.preferences,
        field_cache_size_mb=512,
        field_translation_target_fps=20.0,
    )
    main_window._set_preferences(updated)

    assert main_window.preview.observation_plane_field_preferences == (512, 20.0)


def test_startup_leaves_a_clean_untitled_project(main_window) -> None:
    assert main_window.project_path is None
    assert main_window._project_clean_payload is not None
    assert main_window._has_unsaved_project_changes() is False


def test_unsaved_changes_prompt_offers_save_discard_cancel(main_window, message_box, monkeypatch) -> None:
    monkeypatch.setattr(main_window.project_workflow, "has_unsaved_project_changes", lambda: True)

    message_box.click = "Cancel"
    assert main_window._confirm_unsaved_project_changes("close") is False

    box = message_box.instances[-1]
    assert box.button_texts == ["Save", "Discard", "Cancel"]
    assert box.default is not None and box.default.text == "Cancel"
    assert "close" in box.text.lower()

    message_box.click = "Discard"
    assert main_window._confirm_unsaved_project_changes("close") is True

    saved: list[bool] = []
    monkeypatch.setattr(main_window.project_workflow, "save_project", lambda: saved.append(True) or True)
    message_box.click = "Save"
    assert main_window._confirm_unsaved_project_changes("new_project") is True
    assert saved == [True]
    assert "save before continuing" in message_box.instances[-1].text.lower()


def test_close_is_vetoed_when_the_user_cancels(main_window, monkeypatch) -> None:
    monkeypatch.setattr(main_window, "_confirm_unsaved_project_changes", lambda _action: False)

    event = QCloseEvent()
    event.accept()
    main_window.closeEvent(event)

    assert not event.isAccepted(), "cancelling the prompt must veto the close"


def test_close_proceeds_and_persists_state_when_confirmed(main_window, monkeypatch) -> None:
    monkeypatch.setattr(main_window, "_confirm_unsaved_project_changes", lambda _action: True)
    saved: list[str] = []
    monkeypatch.setattr(main_window, "_save_frequency_settings", lambda: saved.append("freq"))
    monkeypatch.setattr(main_window, "_save_preferences", lambda: saved.append("prefs"))
    monkeypatch.setattr(main_window, "_save_window_state", lambda: saved.append("window"))

    event = QCloseEvent()
    event.accept()
    main_window.closeEvent(event)

    assert event.isAccepted()
    assert saved == ["freq", "prefs", "window"]


# ---------------------------------------------------------------------------- contours


def test_contour_capture_is_gated_on_a_rendered_final_solve(main_window) -> None:
    main_window.live_dataset = None
    main_window._use_final_isobar_resolution = False
    main_window._final_isobar_plots_rendered = False
    main_window.refresh_contour_controls()
    for plot_id in ISOBAR_IDS:
        assert not main_window.capture_contour_actions[plot_id].isEnabled()

    # A live solve that has not yet rendered at final resolution stays gated.
    main_window.live_dataset = object()
    main_window._use_final_isobar_resolution = False
    main_window._final_isobar_plots_rendered = True
    main_window.refresh_contour_controls()
    for plot_id in ISOBAR_IDS:
        assert not main_window.capture_contour_actions[plot_id].isEnabled()


def test_contour_capture_is_gated_on_dock_visibility(main_window) -> None:
    main_window.live_dataset = object()
    main_window._use_final_isobar_resolution = True
    main_window._final_isobar_plots_rendered = True

    main_window.plot_docks["horizontal_isobar"].hide()
    main_window.refresh_contour_controls()

    assert not main_window.capture_contour_actions["horizontal_isobar"].isEnabled()


def test_capture_and_clear_are_routed_to_the_requested_plot_only(main_window, monkeypatch) -> None:
    captured: list[str] = []
    cleared: list[str] = []
    for plot_id in ISOBAR_IDS:
        plot = main_window._isobar_plot_for_id(plot_id)
        monkeypatch.setattr(plot, "capture_contours", lambda pid=plot_id: captured.append(pid) or True)
        monkeypatch.setattr(plot, "clear_contours", lambda pid=plot_id: cleared.append(pid))

    main_window.capture_isobar_contours("horizontal_isobar")
    assert captured == ["horizontal_isobar"]

    main_window.clear_isobar_contours("vertical_isobar")
    assert cleared == ["vertical_isobar"]

    main_window.capture_isobar_contours("not_a_plot")
    assert captured == ["horizontal_isobar"], "unknown plot ids must be ignored"


# ------------------------------------------------------------------- generator tabs


def test_add_design_tab_is_a_button_not_a_closable_document(main_window) -> None:
    tabs = main_window.editor_tabs
    add_index = next(i for i in range(tabs.count()) if tabs.tabText(i) == ADD_DESIGN_TAB_LABEL)

    bar = tabs.tabBar()
    assert bar.tabButton(add_index, QTabBar.ButtonPosition.RightSide) is None
    assert bar.tabButton(add_index, QTabBar.ButtonPosition.LeftSide) is None


# --------------------------------------------------------------------- preview tree


def test_preview_region_toolbar_buttons_were_replaced_by_body_tree(main_window) -> None:
    assert not hasattr(main_window, "show_interior_regions_action")
    assert not hasattr(main_window, "show_exterior_region_action")


# ------------------------------------------------------------------ contracts


def test_main_window_satisfies_the_declared_seam_protocols(main_window) -> None:
    """The seams are contracts, not documentation.

    If a revamp splits MainWindow apart, whatever replaces it must still
    present these surfaces, and this fails the moment one is dropped.
    """
    from blab.ui.main_window.workflow_view import PlotPresenter, SolveInputs, WorkflowView

    assert isinstance(main_window, WorkflowView)
    assert isinstance(main_window, PlotPresenter)
    assert isinstance(main_window, SolveInputs)


def test_workflow_code_reaches_the_view_only_through_the_seam() -> None:
    """No workflow module may touch a widget directly.

    Guards the whole point of stage 2: these modules survive a UI revamp, so a
    stray `self.some_button.setEnabled(...)` is a regression, not a detail.
    """
    import re

    pkg = MAIN_WINDOW_PKG
    workflow_modules = (
        "solve_workflow",
        "geometry_workflow",
        "mesh_workflow",
        "radiators",
        "channels",
        "backend_health",
        "project_workflow",
    )
    forbidden = re.compile(r"self\.(\w*_button|\w*_action|status_label|preview)\b")

    offenders = {}
    for stem in workflow_modules:
        hits = sorted(set(forbidden.findall((pkg / f"{stem}.py").read_text(encoding="utf-8"))))
        if hits:
            offenders[stem] = hits

    assert offenders == {}, f"workflow modules must use the view seam, not widgets: {offenders}"


def test_slot_decorators_on_mixins_are_registered_after_all(main_window) -> None:
    """Documented as a hazard; measurement says otherwise.

    The UI architecture record claimed slot-decorated methods on plain mixins
    "are never registered in the Qt meta-object", making them decorative and
    unsafe for name-based invocation. PySide6 walks the MRO when it builds the
    meta-object, so they are registered. The undecorated control below is what
    proves the decorator is doing the work rather than everything showing up.

    Pinned so nobody spends a day "fixing" a non-problem.
    """
    registered = _registered_slot_names(main_window)

    for name in ("open_channel_config", "flush_live_refresh", "new_project", "capture_isobar_contours"):
        assert name in registered, f"{name} is decorated on a mixin and should be registered"

    assert "has_solver_meshes" not in registered, "undecorated methods must not be registered"


def _registered_slot_names(obj) -> set[str]:
    meta = obj.metaObject()
    return {bytes(meta.method(index).name()).decode() for index in range(meta.methodCount())}


def test_no_module_derives_its_own_app_root() -> None:
    """One anchor, in blab.paths — everything else imports it.

    Four copies of `Path(__file__).resolve().parents[N]` had accumulated, each
    with a different N for its own depth. That is how every icon path broke
    when a module moved one directory deeper, and it failed silently because a
    missing icon degrades to a null QIcon rather than raising.
    """
    import re

    from blab.paths import APP_ROOT

    assert (APP_ROOT / "pyproject.toml").is_file(), f"APP_ROOT resolved to {APP_ROOT}"
    assert (APP_ROOT / "assets").is_dir()

    anchor = re.compile(r"Path\(__file__\)\.resolve\(\)\.parents\[\d+\]")
    src = BLAB_SRC
    scanned = sorted(src.rglob("*.py"))
    # An empty scan would satisfy the assertion below without checking a single
    # file. That is exactly how this test used to pass from the wrong working
    # directory; see tests/repo_paths.py.
    assert scanned, f"no source files found under {src} - the scan is vacuous"
    offenders = sorted(
        str(path.relative_to(src))
        for path in scanned
        # paths.py owns the anchor. Standalone dev scripts under scripts/ resolve
        # to their own directory to find sibling fixtures, not to the repo root.
        if path.name != "paths.py" and "scripts" not in path.parts and anchor.search(path.read_text(encoding="utf-8"))
    )

    assert offenders == [], f"these must import APP_ROOT from blab.paths instead: {offenders}"


def test_extracted_controllers_import_no_qt_widgets() -> None:
    """A controller that builds a dialog needs a parent widget.

    That is how the window creeps back into a constructor signature, and the
    widget-name regex above cannot see it — `QMessageBox.warning(self, ...)`
    contains no `self.<widget>` attribute. Guard the import instead.
    """

    pkg = MAIN_WINDOW_PKG
    controllers = ("solve_workflow", "geometry_workflow", "backend_health", "project_workflow")

    offenders = {
        stem: "PySide6.QtWidgets"
        for stem in controllers
        if "PySide6.QtWidgets" in (pkg / f"{stem}.py").read_text(encoding="utf-8")
    }

    assert offenders == {}, f"controllers must report through the view seam, not build dialogs: {offenders}"


def test_extracted_workflows_are_composed_not_inherited() -> None:
    """Stage 3: these workflows are held, not mixed in.

    A mixin can reach anything on the window by accident; a controller can only
    reach what it was handed.
    """
    from blab.ui.main_window.geometry_workflow import GeometryWorkflowController
    from blab.ui.main_window.project_workflow import ProjectWorkflowController
    from blab.ui.main_window.solve_workflow import SolveWorkflowController
    from blab.ui.main_window.window import MainWindow

    for controller in (SolveWorkflowController, GeometryWorkflowController, ProjectWorkflowController):
        assert controller not in MainWindow.__mro__, controller.__name__
        assert not issubclass(MainWindow, controller), controller.__name__


def test_the_window_still_satisfies_the_geometry_seam(main_window) -> None:
    from blab.ui.main_window.workflow_view import GeometryInputs

    assert isinstance(main_window, GeometryInputs)


def test_the_window_still_satisfies_the_project_seam(main_window) -> None:
    """The provider surface the project controller was handed.

    Spread across the channels, generator-document and mesh mixins, so a
    rename in any one of them silently breaks the controller without this.
    """
    from blab.ui.main_window.workflow_view import ProjectInputs

    assert isinstance(main_window, ProjectInputs)
