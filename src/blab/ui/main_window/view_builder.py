"""Widget-tree construction: menus, docks, layout, and control factories."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from blab.ui.application_state import OperationPhase
from blab.ui.main_window.workflow_view import (
    FrequencyRange,
    UnsavedChoice,
    WorkflowControls,
    workflow_controls,
)
from blab.ui.main_window_widgets import (
    DockTitleBar,
)
from blab.ui.plots import (
    frequency_to_slider_value,
    slider_value_to_frequency,
)


def _framed_dock_content(widget: QWidget) -> QFrame:
    """Sunken 3D border for dock panels; bare canvases draw none."""
    frame = QFrame()
    frame.setFrameShape(QFrame.StyledPanel)
    frame.setFrameShadow(QFrame.Sunken)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(2, 2, 2, 2)
    layout.addWidget(widget)
    return frame


class ViewBuilderMixin:
    """Widget-tree construction: menus, docks, layout, and control factories.

    Mixed into :class:`~blab.ui.main_window.window.MainWindow`.
    """

    def _build_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        new_project_action = QAction("New Project", self)
        new_project_action.triggered.connect(self.new_project)
        file_menu.addAction(new_project_action)

        file_menu.addSeparator()

        save_project_action = QAction("Save Project", self)
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)

        save_project_as_action = QAction("Save Project As", self)
        save_project_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_project_as_action)

        load_project_action = QAction("Open Project", self)
        load_project_action.triggered.connect(self.load_project)
        file_menu.addAction(load_project_action)

        self.open_recent_menu = file_menu.addMenu("Open Recent")
        self._rebuild_open_recent_menu()

        file_menu.addSeparator()

        import_action = QAction("Import Waveguide Design...", self)
        import_action.triggered.connect(self.import_config)
        file_menu.addAction(import_action)

        export_cfg_action = QAction("Export Waveguide Design...", self)
        export_cfg_action.triggered.connect(self.export_config)
        file_menu.addAction(export_cfg_action)

        for entry in self.plot_entries:
            limits_action = QAction("Plot Limits", self)
            limits_action.setToolTip(f"Set axis limits for {entry.title}")
            limits_action.triggered.connect(
                lambda _checked=False, plot_id=entry.plot_id: self.open_plot_limits(plot_id)
            )
            self.plot_limit_actions[entry.plot_id] = limits_action
            action = QAction("Save Plot Image", self)
            action.setToolTip(f"Save {entry.title} image")
            action.setEnabled(False)
            action.triggered.connect(lambda _checked=False, plot_id=entry.plot_id: self.export_plot(plot_id))
            self.export_plot_actions[entry.plot_id] = action
            data_action = QAction("Export Plot Data", self)
            data_action.setToolTip(f"Export {entry.title} data")
            data_action.setEnabled(False)
            data_action.triggered.connect(
                lambda _checked=False, plot_id=entry.plot_id: self.export_plot_data(plot_id)
            )
            self.export_plot_data_actions[entry.plot_id] = data_action
            if entry.plot_id in {"horizontal_isobar", "vertical_isobar"}:
                capture_action = QAction("Capture Contours", self)
                capture_action.setToolTip(f"Capture contours for {entry.title}")
                capture_action.setEnabled(False)
                capture_action.triggered.connect(
                    lambda _checked=False, plot_id=entry.plot_id: self.capture_isobar_contours(plot_id)
                )
                self.capture_contour_actions[entry.plot_id] = capture_action
                clear_action = QAction("Clear Contours", self)
                clear_action.setToolTip(f"Clear contours for {entry.title}")
                clear_action.setEnabled(False)
                clear_action.triggered.connect(
                    lambda _checked=False, plot_id=entry.plot_id: self.clear_isobar_contours(plot_id)
                )
                self.clear_contour_actions[entry.plot_id] = clear_action
            elif entry.plot_id == "spinorama":
                self.spherical_spin_action = QAction("Spherical Spin", self)
                self.spherical_spin_action.setToolTip(
                    "Use full-sphere samples"
                )
                self.spherical_spin_action.setCheckable(True)
                self.spherical_spin_action.setChecked(False)
                self.spherical_spin_action.setEnabled(False)
                self.spherical_spin_action.toggled.connect(self.set_spherical_spin_enabled)

        self.export_speaker_package_action = QAction("Export Speaker Package...", self)
        self.export_speaker_package_action.setToolTip("Configure, solve, and export a .blabsp speaker package")
        self.export_speaker_package_action.triggered.connect(self.export_speaker_package)
        file_menu.addAction(self.export_speaker_package_action)

        view_menu = self.menuBar().addMenu("View")
        self.balloon_plot_action = QAction("Balloon Plot", self)
        self.balloon_plot_action.setEnabled(False)
        self.balloon_plot_action.triggered.connect(self.open_balloon_plot)
        view_menu.addAction(self.balloon_plot_action)
        view_menu.addSeparator()
        for dock_id, title in (
            ("editor", "Waveguide Design Panel"),
            ("preview", "Mesh Preview Panel"),
        ):
            action = QAction(title, self)
            action.setCheckable(True)
            action.setChecked(True)
            view_menu.addAction(action)
            self.panel_view_actions[dock_id] = action
        view_menu.addSeparator()
        for entry in self.plot_entries:
            action = QAction(entry.title, self)
            action.setCheckable(True)
            action.setChecked(True)
            view_menu.addAction(action)
            self.plot_view_actions[entry.plot_id] = action

        edit_menu = self.menuBar().addMenu("Edit")
        preferences_action = QAction("Preferences", self)
        preferences_action.triggered.connect(self.open_preferences)
        edit_menu.addAction(preferences_action)

        about_menu = self.menuBar().addMenu("About")
        diagnostics_action = QAction("Diagnostic Info", self)
        diagnostics_action.triggered.connect(self.open_diagnostics)
        about_menu.addAction(diagnostics_action)

        donate_action = QAction("Donate", self)
        donate_action.triggered.connect(self.open_donate)
        about_menu.addAction(donate_action)

        help_action = QAction("Help", self)
        help_action.triggered.connect(self.open_help)
        about_menu.addAction(help_action)

    def _make_panel_dock(
        self,
        object_name: str,
        title: str,
        widget: QWidget,
        *,
        tool_actions: tuple[QAction, ...] = (),
    ) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setWidget(_framed_dock_content(widget))
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        )
        dock.setTitleBarWidget(DockTitleBar(title, dock, tool_actions=tool_actions))
        return dock

    def _build_layout(self) -> None:
        self.editor_panel = QWidget()
        editor_layout = QVBoxLayout(self.editor_panel)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.addWidget(self.editor_tabs)

        self.editor_container = QWidget()
        editor_container_layout = QHBoxLayout(self.editor_container)
        editor_container_layout.setContentsMargins(0, 0, 0, 0)
        editor_container_layout.setSpacing(0)
        editor_container_layout.addWidget(self.editor_panel, 1)

        self.workspace = QMainWindow()
        self.workspace.setDockOptions(
            QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks | QMainWindow.AnimatedDocks
        )
        self.syntax_highlighting_action = QAction("Syntax highlighting", self)
        self.syntax_highlighting_action.setToolTip("Syntax highlighting")
        self.syntax_highlighting_action.setCheckable(True)
        self.syntax_highlighting_action.setChecked(self.syntax_highlighting_enabled)
        self.syntax_highlighting_action.toggled.connect(self.set_syntax_highlighting_enabled)
        self.editor_dock = self._make_panel_dock(
            "ath_editor_dock",
            "Waveguide Design",
            self.editor_container,
            tool_actions=(self.syntax_highlighting_action,),
        )
        self.preview_dock = self._make_panel_dock(
            "mesh_preview_dock",
            "Mesh Preview",
            self.preview,
        )
        self.workspace.addDockWidget(Qt.LeftDockWidgetArea, self.editor_dock)
        self.workspace.addDockWidget(Qt.LeftDockWidgetArea, self.preview_dock)
        self.workspace.splitDockWidget(self.editor_dock, self.preview_dock, Qt.Horizontal)
        previous_plot_dock = None
        for entry in self.plot_entries:
            tool_actions = [self.plot_limit_actions[entry.plot_id]]
            tool_actions.extend(
                [
                    action
                    for action in (
                        self.export_plot_actions.get(entry.plot_id),
                        self.export_plot_data_actions.get(entry.plot_id),
                        self.capture_contour_actions.get(entry.plot_id),
                        self.clear_contour_actions.get(entry.plot_id),
                    )
                    if action is not None
                ]
            )
            if entry.plot_id in {"electrical_impedance", "on_axis_frequency_response"}:
                tool_actions.extend((entry.widget.trace_filter_action, entry.widget.show_phase_action))
            elif entry.plot_id in {"acoustic_impedance", "group_delay", "transducer_excursion"}:
                tool_actions.append(entry.widget.trace_filter_action)
            elif entry.plot_id == "max_spl":
                tool_actions.extend((entry.widget.calculate_action, entry.widget.trace_filter_action))
            elif entry.plot_id == "spinorama":
                tool_actions.append(self.spherical_spin_action)
            dock = self._make_panel_dock(
                f"{entry.plot_id}_dock",
                entry.title,
                entry.widget,
                tool_actions=tuple(tool_actions),
            )
            self.plot_docks[entry.plot_id] = dock
            self.workspace.addDockWidget(Qt.RightDockWidgetArea, dock)
            if previous_plot_dock is None:
                self.workspace.splitDockWidget(self.preview_dock, dock, Qt.Horizontal)
            else:
                self.workspace.tabifyDockWidget(previous_plot_dock, dock)
            previous_plot_dock = dock
        if previous_plot_dock is not None:
            previous_plot_dock.raise_()
        self.workspace.resizeDocks(
            [self.editor_dock, self.preview_dock, *self.plot_docks.values()],
            [420, 520, *([520] * len(self.plot_docks))],
            Qt.Horizontal,
        )
        for dock_id, dock in (
            ("editor", self.editor_dock),
            ("preview", self.preview_dock),
        ):
            action = self.panel_view_actions.get(dock_id)
            if action is not None:
                action.toggled.connect(lambda checked, dock_id=dock_id: self._set_panel_visible(dock_id, checked))
                dock.visibilityChanged.connect(lambda _visible, dock_id=dock_id: self._sync_panel_view_action(dock_id))
        for entry in self.plot_entries:
            dock = self.plot_docks[entry.plot_id]
            action = self.plot_view_actions.get(entry.plot_id)
            if action is not None:
                action.toggled.connect(lambda checked, plot_id=entry.plot_id: self._set_plot_visible(plot_id, checked))
                dock.visibilityChanged.connect(
                    lambda _visible, plot_id=entry.plot_id: self._sync_plot_view_action(plot_id)
                )

        controls = QFrame()
        controls_layout = QHBoxLayout(controls)
        controls_layout.addWidget(self.generate_button)
        controls_layout.addWidget(self.solve_button)
        controls_layout.addWidget(self.cancel_button)
        controls_layout.addWidget(self.mesh_config_button)
        controls_layout.addWidget(self.system_config_button)
        controls_layout.addWidget(self.channel_config_button)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("Min Hz"))
        controls_layout.addWidget(self.freq_min_slider)
        controls_layout.addWidget(self.freq_min_spin)
        controls_layout.addWidget(QLabel("Max Hz"))
        controls_layout.addWidget(self.freq_max_slider)
        controls_layout.addWidget(self.freq_max_spin)
        controls_layout.addWidget(QLabel("Frequencies"))
        controls_layout.addWidget(self.freq_count_slider)
        controls_layout.addWidget(self.freq_count_spin)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.workspace, stretch=1)
        layout.addWidget(controls)
        # QLabel is a QFrame, so this gives a sunken text box.
        self.status_label.setFrameShape(QFrame.StyledPanel)
        self.status_label.setFrameShadow(QFrame.Sunken)
        self.status_label.setMargin(4)
        layout.addWidget(self.status_label)
        self.setCentralWidget(central)
        self._refresh_plot_export_icons()

    def _wire_controls(self) -> None:
        self.generate_button.clicked.connect(self.generate_geometry)
        self.solve_button.clicked.connect(self.start_solve)
        self.cancel_button.clicked.connect(self.cancel_current_operation)
        self.mesh_config_button.clicked.connect(self.open_mesh_config)
        self.system_config_button.clicked.connect(self.open_system_config)
        self.channel_config_button.clicked.connect(self.open_channel_config)

        self.freq_min_slider.valueChanged.connect(
            lambda value: self._sync_frequency_spin_from_slider(self.freq_min_spin, value)
        )
        self.freq_min_spin.valueChanged.connect(
            lambda value: self._sync_frequency_slider_from_spin(self.freq_min_slider, value)
        )
        self.freq_max_slider.valueChanged.connect(
            lambda value: self._sync_frequency_spin_from_slider(self.freq_max_spin, value)
        )
        self.freq_max_spin.valueChanged.connect(
            lambda value: self._sync_frequency_slider_from_spin(self.freq_max_slider, value)
        )
        self.freq_count_slider.valueChanged.connect(self.freq_count_spin.setValue)
        self.freq_count_spin.valueChanged.connect(self.freq_count_slider.setValue)
        self.freq_min_spin.valueChanged.connect(self._save_frequency_settings)
        self.freq_max_spin.valueChanged.connect(self._save_frequency_settings)
        self.freq_count_spin.valueChanged.connect(self._save_frequency_settings)

    def _sync_frequency_spin_from_slider(self, spin: QSpinBox, slider_value: int) -> None:
        with QSignalBlocker(spin):
            spin.setValue(slider_value_to_frequency(slider_value))

    def _sync_frequency_slider_from_spin(self, slider: QSlider, freq_hz: int) -> None:
        with QSignalBlocker(slider):
            slider.setValue(frequency_to_slider_value(freq_hz))

    def _make_slider(self, minimum: int, maximum: int, value: int) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        return slider

    def _make_spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    # -- WorkflowView -----------------------------------------------------
    # The only sanctioned path from workflow logic to these widgets. Keep this
    # adapter dumb: decisions belong in workflow_controls(), not here.

    def show_status(self, message: str) -> None:
        self.status_label.setText(message)

    def warn(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def confirm(self, title: str, message: str) -> bool:
        return (
            QMessageBox.question(
                self,
                title,
                message,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            == QMessageBox.Yes
        )

    def confirm_mesh_topology_warning(self, report) -> bool:
        from blab.mesh_topology import exterior_mesh_topology_warning_text

        message = QMessageBox(
            QMessageBox.Warning,
            "Exterior Mesh Is Not Watertight",
            exterior_mesh_topology_warning_text(report),
            QMessageBox.NoButton,
            self,
        )
        continue_button = message.addButton("Continue", QMessageBox.AcceptRole)
        cancel_button = message.addButton("Cancel Solve", QMessageBox.RejectRole)
        message.setDefaultButton(cancel_button)
        message.exec()
        return message.clickedButton() is continue_button

    def show_mesh_topology_issues(self, report) -> None:
        self.preview.set_topology_report(report)

    def ask_unsaved_changes(self, *, closing: bool) -> UnsavedChoice:
        message = QMessageBox(
            QMessageBox.Warning,
            "Unsaved Changes",
            (
                "You have unsaved changes. Are you sure you want to close?"
                if closing
                else "You have unsaved changes. Save before continuing?"
            ),
            QMessageBox.NoButton,
            self,
        )
        save_button = message.addButton("Save", QMessageBox.AcceptRole)
        discard_button = message.addButton("Discard", QMessageBox.DestructiveRole)
        cancel_button = message.addButton("Cancel", QMessageBox.RejectRole)
        # Cancel is the safe default: dismissing the dialog must never be the
        # gesture that throws the work away.
        message.setDefaultButton(cancel_button)
        message.exec()
        clicked = message.clickedButton()
        if clicked is save_button:
            return UnsavedChoice.SAVE
        if clicked is discard_button:
            return UnsavedChoice.DISCARD
        return UnsavedChoice.CANCEL

    def choose_open_file(self, title: str, file_filter: str) -> Path | None:
        return self.file_dialogs.open_file(self, title, file_filter)

    def choose_save_file(self, title: str, file_filter: str, suggested_name: str) -> Path | None:
        return self.file_dialogs.save_file(self, title, file_filter, suggested_name)

    def set_busy_cursor(self, busy: bool) -> None:
        if busy:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    def apply_workflow_controls(self, controls: WorkflowControls) -> None:
        self.generate_button.setEnabled(controls.generate)
        self.solve_button.setEnabled(controls.solve)
        self.cancel_button.setEnabled(controls.cancel)
        self.mesh_config_button.setEnabled(controls.mesh_config)
        self.system_config_button.setEnabled(controls.system_config)
        self.channel_config_button.setEnabled(controls.channel_config)
        self.export_speaker_package_action.setEnabled(controls.speaker_package)

    def set_workflow_phase(self, phase: OperationPhase, *, cancel_available: bool = True) -> None:
        """Convenience for controllers: map a phase and apply it in one step."""
        self.apply_workflow_controls(
            workflow_controls(
                phase,
                has_solver_meshes=self.has_solver_meshes(),
                cancel_available=cancel_available,
            )
        )

    def set_plot_exports_available(self, available: bool) -> None:
        for plot_id, action in self.export_plot_actions.items():
            action.setEnabled(available and self.plot_data_is_available(plot_id))
        for plot_id, action in self.export_plot_data_actions.items():
            action.setEnabled(available and self.plot_data_is_available(plot_id))

    def set_max_spl_available(self, available: bool) -> None:
        self.max_spl_plot.calculate_action.setEnabled(available)

    def set_max_spl_export_available(self, available: bool) -> None:
        action = self.export_plot_actions.get("max_spl")
        if action is not None:
            action.setEnabled(available)
        data_action = self.export_plot_data_actions.get("max_spl")
        if data_action is not None:
            data_action.setEnabled(available)

    def set_system_config_available(self, available: bool) -> None:
        self.system_config_button.setEnabled(available)
        self.export_speaker_package_action.setEnabled(available and self.solve_button.isEnabled())

    def clear_mesh_preview(self) -> None:
        self.preview.clear()
        controller = getattr(self, "observation_plane_controller", None)
        if controller is not None:
            controller.sync_view()

    def show_mesh_preview(self, meshes, **options) -> None:
        self.preview.load_mesh_configs(meshes, **options)
        controller = getattr(self, "observation_plane_controller", None)
        if controller is not None:
            controller.sync_view()

    def set_balloon_plot_available(self, available: bool) -> None:
        self.balloon_plot_action.setEnabled(available)

    def frequency_range(self) -> FrequencyRange:
        return FrequencyRange(
            min_hz=int(self.freq_min_spin.value()),
            max_hz=int(self.freq_max_spin.value()),
            count=int(self.freq_count_spin.value()),
        )

    def set_frequency_range(self, value: FrequencyRange) -> None:
        controls = (
            self.freq_min_spin,
            self.freq_max_spin,
            self.freq_count_spin,
            self.freq_min_slider,
            self.freq_max_slider,
            self.freq_count_slider,
        )
        # Blocked so restoring a project's range does not look like the user
        # editing it, which would re-save settings and invalidate the solve.
        blockers = [QSignalBlocker(control) for control in controls]
        self.freq_min_spin.setValue(value.min_hz)
        self.freq_max_spin.setValue(value.max_hz)
        self.freq_count_spin.setValue(value.count)
        self.freq_min_slider.setValue(frequency_to_slider_value(value.min_hz))
        self.freq_max_slider.setValue(frequency_to_slider_value(value.max_hz))
        self.freq_count_slider.setValue(value.count)
        del blockers
