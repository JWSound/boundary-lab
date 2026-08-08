from pathlib import Path

from repo_paths import MAIN_WINDOW_PKG, source_text


def main_window_source(*stems: str) -> str:
    """Concatenated source of the main_window package.

    ``main_window.py`` was split into a package of mixins; these remaining
    assertions still describe wiring rather than behaviour, so they read the
    package as one blob. Pass module stems to narrow the read when a test
    slices between two markers that must stay in the same module.
    """
    paths = [MAIN_WINDOW_PKG / f"{stem}.py" for stem in stems] if stems else sorted(MAIN_WINDOW_PKG.glob("*.py"))
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)



def test_on_axis_and_spinorama_canvases_keep_distinct_update_signatures() -> None:
    source = source_text("ui", "plots.py")

    on_axis_start = source.index("class OnAxisResponseCanvas")
    spinorama_start = source.index("class SpinoramaCanvas")
    on_axis_block = source[on_axis_start:spinorama_start]
    spinorama_block = source[spinorama_start:]

    assert "def update_plot(" in on_axis_block
    assert "horizontal_spl_db: np.ndarray," in on_axis_block
    assert "vertical_spl_db: np.ndarray," not in on_axis_block

    assert "def update_plot(" in spinorama_block
    assert "horizontal_spl_db: np.ndarray," in spinorama_block
    assert "vertical_spl_db: np.ndarray," in spinorama_block


def test_spinorama_canvas_uses_fixed_layout_and_external_legend() -> None:
    source = source_text("ui", "plots.py")
    spinorama_block = source[source.index("class SpinoramaCanvas") :]

    assert "tight_layout=True" not in spinorama_block
    assert "subplots_adjust" in spinorama_block
    assert "left=PLOT_LEFT_MARGIN" in spinorama_block
    assert "right=PLOT_RIGHT_MARGIN" in spinorama_block
    assert 'set_label_position("right")' in spinorama_block
    assert "bbox_to_anchor=(0.5, SPINORAMA_LEGEND_BOTTOM)" in spinorama_block
    # Figure coords: an axes-relative drop scales with the axes height.
    assert "bbox_transform=self.figure.transFigure" in spinorama_block
    assert "SPINORAMA_SPL_LIMITS" in spinorama_block
    assert "SPINORAMA_DI_LIMITS" in spinorama_block
    assert "ncols=4" in spinorama_block


def test_plot_widgets_use_compact_title_padding() -> None:
    plot_source = source_text("ui", "plots.py")

    assert "PLOT_TITLE_PAD = 7" in plot_source
    assert "GRID_LINE_ALPHA = 0.6" in plot_source
    assert "set_title(self.title, pad=PLOT_TITLE_PAD)" in plot_source
    assert "set_yticks(np.arange(-180, 181, 45))" in plot_source
    assert 'grid(which="major", color="#808080", linewidth=0.8, alpha=GRID_LINE_ALPHA)' in plot_source


def test_main_window_uses_detachable_panel_docks() -> None:
    source = main_window_source()
    widgets_source = source_text("ui", "main_window_widgets.py")

    assert "QDockWidget" in source
    assert "class DockTitleBar" in widgets_source
    assert "save_action: QAction | None = None" in widgets_source
    assert "tool_actions: tuple[QAction, ...] = ()" in widgets_source
    assert "button.setDefaultAction(action)" in widgets_source
    assert "self.tool_buttons.append(button)" in widgets_source
    assert (
        "dock.setTitleBarWidget(DockTitleBar(title, dock, save_action=save_action, tool_actions=tool_actions))"
        in source
    )
    assert "save_action=self.export_plot_actions.get(entry.plot_id)" in source
    assert "tool_actions=tuple(" in source
    assert "close_button.clicked.connect(dock.close)" in widgets_source
    assert "event.ignore()" in widgets_source
    assert "collapse_editor" not in source
    assert "ath_editor_collapsed" not in source
    assert "ath_editor_width" not in source
    assert "self.workspace = QMainWindow()" in source
    assert "self.workspace.setCentralWidget(QWidget())" not in source
    assert "QMainWindow.AllowNestedDocks" in source
    assert "QMainWindow.AllowTabbedDocks" in source
    assert "self.workspace.addDockWidget" in source
    assert "self.workspace.splitDockWidget" in source
    assert "self.plot_docks: dict[str, QDockWidget]" in source
    assert "self.plot_docks[entry.plot_id] = dock" in source
    assert "self.workspace.tabifyDockWidget(previous_plot_dock, dock)" in source
    assert '"Plots Panel"' not in source
    assert "self.plots_dock" not in source
    assert 'settings_bool(self.settings, f"plots/{entry.plot_id}/visible", True)' not in source
    assert 'settings.setValue(f"plots/{plot_id}/visible"' not in source
    assert (
        "action.toggled.connect(lambda checked, dock_id=dock_id: self._set_panel_visible(dock_id, checked))" in source
    )
    assert (
        "dock.visibilityChanged.connect(lambda _visible, dock_id=dock_id: self._sync_panel_view_action(dock_id))"
        in source
    )
    assert "def _set_panel_visible(" in source
    assert "def _sync_panel_view_action(" in source
    assert '("channel_config", "Channel Config Panel")' not in source
    assert "dock.visibilityChanged.connect(" in source
    assert "lambda _visible, plot_id=entry.plot_id: self._sync_plot_view_action(plot_id)" in source
    assert "self.workspace.saveState()" in source
    assert "self.workspace.restoreState(dock_state)" in source
    assert "window/dock_state" in source
    assert "DEFAULT_DOCK_STATE_B64" in source
    assert 'QByteArray.fromBase64(DEFAULT_DOCK_STATE_B64.encode("ascii"))' in source


def test_streaming_plot_canvases_reuse_artists_and_layout() -> None:
    source = source_text("ui", "plots.py")
    impedance_block = source[source.index("class ImpedanceCanvas") : source.index("class OnAxisResponseCanvas")]
    on_axis_block = source[source.index("class OnAxisResponseCanvas") : source.index("class SpinoramaCanvas")]
    spinorama_block = source[source.index("class SpinoramaCanvas") :]

    assert "tight_layout=True" not in impedance_block
    assert "tight_layout=True" not in on_axis_block
    assert "self._lines" in impedance_block
    assert "self._lines" in on_axis_block
    assert ".set_data(" in impedance_block
    assert ".set_data(" in on_axis_block
    assert impedance_block.count("clear_plot_axes(self.axes)") == 1
    assert on_axis_block.count("clear_plot_axes(self.axes)") == 1
    assert "self._spl_lines" in spinorama_block
    assert "self._di_lines" in spinorama_block
    assert "self._spl_lines[name].set_data" in spinorama_block
    assert "self._di_lines[name].set_data" in spinorama_block
    assert spinorama_block.count("clear_plot_axes(self.axes)") == 1
    assert spinorama_block.count("clear_plot_axes(self.di_axes)") == 1


def test_channel_config_changes_apply_only_on_apply_button() -> None:
    dialog_source = source_text("ui", "dialogs.py")
    main_source = main_window_source()
    channel_dialog = dialog_source[dialog_source.index("class ChannelConfigDialog") :]

    assert "channelsChanged" not in channel_dialog
    assert "_emit_channels_changed" not in channel_dialog
    assert (
        "button_flags = QDialogButtonBox.Apply if self._embedded else QDialogButtonBox.Apply | QDialogButtonBox.Close"
        in channel_dialog
    )
    assert "buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)" in channel_dialog
    assert "buttons.rejected.connect(self.closeRequested.emit if self._embedded else self.reject)" in channel_dialog
    assert "button_row.addWidget(buttons)" in channel_dialog
    assert "layout.addWidget(buttons)" not in channel_dialog
    assert "_preview_channel_config" not in main_source
    assert "dialog.channelsApplied.connect(self._apply_channel_config)" in main_source


def test_server_health_worker_runs_query_with_timeout() -> None:
    worker_source = (Path(__file__).resolve().parents[1] / "src" / "blab" / "ui" / "server_health_worker.py").read_text(
        encoding="utf-8"
    )

    assert "class ServerHealthCheckWorker(QObject)" in worker_source
    assert "succeeded = Signal(str, object)" in worker_source
    assert "failed = Signal(str)" in worker_source
    assert "access_token=self.access_token" in worker_source
    assert "timeout_s=self.timeout_s" in worker_source
    assert "self.finished.emit()" in worker_source


def test_preferences_no_longer_expose_worker_count() -> None:
    dialog_source = source_text("ui", "dialogs.py")
    settings_source = source_text("ui", "settings.py")
    main_source = main_window_source()
    config_source = source_text("config.py")
    assembler_source = source_text("ui", "simulation_assembler.py")
    solve_source = main_window_source("solve_workflow")

    assert "worker_count_spin" not in dialog_source
    assert '"Worker Count"' not in dialog_source
    assert "worker_count:" not in settings_source
    assert '"preferences/worker_count"' not in settings_source
    assert "preferences.worker_count" not in main_source
    assert "workers=1" in assembler_source
    assert "worker_count=1" in solve_source
    assert "workers: int = 1" in config_source


def test_completed_solves_use_final_isobar_resolution() -> None:
    plot_source = source_text("ui", "plots.py")
    main_source = main_window_source()
    dialog_source = source_text("ui", "dialogs.py")
    settings_source = source_text("ui", "settings.py")

    assert '"low": 180' in settings_source
    assert '"medium": 250' in settings_source
    assert '"high": 500' in settings_source
    assert "def live_plot_freq_samples(" in settings_source
    assert '"Live Plot Quality", self.live_plot_quality_combo' in dialog_source
    assert '"Live Plot Streaming", self.live_plot_streaming_check' in dialog_source
    assert '("Live Plot Streaming", self.live_plot_streaming_check, "")' in dialog_source
    assert 'INFO_ICON_PATH = APP_ROOT / "assets" / "info-16.ico"' in dialog_source
    assert "info_icon = QIcon(str(INFO_ICON_PATH))" in dialog_source
    assert "label.setToolTip(tooltip)" in dialog_source
    assert "widget.setToolTip(tooltip)" in dialog_source
    assert "icon_label.setToolTip(tooltip)" in dialog_source
    assert "self.live_plot_quality_combo.setEnabled(preferences.live_plot_streaming)" in dialog_source
    assert "self.live_plot_streaming_check.toggled.connect(self.live_plot_quality_combo.setEnabled)" in dialog_source
    solver_config_block = dialog_source[
        dialog_source.index('"Solver Config"') : dialog_source.index('"Observation Config"')
    ]
    application_block = dialog_source[
        dialog_source.index('"Application"') : dialog_source.index(
            "right_column.addStretch", dialog_source.index('"Application"')
        )
    ]
    assert '"BEM Solver", self.solve_backend_combo' in dialog_source
    assert '"BEM Solver", self.solve_backend_combo' in solver_config_block
    assert '"Solve Server URL", self.solve_server_url_edit' in solver_config_block
    assert 'self.check_server_button = QPushButton("Check Server")' in dialog_source
    assert 'self.generate_server_access_token_button = QPushButton("Generate")' in dialog_source
    assert 'self.copy_server_access_token_button = QPushButton("Copy")' in dialog_source
    assert '"Server access token",' in solver_config_block
    assert "Keep this code safe" in solver_config_block
    assert "operating system credential vault" not in dialog_source
    assert "self.server_access_token_row.setEnabled(uses_remote)" in dialog_source
    assert '"", self.check_server_button' in solver_config_block or "self.check_server_button," in solver_config_block
    assert "self.check_server_button.setEnabled(uses_remote)" in dialog_source
    assert '"BEM Solver", self.solve_backend_combo' not in application_block
    assert '"Solve Server URL", self.solve_server_url_edit' not in application_block
    assert '"Solve Backend", self.solve_backend_combo' not in dialog_source
    assert 'uses_bempp = backend_id in {"local", "server"}' not in dialog_source
    # Server health probing moved to BackendHealthController and is covered
    # behaviourally by tests/test_backend_health.py.
    assert "QTimer.singleShot(0, self.backend_health.check_on_startup)" in main_source
    assert "GMRES Tolerance" not in dialog_source
    assert "Burton Miller Formulation" not in dialog_source
    assert "self.gmres_spin" not in dialog_source
    assert "self.burton_miller_check" not in dialog_source
    assert '"Balloon Sampling",\n                        self.spherical_sampling_check,' in dialog_source
    assert '"Balloon Angle Precision",\n                        self.balloon_angle_precision_spin,' in dialog_source
    assert "Gather spherical observation data for 3d ballon viewer" in dialog_source
    assert (
        '"Normalized Channel Correction",\n                        self.normalized_channel_correction_check,'
        in dialog_source
    )
    assert (
        "Applies a per-channel reference-axis magnitude correction before channel gain, delay, and crossover filters."
        in dialog_source
    )
    assert '"preferences/normalized_channel_correction"' in settings_source
    assert "normalized_channel_correction: bool = True" in settings_source
    assert "flat_target_normalization_enabled=preferences.normalized_channel_correction" in main_source
    assert '"preferences/live_plot_quality"' in settings_source
    assert '"preferences/live_plot_streaming"' in settings_source
    assert '"preferences/isobar_contour_step_db"' in settings_source
    assert "live_plot_streaming: bool = True" in settings_source
    assert "isobar_contour_step_db: float = 3.0" in settings_source
    assert '"Isobar Contour Step"' in dialog_source
    assert "live_plot_angle_samples(self.preferences.live_plot_quality)" in main_source
    assert "live_plot_freq_samples(self.preferences.live_plot_quality)" in main_source
    solve_source = main_window_source("solve_workflow")
    # Every solve path opens by discarding the previous run's results.
    assert "def _begin_run(" in solve_source
    assert "self._plots.clear_plots()" in solve_source
    assert "if not self._read_preferences().live_plot_streaming:" in solve_source
    assert "if self._read_preferences().live_plot_streaming or solve_completed:" in solve_source
    assert "FINAL_ISOBAR_ANGLE_SAMPLES = 1000" in plot_source
    assert "FINAL_ISOBAR_FREQ_SAMPLES = 500" in plot_source
    assert 'LIVE_ISOBAR_SHADING = "nearest"' in plot_source
    assert 'FINAL_ISOBAR_SHADING = "gouraud"' in plot_source
    assert "session.use_final_isobar_resolution = solve_completed" in solve_source
    # _on_solve_finished is the last method in the module, so this runs to the end.
    # Slicing the whole package instead would silently produce an empty string.
    solve_finished = solve_source[solve_source.index("def _on_solve_finished") :]
    assert solve_finished, "the slice must not be empty, or the next assertion is vacuous"
    assert "QApplication.processEvents()" not in solve_finished
    assert "angle_samples=FINAL_ISOBAR_ANGLE_SAMPLES" in main_source
    assert "freq_samples=FINAL_ISOBAR_FREQ_SAMPLES" in main_source
    assert (
        'angle_samples=FINAL_ISOBAR_ANGLE_SAMPLES if plot_id in {"horizontal_isobar", "vertical_isobar"} else None'
        in main_source
    )
    assert (
        'freq_samples=FINAL_ISOBAR_FREQ_SAMPLES if plot_id in {"horizontal_isobar", "vertical_isobar"} else None'
        in main_source
    )
    assert "shading=FINAL_ISOBAR_SHADING if self._use_final_isobar_resolution else LIVE_ISOBAR_SHADING" in main_source
    assert "contour_step_db=self.preferences.isobar_contour_step_db" in main_source


def test_isobar_canvas_allows_custom_right_margin() -> None:
    source = source_text("ui", "plots.py")

    assert "left_margin: float | None = None" in source
    assert "right_margin: float | None = None" in source
    assert "show_colorbar: bool = True" in source
    assert "self.left_margin = PLOT_LEFT_MARGIN if left_margin is None else float(left_margin)" in source
    assert "self.right_margin = PLOT_RIGHT_MARGIN if right_margin is None else float(right_margin)" in source
    assert "left=self.left_margin" in source
    assert "right=self.right_margin" in source


def test_isobar_canvas_reuses_heatmap_artist_between_grid_changes() -> None:
    source = source_text("ui", "plots.py")
    isobar_block = source[source.index("class IsobarCanvas") : source.index("class ImpedanceCanvas")]

    assert "self._mesh_artist" in isobar_block
    assert "self._image_artist" in isobar_block
    assert "self._colorbar" in isobar_block
    assert "self._colorbar_axes" in isobar_block
    assert "self.show_colorbar = bool(show_colorbar)" in isobar_block
    assert "def _mesh_matches(" in isobar_block
    assert "self._mesh_artist.set_array" in isobar_block
    assert "self.axes.pcolormesh(" in isobar_block
    assert "shading=shading" in isobar_block
    assert "self._mesh_shading == shading" in isobar_block
    assert "self._mesh_contour_step_db == contour_step_db" in isobar_block
    assert 'render_mode = "image" if shading == FINAL_ISOBAR_SHADING else "mesh"' in isobar_block
    assert "self.axes.imshow(" in isobar_block
    assert 'interpolation="bilinear"' in isobar_block
    assert "np.log10(freqs_hz)" in isobar_block
    assert "LinearSegmentedColormap.from_list" in isobar_block
    assert "Normalize(vmin=clip_min_db, vmax=clip_max_db)" in isobar_block
    assert "BoundaryNorm(boundaries, cmap.N)" in isobar_block
    assert "ScalarMappable(norm=norm, cmap=cmap)" in isobar_block
    assert "self.figure.add_axes" in isobar_block
    assert "cax=self._colorbar_axes" in isobar_block
    assert "ISOBAR_COLORBAR_MIN_TICK_STEP_DB = 3.0" in source
    assert "tick_step_db = max(ISOBAR_COLORBAR_MIN_TICK_STEP_DB, contour_step_db)" in isobar_block
    assert "if not self.show_colorbar" in isobar_block
    assert "apply_log_image_frequency_axis" in source
    assert isobar_block.count("clear_plot_axes(self.axes)") == 1


def test_isobar_canvas_captures_and_redraws_persistent_contours() -> None:
    source = source_text("ui", "plots.py")
    isobar_block = source[source.index("class IsobarCanvas") : source.index("class ImpedanceCanvas")]
    draw_empty_block = isobar_block[
        isobar_block.index("    def _draw_empty") : isobar_block.index("    def _remove_artist")
    ]
    remove_contour_block = isobar_block[
        isobar_block.index("    def _remove_contour_artist") : isobar_block.index("    @property")
    ]

    assert "self._captured_contours" in isobar_block
    assert "self._mesh_values_db" in isobar_block
    assert "def capture_contours(" in isobar_block
    assert "def clear_contours(" in isobar_block
    assert "def _redraw_captured_contours(" in isobar_block
    assert "np.ceil(clip_min_db / contour_step_db) * contour_step_db" in isobar_block
    assert "if contour_step_db <= 0.0" in isobar_block
    assert "levels.copy()" in isobar_block
    assert 'colors="white"' in isobar_block
    assert "linewidths=0.9" in isobar_block
    assert 'linestyles="solid"' in isobar_block
    assert "alpha=0.85" in isobar_block
    assert "self._redraw_captured_contours()" in isobar_block
    assert "self._captured_contours = None" not in draw_empty_block
    assert "self._contour_artist = None" in draw_empty_block
    assert "NotImplementedError" in remove_contour_block


def test_isobar_canvas_has_click_drag_crosshair_readout() -> None:
    source = source_text("ui", "plots.py")
    isobar_block = source[source.index("class IsobarCanvas") : source.index("class ImpedanceCanvas")]
    interaction_block = source[source.index("class InteractivePlotCanvas") : source.index("class IsobarCanvas")]

    assert "from matplotlib.backend_bases import MouseButton" in source
    assert "class IsobarCanvas(InteractivePlotCanvas)" in source
    assert "self._crosshair_visible = False" in interaction_block
    assert "self._connect_interaction_events()" in interaction_block
    assert "button_press_event" in interaction_block
    assert "motion_notify_event" in interaction_block
    assert "button_release_event" in interaction_block
    assert "event.button != MouseButton.LEFT" in interaction_block
    assert 'getattr(event, "dblclick", False)' in interaction_block
    assert "self._hide_crosshair()" in interaction_block
    assert "self.axes.axvline(" in isobar_block
    assert "self.axes.axhline(" in isobar_block
    assert "self.axes.get_xaxis_transform()" in isobar_block
    assert "self.axes.get_yaxis_transform()" in isobar_block
    assert "self.axes.annotate(" in isobar_block
    assert "_format_isobar_crosshair_frequency" in source
    assert 'f"{int(round(self._crosshair_angle_deg)):+d} deg"' in isobar_block
    assert 'f"{db_value:.1f} dB"' in isobar_block
    assert "self._crosshair_db_label.xy = (axis_x, self._crosshair_angle_deg)" in isobar_block
    assert "if self._crosshair_visible:" in isobar_block
    assert "self._redraw_crosshair()" in isobar_block
    assert "draw_event" in interaction_block
    assert "self.copy_from_bbox(self.figure.bbox)" in interaction_block
    assert "self.restore_region(self._crosshair_background)" in interaction_block
    assert "artist.axes.draw_artist(artist)" in interaction_block
    assert "self.blit(self.figure.bbox)" in interaction_block
    assert "artist.set_animated(True)" in isobar_block
    assert "_grid_bracket(log_freqs, log_freq)" in isobar_block


def test_isobar_canvas_has_hold_right_button_previous_solve_comparison() -> None:
    plot_source = source_text("ui", "plots.py")
    main_source = main_window_source()
    state_source = source_text("ui", "application_state.py")
    isobar_block = plot_source[plot_source.index("class IsobarCanvas") : plot_source.index("class ImpedanceCanvas")]
    interaction_block = plot_source[
        plot_source.index("class InteractivePlotCanvas") : plot_source.index("class IsobarCanvas")
    ]

    assert "self._comparison_plot" in interaction_block
    assert "def set_comparison_plot(" in isobar_block
    assert "def clear_comparison_plot(" in interaction_block
    assert "event.button != MouseButton.RIGHT" in interaction_block
    assert 'f"{self.title} - Previous Solve"' in interaction_block
    assert "self._apply_plot_state(self._comparison_plot)" in interaction_block
    assert "self._apply_plot_state(restore_plot)" in interaction_block
    assert "self.axes.set_title(self.title, pad=PLOT_TITLE_PAD)" in interaction_block
    assert 'self.mpl_connect("figure_leave_event", self._on_figure_leave)' in interaction_block
    assert "self._last_completed_visualization_dataset" in main_source
    assert "refreshed_dataset.snapshot()" in main_source
    assert "self._plots.apply_last_completed_comparison()" in main_source
    assert "self.impedance_plot.set_comparison_plot(" in main_source
    assert "self.on_axis_plot.set_comparison_plot(" in main_source
    assert "self.spinorama_plot.set_comparison_plot(" in main_source
    assert "SolveInvalidationReason.NEW_PROJECT" in state_source
    assert "clear_comparison_history=True" in state_source


def test_geometry_generation_uses_registry_worker_and_delayed_stop_button() -> None:
    main_source = main_window_source()
    worker_source = source_text("ui", "generator_worker.py")
    controller_source = source_text("ui", "operation_controllers.py")

    assert "GeometryController" in main_source
    assert "GenerationRequest(" in main_source
    assert "self.cancel_button.clicked.connect(self.cancel_current_operation)" in main_source
    from blab.ui.main_window.geometry_workflow import CANCEL_DELAY_MS

    assert CANCEL_DELAY_MS == 3000, "Stop stays hidden for the first three seconds of a generation"
    assert "QTimer.singleShot(CANCEL_DELAY_MS, self._enable_geometry_cancel_if_active)" in main_source
    assert "def cancel_geometry_generation(" in main_source
    assert "self._geometry_controller.cancel()" in main_source
    assert "self._worker: GeneratorWorker | None = None" in controller_source
    assert "worker.moveToThread(thread)" in controller_source
    assert "self._worker.stop()" in controller_source
    assert "class GeneratorWorker(QObject)" in worker_source
    assert "create_generator(self.request.provider_id" in worker_source
