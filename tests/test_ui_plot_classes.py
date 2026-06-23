from pathlib import Path


def test_on_axis_and_spinorama_canvases_keep_distinct_update_signatures() -> None:
    source = Path("src/blab/ui/plots.py").read_text(encoding="utf-8")

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
    source = Path("src/blab/ui/plots.py").read_text(encoding="utf-8")
    spinorama_block = source[source.index("class SpinoramaCanvas") :]

    assert "tight_layout=True" not in spinorama_block
    assert "subplots_adjust" in spinorama_block
    assert "left=0.14" in spinorama_block
    assert 'set_label_position("right")' in spinorama_block
    assert "bbox_to_anchor=(0.5, -0.2)" in spinorama_block
    assert "SPINORAMA_SPL_LIMITS" in spinorama_block
    assert "SPINORAMA_DI_LIMITS" in spinorama_block
    assert "ncols=4" in spinorama_block


def test_plot_widgets_use_compact_title_padding() -> None:
    plot_source = Path("src/blab/ui/plots.py").read_text(encoding="utf-8")

    assert "PLOT_TITLE_PAD = 1" in plot_source
    assert "GRID_LINE_ALPHA = 0.6" in plot_source
    assert "set_title(self.title, pad=PLOT_TITLE_PAD)" in plot_source
    assert "set_yticks(np.arange(-180, 181, 45))" in plot_source
    assert 'grid(which="major", color="#808080", linewidth=0.8, alpha=GRID_LINE_ALPHA)' in plot_source


def test_main_window_uses_detachable_panel_docks() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")
    widgets_source = Path("src/blab/ui/main_window_widgets.py").read_text(encoding="utf-8")

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


def test_plot_export_uses_dock_title_save_buttons_not_file_menu() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")

    assert 'file_menu.addMenu("Export Plot")' not in source
    assert "self.export_plot_actions[entry.plot_id] = action" in source
    assert 'action.setToolTip(f"Export {entry.title}")' in source
    assert "SAVE_DARK_ICON" in source
    assert "SAVE_LIGHT_ICON" in source
    assert "CAPTURE_CONTOURS_DARK_ICON" in source
    assert "CAPTURE_CONTOURS_LIGHT_ICON" in source
    assert "CLEAR_CONTOURS_DARK_ICON" in source
    assert "CLEAR_CONTOURS_LIGHT_ICON" in source
    assert "def _refresh_plot_export_icons(" in source
    assert "light_theme = window_color.lightness() >= 128" in source
    assert "QIcon(str(SAVE_LIGHT_ICON if light_theme else SAVE_DARK_ICON))" in source
    assert "action.setIcon(icon)" in source
    assert "action.setIcon(capture_icon)" in source
    assert "action.setIcon(clear_icon)" in source
    assert "self._refresh_plot_export_icons()" in source


def test_live_plot_refresh_is_immediate_and_visibility_aware() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")

    assert "_request_plot_refresh" not in source
    assert "_plot_refresh_timer" not in source
    assert "self.plot_docks.get(entry.plot_id)" in source
    assert "not dock.isHidden()" in source
    assert "for entry in visible_entries:" in source
    assert "self._refresh_plots()" in source


def test_channel_config_changes_apply_only_on_apply_button() -> None:
    dialog_source = Path("src/blab/ui/dialogs.py").read_text(encoding="utf-8")
    main_source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")
    channel_dialog = dialog_source[
        dialog_source.index("class ChannelConfigDialog") : dialog_source.index("class SourceConfigDialog")
    ]

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


def test_invalidating_user_config_changes_confirm_before_clearing_solved_data() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")
    confirm_block = source[
        source.index("def _confirm_clear_solved_data") : source.index(
            "    @Slot(str)", source.index("def _confirm_clear_solved_data")
        )
    ]
    preferences_block = source[source.index("def open_preferences") : source.index("def open_diagnostics")]
    mesh_block = source[source.index("def open_mesh_config") : source.index("def open_channel_config")]
    channel_block = source[
        source.index("def _apply_channel_config") : source.index(
            "    @Slot()", source.index("def _apply_channel_config")
        )
    ]
    source_block = source[source.index("def open_source_config") : source.index("def generate_geometry")]

    assert "Applying this action will clear solved data" in confirm_block
    assert 'message.addButton("Continue", QMessageBox.AcceptRole)' in confirm_block
    assert 'message.addButton("Cancel", QMessageBox.RejectRole)' in confirm_block
    assert "message.setDefaultButton(cancel_button)" in confirm_block
    assert "if requires_invalidation and not self._confirm_clear_solved_data():" in preferences_block
    assert "if not config_changed:" in mesh_block
    assert "if not self._confirm_clear_solved_data():" in mesh_block
    assert "if not channel_config_changed and not radiator_assignments_changed:" in channel_block
    assert "if not can_resynthesize and not self._confirm_clear_solved_data():" in channel_block
    assert "if radiators == self._all_radiators():" in source_block
    assert "if not self._confirm_clear_solved_data():" in source_block


def test_application_startup_invokes_new_project_reset() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")
    init_block = source[source.index("    def __init__(") : source.index("    def changeEvent")]

    assert 'startup("Starting new project...")' in init_block
    assert "self.new_project()" in init_block
    assert "self._project_clean_payload: dict | None = None" in init_block
    assert "_load_initial_ath_scripts" not in source
    assert "_load_imported_meshes" not in source
    assert "mesh/imported_meshes" not in source
    assert "mesh/ath_mesh" not in source


def test_unsaved_project_changes_guard_close_new_and_open() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")
    close_block = source[source.index("def closeEvent") : source.index("def _result_from_script_state")]
    new_block = source[source.index("def new_project") : source.index("def save_project")]
    save_block = source[source.index("def save_project") : source.index("def load_project")]
    load_block = source[source.index("def load_project") : source.index("def _project_payload")]
    confirm_block = source[
        source.index("def _confirm_unsaved_project_changes") : source.index("def _apply_project_payload")
    ]

    assert 'if not self._confirm_unsaved_project_changes("close"):' in close_block
    assert "event.ignore()" in close_block
    assert 'if not self._confirm_unsaved_project_changes("new_project"):' in new_block
    assert "self._mark_project_clean()" in new_block
    assert "def save_project(self) -> bool:" in save_block
    assert "def save_project_as(self) -> bool:" in save_block
    assert "return False" in save_block
    assert "def _save_project_to_path(self, path: Path) -> bool:" in save_block
    assert "self._mark_project_clean()" in save_block
    assert 'if not self._confirm_unsaved_project_changes("open_project"):' in load_block
    assert "self._mark_project_clean()" in load_block
    assert "You have unsaved changes. Are you sure you want to close?" in confirm_block
    assert "You have unsaved changes. Save before continuing?" in confirm_block
    assert 'message.addButton("Save", QMessageBox.AcceptRole)' in confirm_block
    assert 'message.addButton("Discard", QMessageBox.DestructiveRole)' in confirm_block
    assert 'message.addButton("Cancel", QMessageBox.RejectRole)' in confirm_block
    assert "message.setDefaultButton(cancel_button)" in confirm_block


def test_plot_canvases_refresh_backing_store_on_screen_dpi_changes() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")
    init_block = source[source.index("    def __init__(") : source.index("    def changeEvent")]
    screen_block = source[source.index("def showEvent") : source.index("    def eventFilter")]

    assert "self._plot_dpi_screen = None" in init_block
    assert "self._plot_dpi_window_handle = None" in init_block
    assert "self._plot_dpi_refresh_pending = False" in init_block
    assert "window.screenChanged.connect(self._on_plot_screen_changed)" in screen_block
    assert "screen.logicalDotsPerInchChanged.connect(self._schedule_plot_canvas_dpi_refresh)" in screen_block
    assert "screen.physicalDotsPerInchChanged.connect(self._schedule_plot_canvas_dpi_refresh)" in screen_block
    assert "screen.geometryChanged.connect(self._schedule_plot_canvas_dpi_refresh)" in screen_block
    assert "QTimer.singleShot(0, self._refresh_plot_canvas_dpi)" in screen_block
    assert "canvas._update_screen(screen)" in screen_block
    assert "canvas._update_pixel_ratio()" in screen_block
    assert "canvas.draw_idle()" in screen_block
    assert "_refresh_plots()" not in screen_block
    assert "_prepared_live_plot_dataset" not in screen_block


def test_server_health_worker_runs_query_with_timeout() -> None:
    worker_source = (Path(__file__).resolve().parents[1] / "src" / "blab" / "ui" / "server_health_worker.py").read_text(
        encoding="utf-8"
    )

    assert "class ServerHealthCheckWorker(QObject)" in worker_source
    assert "succeeded = Signal(str, object)" in worker_source
    assert "failed = Signal(str)" in worker_source
    assert "query_server_health(self.server_url, timeout_s=self.timeout_s)" in worker_source
    assert "self.finished.emit()" in worker_source


def test_preferences_no_longer_expose_worker_count() -> None:
    dialog_source = Path("src/blab/ui/dialogs.py").read_text(encoding="utf-8")
    settings_source = Path("src/blab/ui/settings.py").read_text(encoding="utf-8")
    main_source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")
    config_source = Path("src/blab/config.py").read_text(encoding="utf-8")
    start_solve = main_source[
        main_source.index("def start_solve") : main_source.index("    @Slot()", main_source.index("def start_solve"))
    ]

    assert "worker_count_spin" not in dialog_source
    assert '"Worker Count"' not in dialog_source
    assert "worker_count:" not in settings_source
    assert '"preferences/worker_count"' not in settings_source
    assert "preferences.worker_count" not in main_source
    assert "workers=1" in start_solve
    assert "worker_count=1" in start_solve
    assert "workers: int = 1" in config_source


def test_completed_solves_use_final_isobar_resolution() -> None:
    plot_source = Path("src/blab/ui/plots.py").read_text(encoding="utf-8")
    main_source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")
    dialog_source = Path("src/blab/ui/dialogs.py").read_text(encoding="utf-8")
    settings_source = Path("src/blab/ui/settings.py").read_text(encoding="utf-8")

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
    assert '"", self.check_server_button' in solver_config_block or 'self.check_server_button,' in solver_config_block
    assert 'self.check_server_button.setEnabled(uses_remote)' in dialog_source
    assert '"BEM Solver", self.solve_backend_combo' not in application_block
    assert '"Solve Server URL", self.solve_server_url_edit' not in application_block
    assert '"Solve Backend", self.solve_backend_combo' not in dialog_source
    assert 'uses_bempp = backend_id in {"local", "server"}' in dialog_source
    assert "self.server_health_payload: dict | None = None" in main_source
    assert "self.server_health_thread: QThread | None = None" in main_source
    assert "QTimer.singleShot(0, self._check_configured_server_health_on_startup)" in main_source
    assert "def _backend_supports_symmetry(" in main_source
    assert "server_health_supports_symmetry" in main_source
    assert "def _check_configured_server_health_on_startup" in main_source
    assert 'self.preferences.solve_backend != "server"' in main_source
    assert "ServerHealthCheckWorker(self.preferences.solve_server_url, timeout_s=5.0)" in main_source
    assert "worker.failed.connect(lambda _message: None)" in main_source
    assert 'self.mesh_state_changed.emit("server_health_checked")' in main_source
    assert "self.gmres_spin.setEnabled(uses_bempp)" in dialog_source
    assert "self.burton_miller_check.setEnabled(uses_bempp)" in dialog_source
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
    assert "flat_target_normalization_enabled=self.preferences.normalized_channel_correction" in main_source
    assert '"preferences/live_plot_quality"' in settings_source
    assert '"preferences/live_plot_streaming"' in settings_source
    assert '"preferences/isobar_contour_step_db"' in settings_source
    assert "live_plot_streaming: bool = True" in settings_source
    assert "isobar_contour_step_db: float = 3.0" in settings_source
    assert '"Isobar Contour Step"' in dialog_source
    assert "live_plot_angle_samples(self.preferences.live_plot_quality)" in main_source
    assert "live_plot_freq_samples(self.preferences.live_plot_quality)" in main_source
    start_solve = main_source[
        main_source.index("def start_solve") : main_source.index("    @Slot()", main_source.index("def start_solve"))
    ]
    assert "self.live_dataset = None\n        self._clear_plots()" in start_solve
    assert "if not self.preferences.live_plot_streaming:" in main_source
    assert "if self.preferences.live_plot_streaming or solve_completed:" in main_source
    assert "FINAL_ISOBAR_ANGLE_SAMPLES = 1000" in plot_source
    assert "FINAL_ISOBAR_FREQ_SAMPLES = 500" in plot_source
    assert 'LIVE_ISOBAR_SHADING = "nearest"' in plot_source
    assert 'FINAL_ISOBAR_SHADING = "gouraud"' in plot_source
    assert "self._use_final_isobar_resolution = solve_completed" in main_source
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
    source = Path("src/blab/ui/plots.py").read_text(encoding="utf-8")

    assert "left_margin: float = 0.14" in source
    assert "right_margin: float = 0.88" in source
    assert "show_colorbar: bool = True" in source
    assert "left=self.left_margin" in source
    assert "right=self.right_margin" in source


def test_isobar_canvas_reuses_heatmap_artist_between_grid_changes() -> None:
    source = Path("src/blab/ui/plots.py").read_text(encoding="utf-8")
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
    source = Path("src/blab/ui/plots.py").read_text(encoding="utf-8")
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


def test_main_window_contour_buttons_are_final_render_and_visibility_gated() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")

    assert "self.capture_contour_actions: dict[str, QAction]" in source
    assert "self.clear_contour_actions: dict[str, QAction]" in source
    assert 'capture_action = QAction("Capture Contours", self)' in source
    assert 'clear_action = QAction("Clear Contours", self)' in source
    assert "self.capture_contour_actions[entry.plot_id] = capture_action" in source
    assert "self.clear_contour_actions[entry.plot_id] = clear_action" in source
    assert "self.capture_contours_button" not in source
    assert "self.clear_contours_button" not in source
    assert "self._final_isobar_plots_rendered = False" in source
    assert "self._final_isobar_plots_rendered = solve_completed and bool(self._visible_isobar_plots())" in source
    assert "self._use_final_isobar_resolution" in source
    assert "and self._final_isobar_plots_rendered" in source
    assert "capture_action.setEnabled(capture_base_enabled and visible)" in source
    assert "clear_action.setEnabled(plot.has_captured_contours)" in source


def test_main_window_captures_and_clears_contours_per_isobar_plot() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")

    assert "def _visible_isobar_plots(" in source
    assert "def _sync_plot_view_action(" in source
    assert "def capture_isobar_contours(" in source
    assert "def clear_isobar_contours(" in source
    assert "def _isobar_plot_for_id(" in source
    assert 'self.plot_docks.get("horizontal_isobar")' in source
    assert 'self.plot_docks.get("vertical_isobar")' in source
    assert "action.setChecked(not dock.isHidden())" in source
    assert "plot = self._isobar_plot_for_id(plot_id)" in source
    assert "plot.capture_contours()" in source
    assert "plot.clear_contours()" in source


def test_balloon_contours_exclude_configured_maximum() -> None:
    source = Path("src/blab/ui/balloon.py").read_text(encoding="utf-8")

    assert "if min_db < level < max_db" in source


def test_balloon_window_uses_dockable_widgets_and_bottom_controls() -> None:
    source = Path("src/blab/ui/balloon.py").read_text(encoding="utf-8")

    assert "class BalloonPlotWindow(QMainWindow):" in source
    assert 'view_menu = menu_bar.addMenu("View")' in source
    assert "self.workspace = QMainWindow()" in source
    assert "QMainWindow.AllowNestedDocks" in source
    assert "workspace_placeholder.setMaximumSize(QSize(0, 0))" in source
    assert "workspace_placeholder.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)" in source
    assert "self.workspace.setCentralWidget(workspace_placeholder)" in source
    assert 'controls_bar = QFrame()' in source
    assert 'layout.addWidget(self.workspace, stretch=1)' in source
    assert 'layout.addWidget(controls_bar)' in source
    assert 'controls_layout = QGridLayout(controls_bar)' in source
    assert 'controls_bar.setStyleSheet' not in source
    assert 'controls_layout.addWidget(QLabel("Frequency"), 0, 0)' in source
    assert 'controls_layout.addWidget(self.frequency_slider, 0, 1)' in source
    assert 'controls_layout.addWidget(self.protractor_toggle, 0, 3)' in source
    assert 'controls_layout.addWidget(QLabel("Slice Angle"), 1, 0)' in source
    assert 'controls_layout.addWidget(self.protractor_angle_slider, 1, 1)' in source
    assert 'self.balloon_dock = self._make_dock("3D Balloon Plot", viewport)' in source
    assert 'self.radar_dock = self._make_dock("Radar Slicer Plot", self.radar_plot)' in source
    assert 'self.wavefront_shape_dock = self._make_dock(' in source
    assert '"Forward Beam Shape",' in source
    assert 'self.wavefront_shape_plot,' in source
    assert 'tool_actions=(self.save_wavefront_shape_action,),' in source
    assert 'self.isobar_dock = self._make_dock(' in source
    assert 'view_menu.addAction(self.balloon_dock.toggleViewAction())' in source
    assert 'view_menu.addAction(self.radar_dock.toggleViewAction())' in source
    assert 'view_menu.addAction(self.wavefront_shape_dock.toggleViewAction())' in source
    assert 'view_menu.addAction(self.isobar_dock.toggleViewAction())' in source
    assert 'self.wavefront_shape_dock.hide()' in source
    assert 'self.workspace.addDockWidget(Qt.LeftDockWidgetArea, self.balloon_dock)' in source
    assert 'self.workspace.addDockWidget(Qt.RightDockWidgetArea, self.radar_dock)' in source
    assert 'self.workspace.addDockWidget(Qt.RightDockWidgetArea, self.wavefront_shape_dock)' in source
    assert 'self.wavefront_shape_dock.visibilityChanged.connect(self._on_wavefront_shape_visibility_changed)' in source
    assert 'self.workspace.splitDockWidget(self.balloon_dock, self.radar_dock, Qt.Horizontal)' in source
    assert 'self.workspace.splitDockWidget(self.radar_dock, self.wavefront_shape_dock, Qt.Vertical)' in source
    assert 'self.workspace.splitDockWidget(self.wavefront_shape_dock, self.isobar_dock, Qt.Vertical)' in source
    assert 'self.workspace.resizeDocks(' in source
    assert '[self.balloon_dock, self.radar_dock, self.wavefront_shape_dock, self.isobar_dock]' in source
    assert "QSplitter" not in source


def test_balloon_window_does_not_use_rendering_overlay() -> None:
    source = Path("src/blab/ui/balloon.py").read_text(encoding="utf-8")

    assert "Rendering Balloon" not in source
    assert "loading_label" not in source
    assert "_set_loading_visible" not in source

def test_balloon_viewport_polish_removes_redundant_axes_and_styles_readout() -> None:
    source = Path("src/blab/ui/balloon.py").read_text(encoding="utf-8")

    assert "QLabel {background: #2d2d30;color: #e8e8e8;padding-left: 8px;padding-right: 8px;}" in source
    assert "self.plotter.add_axes()" not in source
    assert "self._axes_added" not in source


def test_balloon_window_has_wavefront_shape_dock_and_fit_helpers() -> None:
    source = Path("src/blab/ui/balloon.py").read_text(encoding="utf-8")

    assert 'WAVEFRONT_LEVEL_DB = -6.0' in source
    assert 'class WavefrontShapeCanvas(FigureCanvas):' in source
    assert 'Forward Beam Shape' in source
    assert 'Wavefront Shape' not in source
    assert 'def _wavefront_shape_summary(' in source
    assert 'def _fit_wavefront_shape_for_frequency(' in source
    assert 'LinearNDInterpolator' in source
    assert 'minimize_scalar' in source
    assert 'self.wavefront_shape_plot.update_plot(_wavefront_shape_summary(self._prepared))' not in source
    assert 'def _on_wavefront_shape_visibility_changed(self, visible: bool) -> None:' in source
    assert 'def _render_wavefront_shape_plot(self) -> None:' in source
    assert 'self._wavefront_shape_summary_cache = None' in source
    assert 'self._wavefront_shape_summary_cache = _wavefront_shape_summary(' in source
    assert 'raw_balloon_data=self._raw_balloon_data' in source
    assert 'self.save_wavefront_shape_action = QAction("Save Plot Image", self)' in source
    assert 'self.save_wavefront_shape_action.triggered.connect(self._save_wavefront_shape_image)' in source
    assert 'self.save_wavefront_shape_action.setIcon' in source
    assert 'tool_actions=(self.save_wavefront_shape_action,)' in source
    assert 'def _save_wavefront_shape_image(self) -> None:' in source
    assert 'str(Path.cwd() / "forward_beam_shape.png")' in source
    assert 'export_plot_png(self.wavefront_shape_plot.figure, output_path, dpi=VisualizerConfig.figure_dpi)' in source
    assert 'self._update_wavefront_shape_frequency_cursor(index)' in source
    assert 'def _update_wavefront_shape_frequency_cursor(self, index: int) -> None:' in source
    assert 'self.wavefront_shape_plot.set_frequency_cursor(float(self._prepared["freq_hz"][safe_index]))' in source
    assert 'def set_frequency_cursor(self, freq_hz: float | None) -> None:' in source
    assert 'self.axes.axvline(' in source
    assert 'self._colorbar_axes = self.figure.add_axes([0.89, 0.22, 0.025, 0.68])' in source
    assert 'self._colorbar = self.figure.colorbar(scatter, cax=self._colorbar_axes)' in source
    assert 'self._colorbar.set_label("Fit residual (%)", fontsize=8)' in source
    assert 'self.di_axes = self.axes.twinx()' in source
    assert 'self.di_axes.set_ylabel("Spherical DI (dB)", labelpad=10)' in source
    assert 'self.di_axes.set_ylim(-5.0, 50.0)' in source
    assert 'self.di_axes.yaxis.set_label_position("right")' in source
    assert 'self.di_axes.yaxis.tick_right()' in source
    assert 'self.di_axes.yaxis.set_label_coords(1.17, 0.5)' in source
    assert 'subplots_adjust(left=0.18, right=0.74' in source
    assert 'def _plot_directivity_index(self, freqs: np.ndarray, directivity_index: np.ndarray) -> None:' in source
    assert 'Normalize(vmin=0.0, vmax=15.0, clip=True)' in source
    assert '"directivity_index_db": _spherical_directivity_index_db(prepared, raw_balloon_data)' in source
    assert 'def _spherical_directivity_index_from_raw(' in source
    assert 'def _style_colorbar(self) -> None:' in source
    assert 'self._colorbar.ax.yaxis.label.set_color(text_color)' in source
    assert 'self._colorbar.ax.tick_params(colors=text_color)' in source
    assert 'self._colorbar.outline.set_edgecolor(spine_color)' in source
    assert 'def _draw_shape_reference_markers(self) -> None:' in source
    assert '(1.0, "D")' in source
    assert '(2.0, "o")' in source
    assert '(4.0, _rounded_square_marker())' in source
    assert '(8.0, "s")' in source
    assert 'def _rounded_square_marker() -> MplPath:' in source
    assert 'self.axes.set_ylim(0.75, 8.5)' in source
    assert 'transform = self.axes.get_yaxis_transform()' in source
    assert 'marker_color = self._theme_marker_color()' in source
    assert 'facecolors="none"' in source
    assert 'edgecolors=marker_color' in source
    assert 'def _theme_marker_color(self) -> str:' in source
    assert '"#101214" if self.palette().color(QPalette.Window).lightness() >= 128 else "#f2f2f2"' in source
    assert 'clip_on=False' in source


def test_balloon_slice_plot_has_hires_render_and_save_actions() -> None:
    source = Path("src/blab/ui/balloon.py").read_text(encoding="utf-8")

    assert 'HIRES_RENDER_DARK_ICON = APP_ROOT / "assets" / "hiresrender_dark.ico"' in source
    assert 'HIRES_RENDER_LIGHT_ICON = APP_ROOT / "assets" / "hiresrender_light.ico"' in source
    assert 'self.hires_slice_action = QAction("Render High Resolution", self)' in source
    assert 'self.hires_slice_action.setToolTip("Render high resolution plot")' in source
    assert 'self.save_slice_action = QAction("Save Plot Image", self)' in source
    assert 'self.save_slice_action.setToolTip("Save plot image")' in source
    assert 'self.hires_slice_action.triggered.connect(self._render_high_resolution_isobar_slice)' in source
    assert 'show_colorbar=False' in source
    assert 'self.save_slice_action.triggered.connect(self._save_isobar_slice_image)' in source
    assert 'tool_actions=(self.hires_slice_action, self.save_slice_action)' in source
    assert 'DockTitleBar(title, dock, tool_actions=tool_actions)' in source
    assert 'self.hires_slice_action.setIcon' in source
    assert 'self.save_slice_action.setIcon' in source
    assert 'self.save_wavefront_shape_action.setIcon' in source
    assert 'def _render_isobar_slice(self, *, final_resolution: bool = False)' in source
    assert 'angle_samples=FINAL_ISOBAR_ANGLE_SAMPLES if final_resolution else LIVE_ISOBAR_ANGLE_SAMPLES' in source
    assert 'freq_samples=FINAL_ISOBAR_FREQ_SAMPLES if final_resolution else LIVE_ISOBAR_FREQ_SAMPLES' in source
    assert 'shading=FINAL_ISOBAR_SHADING if final_resolution else LIVE_ISOBAR_SHADING' in source
    assert 'self._render_isobar_slice(final_resolution=True)' in source
    assert 'export_plot_png(self.slice_plot.figure, output_path, dpi=VisualizerConfig.figure_dpi)' in source


def test_balloon_spl_legend_lives_in_bottom_control_bar() -> None:
    source = Path("src/blab/ui/balloon.py").read_text(encoding="utf-8")

    assert 'self.spl_legend = ColorLegend(self._min_db, self._max_db, orientation="horizontal")' in source
    assert 'controls_layout.addWidget(self.spl_legend, 0, 4, 2, 1)' in source
    assert 'controls_layout.setColumnStretch(4, 1)' in source
    assert 'self.legend_overlay' not in source
    assert 'legend_layout.addWidget(self.spl_legend)' not in source
    assert 'self.spl_legend.set_range(self._min_db, self._max_db)' in source
    assert 'orientation: str = "vertical"' in source
    assert 'def _paint_horizontal(self) -> None:' in source
    assert "self.setMinimumSize(330, 62)" in source
    assert "label_edge_pad = 22" in source
    assert "bar_left = label_edge_pad" in source
    assert "bar_top = 28" in source
    assert "bar_width = max(self.width() - 2 * label_edge_pad, 40)" in source
    assert "label_x = int(round(np.clip" in source
    assert 'gradient = QLinearGradient(bar_left, 0, bar_left + bar_width, 0)' in source

def test_ath_tab_add_button_uses_qtabbar_button_position_enum() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")

    assert "QTabBar.ButtonPosition.RightSide" in source
    assert "tabBar().RightSide" not in source


def test_ath_generation_uses_worker_and_delayed_stop_button() -> None:
    main_source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")
    worker_source = Path("src/blab/ui/ath_worker.py").read_text(encoding="utf-8")

    assert "from blab.ui.ath_worker import AthGenerationWorker" in main_source
    assert "self.ath_thread: QThread | None = None" in main_source
    assert "self.ath_worker: AthGenerationWorker | None = None" in main_source
    assert "self.cancel_button.clicked.connect(self.cancel_current_operation)" in main_source
    assert "self.ath_worker = AthGenerationWorker(" in main_source
    assert "self.ath_worker.moveToThread(self.ath_thread)" in main_source
    assert "QTimer.singleShot(3000, self._enable_ath_cancel_if_active)" in main_source
    assert "def cancel_ath_generation(" in main_source
    assert "self.ath_worker.stop()" in main_source
    assert "class AthGenerationWorker(QObject)" in worker_source
    assert "AthProcessRunner()" in worker_source
    assert "clean_ath_mesh_output(raw_result)" in worker_source
