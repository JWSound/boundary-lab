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
    spinorama_block = source[source.index("class SpinoramaCanvas"):]

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

    assert "QDockWidget" in source
    assert "class DockTitleBar" in source
    assert "save_action: QAction | None = None" in source
    assert "self.save_button.setDefaultAction(save_action)" in source
    assert "dock.setTitleBarWidget(DockTitleBar(title, dock, save_action=save_action))" in source
    assert "save_action=self.export_plot_actions.get(entry.plot_id)" in source
    assert "close_button.clicked.connect(dock.close)" in source
    assert "event.ignore()" in source
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
    assert "settings_bool(self.settings, f\"plots/{entry.plot_id}/visible\", True)" not in source
    assert "settings.setValue(f\"plots/{plot_id}/visible\"" not in source
    assert "action.toggled.connect(lambda checked, dock=dock: dock.setVisible(bool(checked)))" in source
    assert "dock.visibilityChanged.connect(lambda _visible, plot_id=entry.plot_id: self._sync_plot_view_action(plot_id))" in source
    assert "self.workspace.saveState()" in source
    assert "self.workspace.restoreState(dock_state)" in source
    assert "window/dock_state" in source
    assert "DEFAULT_DOCK_STATE_B64" in source
    assert "QByteArray.fromBase64(DEFAULT_DOCK_STATE_B64.encode(\"ascii\"))" in source


def test_plot_export_uses_dock_title_save_buttons_not_file_menu() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")

    assert 'file_menu.addMenu("Export Plot")' not in source
    assert "self.export_plot_actions[entry.plot_id] = action" in source
    assert "action.setToolTip(f\"Export {entry.title}\")" in source
    assert "SAVE_DARK_ICON" in source
    assert "SAVE_LIGHT_ICON" in source
    assert "def _refresh_plot_export_icons(" in source
    assert "icon_path = SAVE_LIGHT_ICON if window_color.lightness() >= 128 else SAVE_DARK_ICON" in source
    assert "action.setIcon(icon)" in source
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
    channel_dialog = dialog_source[dialog_source.index("class ChannelConfigDialog"):dialog_source.index("class SourceConfigDialog")]

    assert "channelsChanged" not in channel_dialog
    assert "_emit_channels_changed" not in channel_dialog
    assert "buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)" in channel_dialog
    assert "buttons.rejected.connect(self.reject)" in channel_dialog
    assert "_preview_channel_config" not in main_source
    assert "dialog.channelsApplied.connect(self._apply_channel_config)" in main_source


def test_application_startup_invokes_new_project_reset() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")
    init_block = source[source.index("    def __init__("):source.index("    def changeEvent")]

    assert 'startup("Starting new project...")' in init_block
    assert "self.new_project()" in init_block
    assert "_load_initial_ath_scripts" not in source
    assert "_load_imported_meshes" not in source
    assert "mesh/imported_meshes" not in source
    assert "mesh/ath_mesh" not in source


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
    assert '"Balloon Sampling", self.spherical_sampling_check' in dialog_source
    assert '"Balloon Angle Precision", self.balloon_angle_precision_spin' in dialog_source
    assert '"preferences/live_plot_quality"' in main_source
    assert "live_plot_angle_samples(self.preferences.live_plot_quality)" in main_source
    assert "live_plot_freq_samples(self.preferences.live_plot_quality)" in main_source
    assert "FINAL_ISOBAR_ANGLE_SAMPLES = 1000" in plot_source
    assert "FINAL_ISOBAR_FREQ_SAMPLES = 500" in plot_source
    assert 'LIVE_ISOBAR_SHADING = "nearest"' in plot_source
    assert 'FINAL_ISOBAR_SHADING = "gouraud"' in plot_source
    assert "self._use_final_isobar_resolution = solve_completed" in main_source
    assert "angle_samples=FINAL_ISOBAR_ANGLE_SAMPLES" in main_source
    assert "freq_samples=FINAL_ISOBAR_FREQ_SAMPLES" in main_source
    assert 'angle_samples=FINAL_ISOBAR_ANGLE_SAMPLES if plot_id in {"horizontal_isobar", "vertical_isobar"} else None' in main_source
    assert 'freq_samples=FINAL_ISOBAR_FREQ_SAMPLES if plot_id in {"horizontal_isobar", "vertical_isobar"} else None' in main_source
    assert "shading=FINAL_ISOBAR_SHADING if self._use_final_isobar_resolution else LIVE_ISOBAR_SHADING" in main_source


def test_isobar_canvas_allows_custom_right_margin() -> None:
    source = Path("src/blab/ui/plots.py").read_text(encoding="utf-8")

    assert "left_margin: float = 0.14" in source
    assert "right_margin: float = 0.98" in source
    assert "left=self.left_margin" in source
    assert "right=self.right_margin" in source


def test_isobar_canvas_reuses_heatmap_artist_between_grid_changes() -> None:
    source = Path("src/blab/ui/plots.py").read_text(encoding="utf-8")
    isobar_block = source[source.index("class IsobarCanvas"):source.index("class ImpedanceCanvas")]

    assert "self._mesh_artist" in isobar_block
    assert "self._image_artist" in isobar_block
    assert "def _mesh_matches(" in isobar_block
    assert "self._mesh_artist.set_array" in isobar_block
    assert "self.axes.pcolormesh(" in isobar_block
    assert "shading=shading" in isobar_block
    assert "self._mesh_shading == shading" in isobar_block
    assert 'render_mode = "image" if shading == FINAL_ISOBAR_SHADING else "mesh"' in isobar_block
    assert "self.axes.imshow(" in isobar_block
    assert 'interpolation="bilinear"' in isobar_block
    assert "np.log10(freqs_hz)" in isobar_block
    assert "apply_log_image_frequency_axis" in source
    assert isobar_block.count("clear_plot_axes(self.axes)") == 1


def test_isobar_canvas_captures_and_redraws_persistent_contours() -> None:
    source = Path("src/blab/ui/plots.py").read_text(encoding="utf-8")
    isobar_block = source[source.index("class IsobarCanvas"):source.index("class ImpedanceCanvas")]
    draw_empty_block = isobar_block[isobar_block.index("    def _draw_empty"):isobar_block.index("    def _remove_artist")]
    remove_contour_block = isobar_block[
        isobar_block.index("    def _remove_contour_artist"):isobar_block.index("    @property")
    ]

    assert "self._captured_contours" in isobar_block
    assert "self._mesh_values_db" in isobar_block
    assert "def capture_contours(" in isobar_block
    assert "def clear_contours(" in isobar_block
    assert "def _redraw_captured_contours(" in isobar_block
    assert "np.arange(np.ceil(clip_min_db / 3.0) * 3.0" in isobar_block
    assert "levels.copy()" in isobar_block
    assert "colors=\"white\"" in isobar_block
    assert "linewidths=0.9" in isobar_block
    assert "linestyles=\"solid\"" in isobar_block
    assert "alpha=0.85" in isobar_block
    assert "self._redraw_captured_contours()" in isobar_block
    assert "self._captured_contours = None" not in draw_empty_block
    assert "self._contour_artist = None" in draw_empty_block
    assert "NotImplementedError" in remove_contour_block


def test_main_window_contour_buttons_are_final_render_and_visibility_gated() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")

    assert 'QPushButton("Capture Contours")' in source
    assert 'QPushButton("Clear Contours")' in source
    assert "controls_layout.addWidget(self.capture_contours_button)" in source
    assert "controls_layout.addWidget(self.clear_contours_button)" in source
    assert "self.capture_contours_button.clicked.connect(self.capture_visible_isobar_contours)" in source
    assert "self.clear_contours_button.clicked.connect(self.clear_isobar_contours)" in source
    assert "self._final_isobar_plots_rendered = False" in source
    assert "self._final_isobar_plots_rendered = solve_completed and bool(self._visible_isobar_plots())" in source
    assert "self._use_final_isobar_resolution" in source
    assert "and self._final_isobar_plots_rendered" in source
    assert "and self._visible_isobar_plots()" in source


def test_main_window_captures_only_visible_isobar_plots_and_preserves_until_clear() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")

    assert "def _visible_isobar_plots(" in source
    assert "def _sync_plot_view_action(" in source
    assert 'self.plot_docks.get("horizontal_isobar")' in source
    assert 'self.plot_docks.get("vertical_isobar")' in source
    assert "action.setChecked(not dock.isHidden())" in source
    assert "for plot in self._visible_isobar_plots():" in source
    assert "plot.capture_contours()" in source
    assert "self.horizontal_plot.clear_contours()" in source
    assert "self.vertical_plot.clear_contours()" in source
    assert "self.clear_contours_button.setEnabled(self._has_captured_isobar_contours())" in source


def test_balloon_contours_exclude_configured_maximum() -> None:
    source = Path("src/blab/ui/balloon.py").read_text(encoding="utf-8")

    assert "if min_db < level < max_db" in source


def test_ath_tab_add_button_uses_qtabbar_button_position_enum() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")

    assert "QTabBar.ButtonPosition.RightSide" in source
    assert "tabBar().RightSide" not in source
