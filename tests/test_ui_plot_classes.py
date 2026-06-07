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


def test_plot_panel_uses_compact_spacing_and_title_padding() -> None:
    plot_source = Path("src/blab/ui/plots.py").read_text(encoding="utf-8")
    main_source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")

    assert "PLOT_TITLE_PAD = 1" in plot_source
    assert "set_title(self.title, pad=PLOT_TITLE_PAD)" in plot_source
    assert "plot_layout.setSpacing(4)" in main_source


def test_main_splitter_defers_expensive_resizes_until_drag_release() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")

    assert "self.main_splitter.setOpaqueResize(False)" in source


def test_live_plot_refresh_is_immediate_and_visibility_aware() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")

    assert "_request_plot_refresh" not in source
    assert "_plot_refresh_timer" not in source
    assert "visible_entries = [entry for entry in self.plot_entries if entry.widget.isVisible()]" in source
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
    assert "def _mesh_matches(" in isobar_block
    assert "self._mesh_artist.set_array" in isobar_block
    assert "self.axes.pcolormesh(" in isobar_block
    assert "shading=shading" in isobar_block
    assert "self._mesh_shading == shading" in isobar_block
    assert isobar_block.count("clear_plot_axes(self.axes)") == 1


def test_balloon_contours_exclude_configured_maximum() -> None:
    source = Path("src/blab/ui/balloon.py").read_text(encoding="utf-8")

    assert "if min_db < level < max_db" in source


def test_ath_tab_add_button_uses_qtabbar_button_position_enum() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")

    assert "QTabBar.ButtonPosition.RightSide" in source
    assert "tabBar().RightSide" not in source
