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


def test_balloon_contours_exclude_configured_maximum() -> None:
    source = Path("src/blab/ui/balloon.py").read_text(encoding="utf-8")

    assert "if min_db < level < max_db" in source
