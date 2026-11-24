from dataclasses import replace

import numpy as np
import pytest

from blab.config import ChannelConfig
from blab.exporting import export_polar_text_files
from blab.live import LiveSolveDataset
from blab.solvers.base import FrequencyResult
from blab.ui.main_window.plot_presenter import PlotPresenterMixin
from blab.ui.plots import IsobarCanvas
from blab.ui.result_projection import ProjectionOptions, ResultProjectionService


def make_projection():
    dataset = LiveSolveDataset(
        np.asarray([-90.0, 0.0, 90.0]),
        flat_target_normalization_enabled=False,
    )
    for freq in (500.0, 1000.0):
        dataset.add(
            FrequencyResult(
                freq_hz=freq,
                horizontal_spl_norm_db=np.zeros(3),
                vertical_spl_norm_db=np.zeros(3),
                impedance=np.ones((1, 2)),
                channel_names=np.asarray(["LF", "HF", "Rear"]),
                horizontal_pressure=np.asarray([[0.5, 1, 0.5], [-0.25, 1, -0.25], [0.1j, 0.2j, 0.1j]]),
                vertical_pressure=np.asarray([[0.25, 1, 0.25], [-0.1, 1, -0.1], [0.1j, 0.2j, 0.1j]]),
            )
        )
    channels = (ChannelConfig(name="LF"), ChannelConfig(name="HF"), ChannelConfig(name="Rear"))
    options = ProjectionOptions(3, 2, None, 0, 0, 0, 0, -60, 0)
    projection = ResultProjectionService().prepare(dataset, channels, options)
    return dataset, projection


def test_channel_selection_preserves_complex_interference_and_other_projections():
    dataset, projection = make_projection()
    original_response = projection.response.horizontal_spl_db.copy()
    isobar = projection.isobar
    assert isobar.select_channels(frozenset(["LF", "HF", "Rear"])) is isobar
    lf = isobar.select_channels(frozenset(["LF"]))
    np.testing.assert_allclose(lf.horizontal_db[:, 0], [-6.0206, 0, -6.0206], atol=1e-4)
    np.testing.assert_allclose(lf.vertical_db[:, 0], [-12.0412, 0, -12.0412], atol=1e-4)
    pair = isobar.select_channels(frozenset(["LF", "HF"]))
    np.testing.assert_allclose(
        pair.horizontal_db[:, 0], [20 * np.log10(0.25 / 2), 0, 20 * np.log10(0.25 / 2)], atol=1e-4
    )
    np.testing.assert_array_equal(projection.response.horizontal_spl_db, original_response)
    assert len(dataset.channel_configs) == 3
    assert isobar.select_channels(frozenset()) is None
    assert isobar.select_channels(frozenset(["missing"])) is None


def test_basis_keeps_dsp_and_snapshot_is_independent():
    dataset, projection = make_projection()
    dataset.set_channel_synthesis((ChannelConfig(name="LF", level_db=6, delay_ms=0.5, polarity=-1),))
    basis = dataset.polar_channel_basis()
    expected = -(10 ** (6 / 20)) * np.exp(-1j * 2 * np.pi * 500 * 0.0005)
    np.testing.assert_allclose(basis.horizontal[0, 0], np.asarray([0.5, 1, 0.5]) * expected, atol=1e-6)
    snapshot = projection.snapshot()
    projection.isobar.channel_basis.horizontal[:] = 0
    assert np.any(snapshot.isobar.channel_basis.horizontal)


def test_nulls_and_silent_channels_remain_finite():
    _, projection = make_projection()
    basis = projection.isobar.channel_basis
    basis.horizontal[:, 0] = 0
    basis.vertical[:, 0] = 0
    silent = projection.isobar.select_channels(frozenset(["LF"]))
    np.testing.assert_array_equal(silent.horizontal_db, -60)
    basis.horizontal[:, 0] = [1, 1, 1]
    basis.horizontal[:, 1] = [-1, -1, -0.5]
    cancelled = projection.isobar.select_channels(frozenset(["LF", "HF"]))
    assert np.isfinite(cancelled.horizontal_db).all()


def test_selected_export_matches_magnitude_and_complex_phase(tmp_path):
    dataset, _ = make_projection()
    export_polar_text_files(
        dataset, tmp_path, planes=("H",), selected_channels=frozenset(["HF"]), reference_angles_deg={"H": 0}
    )
    values = np.loadtxt(tmp_path / "H 90.txt")
    np.testing.assert_allclose(values[:, 1], 20 * np.log10(0.25), atol=0.001)
    np.testing.assert_allclose(np.abs(values[:, 2]), 180, atol=0.001)
    assert not list(tmp_path.glob("V *"))
    with pytest.raises(ValueError, match="No polar channels"):
        export_polar_text_files(dataset, tmp_path, selected_channels=frozenset())


def test_canvas_selection_survives_refresh_and_is_independent(qapp):
    _, projection = make_projection()
    horizontal = IsobarCanvas("Horizontal")
    vertical = IsobarCanvas("Vertical")
    try:
        presenter = PlotPresenterMixin()
        presenter._selected_isobar(horizontal, projection)
        presenter._selected_isobar(vertical, projection)
        changes = []
        horizontal.trace_visibility_changed.connect(lambda: changes.append(True))
        horizontal._series_actions["HF"].trigger()
        assert not horizontal.is_series_visible("HF")
        assert vertical.is_series_visible("HF")
        assert changes == [True]
        filtered = presenter._selected_isobar(horizontal, projection)
        assert not np.array_equal(filtered.horizontal_db, projection.isobar.horizontal_db)
        horizontal.clear_for_solve()
        assert not horizontal.trace_filter_action.isEnabled()
        presenter._selected_isobar(horizontal, projection)
        assert not horizontal.is_series_visible("HF")
        for name in horizontal._series_labels:
            horizontal.set_series_visible(name, False)
        assert presenter._selected_isobar(horizontal, projection) is None
        horizontal._draw_empty()
        assert horizontal.trace_filter_action.isEnabled()
        assert not horizontal.has_selected_traces()
        legacy = replace(projection, isobar=replace(projection.isobar, channel_basis=None))
        assert presenter._selected_isobar(horizontal, legacy) is legacy.isobar
        assert not horizontal.trace_filter_action.isEnabled()
    finally:
        horizontal.close()
        vertical.close()


def test_main_window_selection_updates_plot_export_and_previous_solve(main_window, monkeypatch, tmp_path):
    dataset, projection = make_projection()
    main_window.live_dataset = dataset
    main_window.solve_session.last_completed_visualization = projection.snapshot()
    monkeypatch.setattr(main_window, "channel_configs", lambda: dataset.channel_configs)
    monkeypatch.setattr(main_window, "prepared_live_dataset", lambda **_kwargs: projection)
    monkeypatch.setattr(main_window, "_plot_entry_is_actively_visible", lambda _entry: True)
    # Exercise the normal update handlers while avoiding desktop window exposure.
    monkeypatch.setattr(
        main_window,
        "refresh_plots",
        lambda **_kwargs: (
            main_window._update_horizontal_plot(projection),
            main_window._update_vertical_plot(projection),
        ),
    )
    main_window.refresh_plots()
    horizontal, vertical = main_window.horizontal_plot, main_window.vertical_plot
    horizontal._series_actions["HF"].trigger()
    horizontal._series_actions["Rear"].trigger()
    expected = projection.isobar.select_channels(frozenset(["LF"]))
    np.testing.assert_allclose(horizontal._mesh_values_db, expected.horizontal_db)
    np.testing.assert_allclose(vertical._mesh_values_db, projection.isobar.vertical_db)
    np.testing.assert_allclose(horizontal._comparison_plot[2], expected.horizontal_db)
    main_window.preferences.horizontal_normalization_angle = 0
    main_window._write_plot_data("horizontal_isobar", tmp_path)
    np.testing.assert_allclose(np.loadtxt(tmp_path / "H 90.txt")[:, 1], -6.0206, atol=0.001)
    horizontal._series_actions["LF"].trigger()
    assert horizontal._mesh_values_db is None
    assert horizontal._comparison_plot is None
    assert not main_window.export_plot_data_actions["horizontal_isobar"].isEnabled()
    horizontal._series_actions["LF"].trigger()
    np.testing.assert_allclose(horizontal._mesh_values_db, expected.horizontal_db)
