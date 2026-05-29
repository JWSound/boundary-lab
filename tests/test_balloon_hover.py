import numpy as np
import pytest

pytest.importorskip("PySide6")

from blab.ui.balloon import (
    HORIZONTAL_ANGLE_SCALAR_NAME,
    SPL_SCALAR_NAME,
    VERTICAL_ANGLE_SCALAR_NAME,
    _balloon_angle_arrays,
    _balloon_hover_text,
    _balloon_isobar_slice,
    _balloon_radar_slice,
    _contour_levels,
    _nearest_periodic_angle_index,
    _protractor_line_specs,
    _protractor_basis,
    _protractor_ring_radii,
)


def test_balloon_angle_arrays_match_horizontal_and_vertical_planes() -> None:
    theta = np.array(
        [
            [0.0, 0.0],
            [np.pi / 2.0, np.pi / 2.0],
        ],
        dtype=float,
    )
    phi = np.array(
        [
            [0.0, np.pi / 2.0],
            [0.0, np.pi / 2.0],
        ],
        dtype=float,
    )

    horizontal, vertical = _balloon_angle_arrays(theta, phi)

    np.testing.assert_allclose(horizontal, [0.0, 90.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(vertical, [0.0, 0.0, 0.0, 90.0], atol=1e-5)


def test_balloon_hover_text_formats_angles_and_spl() -> None:
    mesh = {
        HORIZONTAL_ANGLE_SCALAR_NAME: np.array([-15.25], dtype=np.float32),
        VERTICAL_ANGLE_SCALAR_NAME: np.array([7.75], dtype=np.float32),
        SPL_SCALAR_NAME: np.array([-12.125], dtype=np.float32),
    }

    text = _balloon_hover_text(mesh, 0)

    assert text == " | ".join(
        (
            "Horizontal Angle: -15.2 deg",
            "Vertical Angle: +7.8 deg",
            "Normalized SPL: -12.1 dB",
        )
    )


def test_balloon_contour_levels_skip_configured_maximum() -> None:
    assert _contour_levels(-30.0, 0.0, 6.0) == [-24.0, -18.0, -12.0, -6.0]


def test_protractor_basis_rotates_horizontal_axis_around_z() -> None:
    u_axis, z_axis = _protractor_basis(90.0)

    np.testing.assert_allclose(u_axis, [0.0, 1.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(z_axis, [0.0, 0.0, 1.0], atol=1e-8)


def test_protractor_ring_radii_use_six_db_spacing() -> None:
    assert _protractor_ring_radii(30.0) == [6.0, 12.0, 18.0, 24.0]


def test_protractor_line_specs_start_in_xz_plane() -> None:
    specs = _protractor_line_specs(12.0, 12.0, 0.0)

    all_points = np.vstack([points for points, _color, _width in specs])

    np.testing.assert_allclose(all_points[:, 1], 0.0, atol=1e-8)


def test_nearest_periodic_angle_index_wraps_around_zero() -> None:
    angles = np.deg2rad(np.array([0.0, 90.0, 180.0, 270.0, 360.0]))

    assert _nearest_periodic_angle_index(angles, np.deg2rad(359.0)) in (0, 4)


def test_balloon_isobar_slice_uses_opposite_azimuth_for_negative_angles() -> None:
    theta = np.deg2rad(np.array([0.0, 90.0, 180.0], dtype=np.float32))
    phi = np.deg2rad(np.array([0.0, 90.0, 180.0, 270.0, 360.0], dtype=np.float32))
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    spl = np.array(
        [
            [
                [0.0, 10.0, 20.0, 30.0, 40.0],
                [1.0, 11.0, 21.0, 31.0, 41.0],
                [2.0, 12.0, 22.0, 32.0, 42.0],
            ]
        ],
        dtype=np.float32,
    )
    prepared = {
        "freq_hz": np.array([1000.0], dtype=np.float32),
        "theta_grid_rad": theta_grid,
        "phi_grid_rad": phi_grid,
        "balloon_surface_spl": spl,
    }

    freqs_hz, angles_deg, values_db = _balloon_isobar_slice(prepared, 90.0)

    np.testing.assert_allclose(freqs_hz, [1000.0])
    np.testing.assert_allclose(angles_deg, [-180.0, -90.0, 0.0, 90.0, 180.0])
    assert values_db.shape == (5, 1)
    np.testing.assert_allclose(values_db[:, 0], [32.0, 31.0, 10.0, 11.0, 12.0])


def test_balloon_radar_slice_selects_current_frequency() -> None:
    theta = np.deg2rad(np.array([0.0, 90.0, 180.0], dtype=np.float32))
    phi = np.deg2rad(np.array([0.0, 90.0, 180.0, 270.0, 360.0], dtype=np.float32))
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    spl = np.array(
        [
            [
                [0.0, 10.0, 20.0, 30.0, 40.0],
                [1.0, 11.0, 21.0, 31.0, 41.0],
                [2.0, 12.0, 22.0, 32.0, 42.0],
            ],
            [
                [100.0, 110.0, 120.0, 130.0, 140.0],
                [101.0, 111.0, 121.0, 131.0, 141.0],
                [102.0, 112.0, 122.0, 132.0, 142.0],
            ],
        ],
        dtype=np.float32,
    )
    prepared = {
        "freq_hz": np.array([1000.0, 2000.0], dtype=np.float32),
        "theta_grid_rad": theta_grid,
        "phi_grid_rad": phi_grid,
        "balloon_surface_spl": spl,
    }

    angles_deg, values_db = _balloon_radar_slice(prepared, 1, 90.0)

    np.testing.assert_allclose(angles_deg, [-180.0, -90.0, 0.0, 90.0, 180.0])
    np.testing.assert_allclose(values_db, [132.0, 131.0, 110.0, 111.0, 112.0])


def test_balloon_isobar_slice_can_smooth_and_interpolate_for_plotting() -> None:
    theta = np.deg2rad(np.array([0.0, 90.0, 180.0], dtype=np.float32))
    phi = np.deg2rad(np.array([0.0, 180.0, 360.0], dtype=np.float32))
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    spl = np.array(
        [
            [[-3.0, -5.0, -3.0], [-6.0, -8.0, -6.0], [-9.0, -11.0, -9.0]],
            [[-2.0, -4.0, -2.0], [-5.0, -7.0, -5.0], [-8.0, -10.0, -8.0]],
        ],
        dtype=np.float32,
    )
    prepared = {
        "freq_hz": np.array([1000.0, 2000.0], dtype=np.float32),
        "theta_grid_rad": theta_grid,
        "phi_grid_rad": phi_grid,
        "balloon_surface_spl": spl,
    }

    freqs_hz, angles_deg, values_db = _balloon_isobar_slice(
        prepared,
        0.0,
        clip_min_db=-30.0,
        angle_samples=9,
        freq_samples=7,
        octave_smoothing=24,
    )

    assert freqs_hz.shape == (7,)
    assert angles_deg.shape == (9,)
    assert values_db.shape == (9, 7)
