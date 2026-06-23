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
    _front_angle_meshes,
    _nearest_periodic_angle_index,
    _protractor_basis,
    _protractor_line_specs,
    _protractor_ring_radii,
    _superellipse_radius,
    _wavefront_shape_summary,
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


def test_superellipse_radius_matches_axis_radii() -> None:
    angles = np.deg2rad(np.array([0.0, 90.0, 180.0, 270.0]))

    radius = _superellipse_radius(angles, 40.0, 25.0, 4.0)

    np.testing.assert_allclose(radius, [40.0, 25.0, 40.0, 25.0], atol=1e-6)


def test_wavefront_shape_summary_fits_aspect_scaled_superellipse() -> None:
    theta_values = np.linspace(0.0, np.pi, 64, dtype=np.float32)
    phi_values = np.linspace(0.0, 2.0 * np.pi, 97, dtype=np.float32)
    theta_grid, phi_grid = np.meshgrid(theta_values, phi_values, indexing="ij")
    horizontal_deg, vertical_deg, front_mask = _front_angle_meshes(theta_grid, phi_grid)
    horizontal_tangent = np.tan(np.deg2rad(horizontal_deg))
    vertical_tangent = np.tan(np.deg2rad(vertical_deg))
    exponent = 4.0
    horizontal_radius_deg = 42.0
    vertical_radius_deg = 28.0
    horizontal_radius = np.tan(np.deg2rad(horizontal_radius_deg))
    vertical_radius = np.tan(np.deg2rad(vertical_radius_deg))
    metric = (np.abs(horizontal_tangent) / horizontal_radius) ** exponent + (
        np.abs(vertical_tangent) / vertical_radius
    ) ** exponent
    spl = np.full(theta_grid.shape, -30.0, dtype=np.float32)
    spl[front_mask] = np.clip(-6.0 * metric[front_mask], -30.0, 0.0)
    prepared = {
        "freq_hz": np.array([1000.0], dtype=np.float32),
        "theta_grid_rad": theta_grid,
        "phi_grid_rad": phi_grid,
        "balloon_surface_spl": spl[np.newaxis, :, :],
    }

    summary = _wavefront_shape_summary(prepared)

    assert summary["valid"].tolist() == [True]
    np.testing.assert_allclose(summary["shape_exponent"], [exponent], atol=0.15)
    np.testing.assert_allclose(summary["aspect_ratio"], [horizontal_radius_deg / vertical_radius_deg], atol=0.03)
    assert float(summary["fit_residual_percent"][0]) < 0.5


def test_wavefront_shape_summary_keeps_wide_axisymmetric_beam_circular() -> None:
    theta_values = np.linspace(0.0, np.pi, 80, dtype=np.float32)
    phi_values = np.linspace(0.0, 2.0 * np.pi, 129, dtype=np.float32)
    theta_grid, phi_grid = np.meshgrid(theta_values, phi_values, indexing="ij")
    _, _, front_mask = _front_angle_meshes(theta_grid, phi_grid)
    radius_deg = 78.0
    radius_tangent = np.tan(np.deg2rad(radius_deg))
    radial_tangent = np.tan(theta_grid)
    metric = (radial_tangent / radius_tangent) ** 2.0
    spl = np.full(theta_grid.shape, -30.0, dtype=np.float32)
    spl[front_mask] = np.clip(-6.0 * metric[front_mask], -30.0, 0.0)
    prepared = {
        "freq_hz": np.array([329.0], dtype=np.float32),
        "theta_grid_rad": theta_grid,
        "phi_grid_rad": phi_grid,
        "balloon_surface_spl": spl[np.newaxis, :, :],
    }

    summary = _wavefront_shape_summary(prepared)

    assert summary["valid"].tolist() == [True]
    np.testing.assert_allclose(summary["shape_exponent"], [2.0], atol=0.08)
    np.testing.assert_allclose(summary["horizontal_beamwidth_deg"], [2.0 * radius_deg], atol=0.8)
    np.testing.assert_allclose(summary["vertical_beamwidth_deg"], [2.0 * radius_deg], atol=0.8)


def test_wavefront_shape_summary_computes_spherical_directivity_index_from_raw_samples() -> None:
    theta_grid, phi_grid = np.meshgrid(
        np.linspace(0.0, np.pi, 3, dtype=np.float32),
        np.linspace(0.0, 2.0 * np.pi, 4, dtype=np.float32),
        indexing="ij",
    )
    freqs = np.array([500.0, 1000.0], dtype=np.float32)
    prepared = {
        "freq_hz": freqs,
        "theta_grid_rad": theta_grid,
        "phi_grid_rad": phi_grid,
        "balloon_surface_spl": np.zeros((2, 3, 4), dtype=np.float32),
    }
    raw = {
        "freq_hz": freqs,
        "spl_norm": np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, -10.0, -10.0, -10.0],
            ],
            dtype=np.float32,
        ),
    }

    summary = _wavefront_shape_summary(prepared, raw_balloon_data=raw)

    np.testing.assert_allclose(summary["directivity_index_db"][0], 0.0, atol=1e-6)
    expected = -10.0 * np.log10(np.mean(10.0 ** (raw["spl_norm"][1] / 10.0)))
    np.testing.assert_allclose(summary["directivity_index_db"][1], expected, atol=1e-6)


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
