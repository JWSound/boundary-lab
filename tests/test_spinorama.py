import numpy as np

from blab.spinorama import compute_spinorama_from_planes


def test_spinorama_flat_directivity_has_zero_di() -> None:
    freqs = np.array([200.0, 1000.0, 5000.0], dtype=np.float32)
    angles = np.arange(-180.0, 181.0, 10.0, dtype=np.float32)
    horizontal = np.zeros((freqs.size, angles.size), dtype=np.float32)
    vertical = np.zeros_like(horizontal)

    curves = compute_spinorama_from_planes(freqs, angles, horizontal, vertical)

    assert np.allclose(curves.on_axis_db, 0.0)
    assert np.allclose(curves.listening_window_db, 0.0)
    assert np.allclose(curves.early_reflections_db, 0.0)
    assert np.allclose(curves.sound_power_db, 0.0)
    assert np.allclose(curves.estimated_in_room_db, 0.0)
    assert np.allclose(curves.early_reflections_di_db, 0.0)
    assert np.allclose(curves.sound_power_di_db, 0.0)


def test_spinorama_detects_increasing_directivity() -> None:
    freqs = np.array([1000.0], dtype=np.float32)
    angles = np.arange(-180.0, 181.0, 10.0, dtype=np.float32)
    off_axis_loss = -24.0 * (np.abs(angles) / 180.0) ** 1.5
    horizontal = off_axis_loss[np.newaxis, :].astype(np.float32)
    vertical = (off_axis_loss * 1.2)[np.newaxis, :].astype(np.float32)

    curves = compute_spinorama_from_planes(freqs, angles, horizontal, vertical)

    assert curves.on_axis_db[0] == 0.0
    assert curves.sound_power_db[0] < curves.listening_window_db[0]
    assert curves.sound_power_di_db[0] > 0.0
    assert curves.early_reflections_di_db[0] > 0.0


def test_spinorama_absolute_spl_is_displayed_relative_to_physical_on_axis() -> None:
    freqs = np.array([1000.0], dtype=np.float32)
    angles = np.arange(-180.0, 181.0, 10.0, dtype=np.float32)
    horizontal = (90.0 - 12.0 * (np.abs(angles) / 180.0))[np.newaxis, :].astype(np.float32)
    vertical = (horizontal - 2.0).astype(np.float32)

    curves = compute_spinorama_from_planes(freqs, angles, horizontal, vertical)

    assert curves.on_axis_db[0] < 0.0
    assert curves.on_axis_db[0] > -2.0
    assert curves.sound_power_db[0] < 0.0


def test_spinorama_interpolates_solver_angle_grid() -> None:
    freqs = np.array([500.0, 2000.0], dtype=np.float32)
    angles = np.array([-180.0, -45.0, 0.0, 45.0, 180.0], dtype=np.float32)
    horizontal = -0.1 * np.abs(angles)[np.newaxis, :] + np.array([[0.0], [2.0]])
    vertical = horizontal - 1.0

    curves = compute_spinorama_from_planes(freqs, angles, horizontal, vertical)

    assert curves.freq_hz.tolist() == [500.0, 2000.0]
    assert np.all(np.isfinite(curves.listening_window_db))
    assert np.all(np.isfinite(curves.sound_power_db))


def test_spinorama_reference_axis_only_moves_reference_dependent_curves() -> None:
    freqs = np.array([1000.0], dtype=np.float32)
    angles = np.arange(-180.0, 181.0, 10.0, dtype=np.float32)
    horizontal = (86.0 - 0.04 * (angles - 20.0) ** 2)[np.newaxis, :].astype(np.float32)
    vertical = (84.0 - 0.035 * (angles + 10.0) ** 2)[np.newaxis, :].astype(np.float32)

    centered = compute_spinorama_from_planes(freqs, angles, horizontal, vertical)
    shifted = compute_spinorama_from_planes(
        freqs,
        angles,
        horizontal,
        vertical,
        horizontal_reference_angle_deg=20.0,
        vertical_reference_angle_deg=-10.0,
    )

    assert not np.allclose(centered.on_axis_db, shifted.on_axis_db)
    assert not np.allclose(centered.listening_window_db, shifted.listening_window_db)
    assert np.allclose(centered.early_reflections_db, shifted.early_reflections_db)
    assert np.allclose(centered.sound_power_db, shifted.sound_power_db)


def test_spinorama_reference_axis_wraps_around_periodic_polar_grid() -> None:
    freqs = np.array([1000.0], dtype=np.float32)
    angles = np.arange(-180.0, 181.0, 10.0, dtype=np.float32)
    horizontal = angles[np.newaxis, :].astype(np.float32)
    vertical = -angles[np.newaxis, :].astype(np.float32)

    curves = compute_spinorama_from_planes(
        freqs,
        angles,
        horizontal,
        vertical,
        horizontal_reference_angle_deg=170.0,
        vertical_reference_angle_deg=-170.0,
    )

    assert np.isfinite(curves.listening_window_db[0])
