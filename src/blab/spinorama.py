"""CEA-2034 style curves derived from horizontal and vertical polar slices.

This module intentionally keeps the implementation small and NumPy-only. The
input data is the solver's existing horizontal/vertical SPL matrices rather
than a third-party measurement file format.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPL_CURVE_NAMES = (
    "On Axis",
    "Listen. Wind.",
    "Early Refl.",
    "Sound Power",
    "PIR",
)
DI_CURVE_NAMES = (
    "ERDI",
    "SPDI",
)


@dataclass(frozen=True)
class SpinoramaCurves:
    freq_hz: np.ndarray
    on_axis_db: np.ndarray
    listening_window_db: np.ndarray
    early_reflections_db: np.ndarray
    sound_power_db: np.ndarray
    estimated_in_room_db: np.ndarray
    early_reflections_di_db: np.ndarray
    sound_power_di_db: np.ndarray

    def spl_curves(self) -> tuple[tuple[str, np.ndarray], ...]:
        return (
            ("On Axis", self.on_axis_db),
            ("Listen. Wind.", self.listening_window_db),
            ("Early Refl.", self.early_reflections_db),
            ("Sound Power", self.sound_power_db),
            ("PIR", self.estimated_in_room_db),
        )

    def di_curves(self) -> tuple[tuple[str, np.ndarray], ...]:
        return (
            ("ERDI", self.early_reflections_di_db),
            ("SPDI", self.sound_power_di_db),
        )


def compute_spinorama_from_planes(
    freq_hz: np.ndarray,
    polar_angle_deg: np.ndarray,
    horizontal_spl_db: np.ndarray,
    vertical_spl_db: np.ndarray,
    *,
    horizontal_reference_angle_deg: float = 0.0,
    vertical_reference_angle_deg: float = 0.0,
) -> SpinoramaCurves:
    """Compute CEA-2034 style curves from H/V polar SPL matrices.

    The solver supplies two orthogonal slices rather than a full spherical
    balloon. Sound power is therefore an H/V estimate: all 10-degree H/V
    samples are energy-averaged with approximate spherical band weights.
    """
    freqs = np.asarray(freq_hz, dtype=np.float32)
    angles = np.asarray(polar_angle_deg, dtype=float)
    horizontal = _validate_spl_matrix("horizontal_spl_db", horizontal_spl_db, freqs, angles)
    vertical = _validate_spl_matrix("vertical_spl_db", vertical_spl_db, freqs, angles)

    h = _PlaneSampler(angles, horizontal)
    v = _PlaneSampler(angles, vertical)
    h_ref = float(horizontal_reference_angle_deg)
    v_ref = float(vertical_reference_angle_deg)

    on_axis = _energy_average_db(np.column_stack((h.at(h_ref), v.at(v_ref))), axis=1)
    listening = _energy_average_db(
        np.column_stack(
            (
                h.at(h_ref - 30),
                h.at(h_ref - 20),
                h.at(h_ref - 10),
                h.at(h_ref + 10),
                h.at(h_ref + 20),
                h.at(h_ref + 30),
                v.at(v_ref - 10),
                v.at(v_ref),
                v.at(v_ref + 10),
            )
        ),
        axis=1,
    )

    floor = _energy_average_db(np.column_stack((v.at(-40), v.at(-30), v.at(-20))), axis=1)
    ceiling = _energy_average_db(np.column_stack((v.at(40), v.at(50), v.at(60))), axis=1)
    front = _energy_average_db(
        np.column_stack((h.at(-30), h.at(-20), h.at(-10), h.at(0), h.at(10), h.at(20), h.at(30))),
        axis=1,
    )
    side = _energy_average_db(
        np.column_stack(tuple(h.at(angle) for angle in (-80, -70, -60, -50, -40, 40, 50, 60, 70, 80))),
        axis=1,
    )
    rear = _energy_average_db(
        np.column_stack(
            tuple(
                h.at(angle)
                for angle in (
                    -170,
                    -160,
                    -150,
                    -140,
                    -130,
                    -120,
                    -110,
                    -100,
                    -90,
                    90,
                    100,
                    110,
                    120,
                    130,
                    140,
                    150,
                    160,
                    170,
                    180,
                )
            )
        ),
        axis=1,
    )
    early = _energy_average_db(np.column_stack((floor, ceiling, front, side, rear)), axis=1)

    power_values, power_weights = _sound_power_samples(h, v)
    sound_power = _energy_average_db(power_values, weights=power_weights, axis=1)

    estimated_in_room = _weighted_energy_sum_db(
        (listening, early, sound_power),
        np.asarray([0.12, 0.44, 0.44], dtype=float),
    )
    spl_reference = h.at(0)

    return SpinoramaCurves(
        freq_hz=freqs,
        on_axis_db=(on_axis - spl_reference).astype(np.float32, copy=False),
        listening_window_db=(listening - spl_reference).astype(np.float32, copy=False),
        early_reflections_db=(early - spl_reference).astype(np.float32, copy=False),
        sound_power_db=(sound_power - spl_reference).astype(np.float32, copy=False),
        estimated_in_room_db=(estimated_in_room - spl_reference).astype(np.float32, copy=False),
        early_reflections_di_db=(listening - early).astype(np.float32, copy=False),
        sound_power_di_db=(listening - sound_power).astype(np.float32, copy=False),
    )


class _PlaneSampler:
    def __init__(self, angles_deg: np.ndarray, spl_db: np.ndarray):
        order = np.argsort(angles_deg)
        sorted_angles = angles_deg[order]
        sorted_spl = spl_db[:, order]
        if sorted_angles.size >= 2 and np.isclose(sorted_angles[0], -180.0) and np.isclose(sorted_angles[-1], 180.0):
            sorted_angles = sorted_angles[:-1]
            sorted_spl = sorted_spl[:, :-1]
            self._periodic = True
        else:
            self._periodic = False
        self.angles_deg = sorted_angles
        self.spl_db = sorted_spl

    def at(self, angle_deg: float) -> np.ndarray:
        angle = float(angle_deg)
        if self._periodic:
            angle = ((angle + 180.0) % 360.0) - 180.0
            angles = np.concatenate((self.angles_deg - 360.0, self.angles_deg, self.angles_deg + 360.0))
            values = np.concatenate((self.spl_db, self.spl_db, self.spl_db), axis=1)
        else:
            angles = self.angles_deg
            values = self.spl_db
        return np.asarray(
            [np.interp(angle, angles, row) for row in values],
            dtype=float,
        )


def _validate_spl_matrix(
    name: str,
    values: np.ndarray,
    freq_hz: np.ndarray,
    angle_deg: np.ndarray,
) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 2D with shape (n_freq, n_angles).")
    if out.shape != (freq_hz.size, angle_deg.size):
        raise ValueError(f"{name} shape must match freq_hz and polar_angle_deg.")
    return out


def _db_to_energy(db: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(db, dtype=float) / 10.0)


def _energy_to_db(energy: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(energy, np.finfo(float).tiny))


def _energy_average_db(
    spl_db: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    axis: int,
) -> np.ndarray:
    energy = _db_to_energy(spl_db)
    if weights is None:
        return _energy_to_db(np.mean(energy, axis=axis))
    normalized = np.asarray(weights, dtype=float)
    normalized = normalized / np.sum(normalized)
    return _energy_to_db(np.sum(energy * normalized[np.newaxis, :], axis=axis))


def _weighted_energy_sum_db(curves_db: tuple[np.ndarray, ...], weights: np.ndarray) -> np.ndarray:
    energy = np.vstack([_db_to_energy(curve) for curve in curves_db])
    normalized = weights / np.sum(weights)
    return _energy_to_db(np.sum(energy * normalized[:, np.newaxis], axis=0))


def _sound_power_samples(h: _PlaneSampler, v: _PlaneSampler) -> tuple[np.ndarray, np.ndarray]:
    samples: list[np.ndarray] = []
    sample_angles: list[int] = []

    for angle in range(-180, 181, 10):
        samples.append(h.at(angle))
        sample_angles.append(abs(angle))

    for angle in range(-170, 180, 10):
        if angle == 0:
            continue
        samples.append(v.at(angle))
        sample_angles.append(abs(angle))

    weights = _spherical_band_weights(np.asarray(sample_angles, dtype=float))
    return np.column_stack(samples), weights


def _spherical_band_weights(abs_angles_deg: np.ndarray) -> np.ndarray:
    unique_angles = np.unique(abs_angles_deg)
    ring_weights: dict[float, float] = {}
    for angle in unique_angles:
        lower = max(0.0, angle - 5.0)
        upper = min(180.0, angle + 5.0)
        ring_weights[float(angle)] = np.cos(np.deg2rad(lower)) - np.cos(np.deg2rad(upper))

    sample_weights = np.empty_like(abs_angles_deg, dtype=float)
    for angle in unique_angles:
        mask = np.isclose(abs_angles_deg, angle)
        sample_weights[mask] = ring_weights[float(angle)] / np.count_nonzero(mask)
    return sample_weights
