"""Explicit conversions between solver-native and standard audio phasors."""

from __future__ import annotations

import numpy as np

SOLVER_PHASOR_CONVENTION = "exp(-i omega t)"
STANDARD_AUDIO_PHASOR_CONVENTION = "exp(+i omega t)"


def solver_to_standard_phasor(values):
    """Convert BEAT ``exp(-i omega t)`` phasors to standard audio phasors."""

    return np.conjugate(values)


def standard_to_solver_phasor(values):
    """Convert standard audio phasors to BEAT's native solver convention."""

    return np.conjugate(values)


def solver_phase_deg(values: np.ndarray) -> np.ndarray:
    """Return standard-audio phase angles for solver-native phasors."""

    standard = solver_to_standard_phasor(np.asarray(values))
    phase = np.rad2deg(np.angle(standard)).astype(np.float32, copy=False)
    return np.where(np.isclose(phase, 0.0, atol=1.0e-6), 0.0, phase).astype(
        np.float32,
        copy=False,
    )


__all__ = [
    "SOLVER_PHASOR_CONVENTION",
    "STANDARD_AUDIO_PHASOR_CONVENTION",
    "solver_phase_deg",
    "solver_to_standard_phasor",
    "standard_to_solver_phasor",
]
