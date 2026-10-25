"""Qt-free sampling and plane projection of retained RMS acoustic fields."""

from __future__ import annotations

import numpy as np


def sample_pressure_velocity(pressure, cells, gradients, densities, frequency_hz, cell_ids, barycentric):
    """Evaluate affine P1/P2 elements; connectivity uses meshio/VTK node order."""
    ids = np.asarray(cell_ids, dtype=np.int64)
    bary = np.asarray(barycentric, dtype=float)
    nodes = np.asarray(cells)[ids]
    grad = np.asarray(gradients)[ids]
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("Velocity evaluation requires positive finite frequency.")
    if bary.shape != (len(ids), 4):
        raise ValueError("Field samples require four barycentric coordinates.")
    if nodes.shape[1] == 10:
        basis = np.empty((len(ids), 10))
        basis[:, :4] = bary * (2 * bary - 1)
        derivatives = np.empty((len(ids), 10, 3))
        derivatives[:, :4] = (4 * bary - 1)[:, :, None] * grad
        for offset, (a, b) in enumerate(((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))):
            basis[:, 4 + offset] = 4 * bary[:, a] * bary[:, b]
            derivatives[:, 4 + offset] = 4 * (bary[:, a, None] * grad[:, b] + bary[:, b, None] * grad[:, a])
    else:
        basis, derivatives = bary, grad
    nodal = np.asarray(pressure)[nodes]
    sampled = np.einsum("si,si->s", nodal, basis)
    gradient = np.einsum("si,sij->sj", nodal, derivatives)
    velocity = gradient / (-1j * 2 * np.pi * frequency_hz * np.asarray(densities)[ids, None])
    return sampled, velocity


def plane_flow_vectors(pressure, velocity, normal, phase_deg=None):
    """Active intensity for RMS phasors, or instantaneous physical velocity."""
    velocity = np.asarray(velocity)
    normal = np.asarray(normal)
    tangential = velocity - (velocity @ normal)[:, None] * normal
    if phase_deg is None:
        vectors = np.real(np.asarray(pressure)[:, None] * np.conj(tangential))
        reference = np.linalg.norm(vectors, axis=1)
    else:
        vectors = np.sqrt(2.0) * np.real(tangential * np.exp(1j * np.deg2rad(phase_deg)))
        # Phase-independent bound prevents arrows being renormalized each frame.
        reference = np.sqrt(2.0 * np.sum(np.abs(tangential) ** 2, axis=1))
    return vectors, max(float(np.max(reference, initial=0.0)), 1e-30)


def flow_arrow_points(anchors, vectors, normal, spacing, reference):
    """Three line segments per arrow, anchored at the tail, with proportional length."""
    vectors = np.asarray(vectors)
    magnitude = np.linalg.norm(vectors, axis=1)
    delta = vectors * (0.8 * spacing / np.maximum(magnitude, reference))[:, None]
    tip = anchors + delta
    side = np.cross(normal, delta) * 0.16
    shoulder = tip - delta * 0.25
    return np.stack((anchors, tip, tip, shoulder + side, tip, shoulder - side), axis=1).reshape(-1, 3)
