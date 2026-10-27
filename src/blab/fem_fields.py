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
    # None selects the full vector for volumetric Element Field arrows.
    tangential = velocity if normal is None else velocity - (velocity @ normal)[:, None] * np.asarray(normal)
    if phase_deg is None:
        vectors = np.real(np.asarray(pressure)[:, None] * np.conj(tangential))
        reference = np.linalg.norm(vectors, axis=1)
    else:
        vectors = np.sqrt(2.0) * np.real(tangential * np.exp(1j * np.deg2rad(phase_deg)))
        # Phase-independent bound prevents arrows being renormalized each frame.
        reference = np.sqrt(2.0 * np.sum(np.abs(tangential) ** 2, axis=1))
    return vectors, max(float(np.max(reference, initial=0.0)), 1e-30)


def flow_arrow_points(anchors, vectors, normal, spacing, reference):
    """Proportional arrows: three planar segments or five crossed 3D segments."""
    vectors = np.asarray(vectors)
    magnitude = np.linalg.norm(vectors, axis=1)
    delta = vectors * (0.8 * spacing / np.maximum(magnitude, reference))[:, None]
    tip = anchors + delta
    shoulder = tip - delta * 0.25
    if normal is None:
        direction = vectors / np.maximum(magnitude[:, None], 1e-30)
        axis = np.eye(3)[np.argmin(np.abs(direction), axis=1)]
        side = np.cross(direction, axis)
        side /= np.maximum(np.linalg.norm(side, axis=1)[:, None], 1e-30)
        other = np.cross(direction, side)
        width = np.linalg.norm(delta, axis=1)[:, None] * 0.16
        side, other = side * width, other * width
        return np.stack(
            (anchors, tip, tip, shoulder + side, tip, shoulder - side, tip, shoulder + other, tip, shoulder - other),
            axis=1,
        ).reshape(-1, 3)
    side = np.cross(normal, delta) * 0.16
    return np.stack((anchors, tip, tip, shoulder + side, tip, shoulder - side), axis=1).reshape(-1, 3)


def volume_arrow_grid(bounds, density, max_candidates=64000):
    """Uniform 3D anchor spacing with a bounded candidate count before cell lookup."""
    bounds = np.asarray(bounds).reshape(3, 2)
    extent = bounds[:, 1] - bounds[:, 0]
    if np.any(extent <= 0) or not np.all(np.isfinite(bounds)):
        return np.empty((0, 3)), 1.0
    spacing = float(np.max(extent) / density)
    shape = np.maximum(1, np.ceil(extent / spacing).astype(int))
    while np.prod(shape) > max_candidates:
        spacing *= max(1.01, (np.prod(shape) / max_candidates) ** (1 / 3))
        shape = np.maximum(1, np.ceil(extent / spacing).astype(int))
    axes = [bounds[d].mean() + (np.arange(shape[d]) - (shape[d] - 1) / 2) * spacing for d in range(3)]
    return np.column_stack([x.ravel() for x in np.meshgrid(*axes, indexing="ij")]), spacing
