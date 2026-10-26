"""Opt-in checks of the real Julia field worker and binary gradient protocol."""

import os

import numpy as np
import pytest

from blab.solvers.coupled_field import BemFieldEvaluationRequest, evaluate_bem_field


@pytest.mark.skipif(os.environ.get("BLAB_TEST_BEM_GRADIENT") != "1", reason="requires Julia CPU/CUDA runtime")
@pytest.mark.parametrize("backend", ["beat_cpu", "beat_cuda"])
def test_real_gradient_worker_matches_solid_angle_and_spatial_derivative(backend):
    from dataclasses import replace

    points = np.array([[0.2, 0.2, 0.02], [2, 3, 4]], dtype=float)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    request = BemFieldEvaluationRequest(
        domain_key="gradient-integration",
        points_m=points,
        boundary_points_m=vertices,
        boundary_triangles=np.array([[0, 1, 2]]),
        boundary_pressure=np.zeros(3, complex),
        boundary_normal_derivative=np.ones(1, complex),
        frequency_hz=1e-5,
        sound_speed_m_per_s=343,
        pressure_gradient=True,
    )
    result = evaluate_bem_field(request, backend_id=backend)
    assert result.shape == (2, 4)
    assert result.dtype == np.complex128
    for i, point in enumerate(points):
        a, b, c = vertices - point
        angle = abs(
            2
            * np.arctan2(
                np.dot(a, np.cross(b, c)),
                np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(c)
                + np.dot(a, b) * np.linalg.norm(c)
                + np.dot(b, c) * np.linalg.norm(a)
                + np.dot(c, a) * np.linalg.norm(b),
            )
        )
        assert result[i, 3].real == pytest.approx(angle / (4 * np.pi), rel=2e-4)
    request = replace(request, frequency_hz=100, boundary_pressure=np.array([1 + 1j, 2 - 1j, 0.5j]))
    result = evaluate_bem_field(request, backend_id=backend)
    for axis in range(3):
        shift = np.eye(3)[axis] * 1e-5
        shifted = evaluate_bem_field(
            replace(request, points_m=np.vstack((points + shift, points - shift))), backend_id=backend
        )
        np.testing.assert_allclose(result[:, axis + 1], (shifted[:2, 0] - shifted[2:, 0]) / 2e-5, rtol=3e-4)
