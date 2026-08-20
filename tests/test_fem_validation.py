from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blab.fem_validation import (
    _ordered_split_surface_entities,
    compare_fem_validation_reports,
    frequency_mesh_resolution,
    quadratic_tetrahedral_shape_gradients,
    surface_pressure_metrics,
    tetrahedral_shape_gradients,
    tetrahedron_edge_statistics_m,
)


def _quadratic_tetrahedron_points() -> np.ndarray:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0))
    )
    edge_pairs = ((0, 1), (1, 2), (2, 0), (0, 3), (3, 2), (1, 3))
    return np.vstack((vertices, [0.5 * (vertices[left] + vertices[right]) for left, right in edge_pairs]))


def test_surface_pressure_metrics_identify_uniform_plane_mode() -> None:
    points = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)))
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    pressure = np.full(4, 2.0 + 3.0j)

    result = surface_pressure_metrics(points, triangles, pressure)

    assert result["area_m2"] == 1.0
    assert result["mean_pressure_pa"]["real"] == 2.0
    assert result["mean_pressure_pa"]["imag"] == 3.0
    assert np.isclose(result["plane_mode_fraction"], 1.0)
    assert np.isclose(result["pressure_coherence"], 1.0)
    assert result["phase_rms_deg"] < 1e-12
    assert result["magnitude_coefficient_of_variation"] < 1e-12


def test_surface_pressure_metrics_detect_nonuniform_phase() -> None:
    points = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)))
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    pressure = np.asarray((1.0 + 0.0j, 1.0j, -1.0 + 0.0j, -1.0j))

    result = surface_pressure_metrics(points, triangles, pressure)

    assert result["plane_mode_fraction"] < 0.1
    assert result["phase_rms_deg"] > 30.0


def test_quadratic_surface_pressure_metrics_identify_uniform_plane_mode() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.0, 0.5, 0.0),
        )
    )
    triangles = np.asarray(((0, 1, 2, 3, 4, 5),), dtype=np.int64)
    pressure = np.full(6, 2.0 + 3.0j)

    result = surface_pressure_metrics(points, triangles, pressure)

    assert np.isclose(result["area_m2"], 0.5)
    assert np.isclose(result["mean_pressure_pa"]["real"], 2.0)
    assert np.isclose(result["mean_pressure_pa"]["imag"], 3.0)
    assert np.isclose(result["plane_mode_fraction"], 1.0)
    assert result["phase_rms_deg"] < 1e-12


def test_tetrahedral_shape_gradients_reproduce_linear_pressure_gradient() -> None:
    points = np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)))
    tetrahedra = np.asarray(((0, 1, 2, 3),), dtype=np.int64)
    pressure = 2.0 * points[:, 0] - 3.0 * points[:, 1] + 4.0 * points[:, 2] + 5.0

    gradients = tetrahedral_shape_gradients(points, tetrahedra)
    reconstructed = np.einsum("ti,tij->tj", pressure[tetrahedra], gradients)

    np.testing.assert_allclose(reconstructed, ((2.0, -3.0, 4.0),), rtol=1e-12, atol=1e-12)


def test_quadratic_tetrahedral_shape_gradients_reproduce_linear_pressure_gradient() -> None:
    points = _quadratic_tetrahedron_points()
    tetrahedra = np.arange(10, dtype=np.int64)[np.newaxis, :]
    pressure = 2.0 * points[:, 0] - 3.0 * points[:, 1] + 4.0 * points[:, 2] + 5.0

    gradients = quadratic_tetrahedral_shape_gradients(
        points,
        tetrahedra,
        np.asarray(((0.1, 0.2, 0.3, 0.4),)),
    )
    reconstructed = np.einsum("ti,tij->tj", pressure[tetrahedra], gradients)

    np.testing.assert_allclose(reconstructed, ((2.0, -3.0, 4.0),), rtol=1e-12, atol=1e-12)


def test_quadratic_mesh_resolution_uses_corner_edges() -> None:
    points = _quadratic_tetrahedron_points()
    tetrahedra = np.arange(10, dtype=np.int64)[np.newaxis, :]

    result = tetrahedron_edge_statistics_m(points, tetrahedra)

    assert np.isclose(result["minimum"], 2.0)
    assert np.isclose(result["maximum"], 5.0)


def test_split_surface_entity_order_groups_rows_before_columns() -> None:
    triangles = np.empty((0, 3), dtype=np.int64)
    entities = [
        (4, triangles, np.asarray((1.0, 1.0 + 1.0e-9, 0.0))),
        (2, triangles, np.asarray((1.0, -1.0 - 1.0e-9, 0.0))),
        (3, triangles, np.asarray((-1.0, 1.0, 0.0))),
        (1, triangles, np.asarray((-1.0, -1.0, 0.0))),
    ]

    ordered = _ordered_split_surface_entities(entities)

    assert [item[0] for item in ordered] == [1, 2, 3, 4]


def test_mesh_resolution_reports_points_per_wavelength_and_gate() -> None:
    points = np.asarray(((0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)))
    tetrahedra = np.asarray(((0, 1, 2, 3),), dtype=np.int64)

    statistics = tetrahedron_edge_statistics_m(points, tetrahedra)
    resolution = frequency_mesh_resolution(statistics, 1000.0, 340.0, 8.0, 4.0)

    assert np.isclose(statistics["minimum"], 0.01)
    assert np.isclose(statistics["maximum"], np.sqrt(2.0) * 0.01)
    assert resolution["points_per_wavelength_at_p95_edge"] > 20.0
    assert resolution["adequate"] is True

    under_resolved = frequency_mesh_resolution(statistics, 10_000.0, 340.0, 8.0, 4.0)
    assert under_resolved["points_per_wavelength_at_maximum_edge"] < 4.0
    assert under_resolved["adequate"] is False


def test_convergence_comparison_removes_common_phase_and_amplitude(tmp_path: Path) -> None:
    def report(phase_offsets_deg: tuple[float, float], scale: float) -> dict:
        surfaces = []
        for name, phase in zip(("exit_a", "exit_b"), phase_offsets_deg, strict=True):
            pressure = scale * np.exp(1j * np.radians(phase))
            surfaces.append(
                {
                    "name": name,
                    "area_m2": 1.0,
                    "mean_pressure_pa": {
                        "real": float(pressure.real),
                        "imag": float(pressure.imag),
                    },
                    "plane_mode_fraction": 0.99,
                }
            )
        return {
            "schema": "boundary-lab-fem-validation",
            "results": [
                {
                    "frequency_hz": 1000.0,
                    "excitation_port_id": "excitation:test",
                    "surfaces": surfaces,
                }
            ],
        }

    coarse = tmp_path / "coarse.json"
    fine = tmp_path / "fine.json"
    coarse.write_text(json.dumps(report((10.0, 10.0), 2.0)), encoding="utf-8")
    fine.write_text(json.dumps(report((40.2, 39.8), 7.0)), encoding="utf-8")

    comparison = compare_fem_validation_reports(coarse, fine)

    assert comparison["passed"] is True
    result = comparison["comparisons"][0]
    assert result["surface_phase_rms_delta_deg"] < 0.3
    assert result["normalized_amplitude_rms_delta"] < 1e-12
