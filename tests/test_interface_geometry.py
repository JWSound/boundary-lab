import meshio
import numpy as np
import pytest

from blab.interface_conform import InterfaceConformError
from blab.interface_geometry import measure_interface_geometry


def square(*, offset=(0, 0, 0), refined=False):
    points = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.2, 0.3, 0]])
    triangles = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]] if refined else [[0, 1, 2], [0, 2, 3]]
    return meshio.Mesh(
        points + offset,
        [("triangle", triangles)],
        cell_data={"gmsh:physical": [np.ones(len(triangles), dtype=int)]},
        field_data={"Mouth": np.array([1, 2])},
    )


def measure(fem, bem):
    return measure_interface_geometry(fem, bem, fem_interface_name="Mouth", bem_interface_name="Mouth")


def test_independent_triangulations_have_identical_geometry_without_mutation():
    fem, bem = square(), square(refined=True)
    original = bem.points.copy()
    report = measure(fem, bem)
    assert report.fem.area_m2 == pytest.approx(1)
    assert report.bem.area_m2 == pytest.approx(1)
    assert len(report.fem.triangles) == 2
    assert len(report.bem.triangles) == 4
    assert report.sampled_perimeter_deviation_m == pytest.approx(0)
    assert report.centroid_normal_offset_m == pytest.approx(0)
    assert report.fem.plane_origin == pytest.approx(report.bem.plane_origin)
    assert report.plane_angle_deg == pytest.approx(0)
    np.testing.assert_array_equal(bem.points, original)


@pytest.mark.parametrize("offset", [(0, 0, 0.002), (0.1, 0, 0)])
def test_perimeter_witness_measures_normal_and_lateral_mismatch(offset):
    report = measure(square(), square(offset=offset))
    expected = np.linalg.norm(offset)
    assert report.sampled_perimeter_deviation_m == pytest.approx(expected)
    assert np.linalg.norm(np.diff(report.discrepancy_segment, axis=0)) == pytest.approx(expected)
    assert report.centroid_normal_offset_m == pytest.approx(abs(offset[2]))
    assert f"{expected * 1000:.6g} mm" in report.summary()


def test_tilted_planes_and_reversed_winding():
    fem, bem = square(), square()
    angle = np.pi / 6
    rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    bem.points = bem.points @ rotation.T
    bem.cells[0].data = bem.cells[0].data[:, ::-1]
    report = measure(fem, bem)
    assert report.plane_angle_deg == pytest.approx(30)
    assert report.bem.planarity_m == pytest.approx(0, abs=1e-14)
    assert report.bem.area_m2 == pytest.approx(1)


def test_closed_surface_reports_no_perimeter_instead_of_zero_error():
    mesh = meshio.Mesh(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [("triangle", [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])],
        cell_data={"gmsh:physical": [np.ones(4, dtype=int)]},
        field_data={"Mouth": np.array([1, 2])},
    )
    report = measure(mesh, mesh)
    assert report.sampled_perimeter_deviation_m is None
    assert report.discrepancy_segment is None
    assert "unavailable" in report.summary()


@pytest.mark.parametrize("bad", ["nonfinite", "degenerate", "missing"])
def test_unmeasurable_surface_has_explicit_error(bad):
    bem = square()
    if bad == "nonfinite":
        bem.points[0, 0] = np.nan
    elif bad == "degenerate":
        bem.points[0] = bem.points[1]
    else:
        bem.field_data = {}
    with pytest.raises(InterfaceConformError):
        measure(square(), bem)
