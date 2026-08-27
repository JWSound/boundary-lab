from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from blab.deploy_geometry import minimum_surface_distance, surface_face_pairs_within
from blab.deploy_solve import (
    CLOSE_PAIR_DISTANCE_M,
    CLOSE_PAIR_QUADRATURE_ORDER,
    DEPLOY_FIELD_SCHEMA,
    DEPLOY_SOLVE_SCHEMA,
    SOURCE_SURFACE_PADDING_M,
    DeploySolveCache,
    prepare_deploy_field_request,
    prepare_deploy_solve_request,
)

PACKAGE_PATH = Path(__file__).parents[1] / "deploy" / "library" / "S218BP_LOD.blabsp"


def _payload() -> dict:
    return {
        "packagePath": str(PACKAGE_PATH),
        "frequencyHz": 80.9498519897461,
        "backend": "cuda",
        "sources": [
            {
                "id": "subwoofer-1",
                "positionX": 1.25,
                "positionHeightM": 0.4,
                "positionZ": -0.5,
                "yawDeg": 12.0,
                "levelDb": -6.0,
                "delayMs": 1.5,
                "polarity": -1,
            },
            {
                "id": "subwoofer-2",
                "positionX": 5.0,
                "positionHeightM": 0.4,
                "positionZ": -0.5,
                "yawDeg": 0.0,
                "levelDb": -3.0,
                "delayMs": 0.0,
                "polarity": 1,
            },
        ],
        "observation": {
            "widthM": 12.0,
            "depthM": 10.0,
            "nearM": 2.0,
            "heightM": 1.2,
            "columns": 7,
            "rows": 5,
        },
    }


def test_prepare_deploy_solve_request_stages_lod_trace_and_grid(tmp_path: Path) -> None:
    request_path, request = prepare_deploy_solve_request(_payload(), tmp_path)

    assert request_path.is_file()
    assert json.loads(request_path.read_text(encoding="utf-8"))["schema"] == DEPLOY_SOLVE_SCHEMA
    assert request["beat_engine_backend"] == "cuda"
    assert request["burton_miller_assembly"] == "direct_system"
    assert request["provenance"]["package_name"] == "S218BP LOD"
    assert request["provenance"]["source_count"] == 2
    assert request["provenance"]["node_count"] == 2580
    assert request["provenance"]["face_count"] == 5152
    assert Path(request["mesh_file"]).is_file()
    assert len(request["boundary_neumann"]["real"]) == 5152
    assert len(request["reference_boundary_pressure"]["real"]) == 2580
    assert len(request["observation_points_m"]) == 35
    assert request["observation_shape"] == [5, 7]
    assert request["observation_points_m"][0] == pytest.approx([-6.0, 1.2, 2.0])
    assert request["observation_points_m"][-1] == pytest.approx([6.0, 1.2, 12.0])
    assert request["source_transforms"][0] == {
        "id": "subwoofer-1",
        "position_m": [1.25, 0.4, -0.5],
        "pitch_deg": 0.0,
        "yaw_deg": 12.0,
        "roll_deg": 0.0,
    }
    assert request["proximity"]["surface_padding_m"] == SOURCE_SURFACE_PADDING_M
    assert request["close_pair_quadrature_order"] == CLOSE_PAIR_QUADRATURE_ORDER
    assert request["proximity"]["close_face_pairs"] == []
    assert request["proximity"]["ground_image_close_face_pairs"] == []
    assert request["proximity"]["minimum_surface_distance_m"] > 2.0
    assert request["boundary"]["ground_plane"] == {
        "type": "rigid_half_space",
        "axis": "y",
        "offset_m": 0.0,
        "reflection_coefficient": 1.0,
    }
    assert request["provenance"]["exterior_domain"] == "rigid_y0_half_space"

    with zipfile.ZipFile(PACKAGE_PATH, "r") as archive:
        with np.load(io.BytesIO(archive.read("data/fixed-sources.npz")), allow_pickle=False) as fixed:
            source_q = np.asarray(fixed["normal_derivative_pa_per_m"])[84, 0]
    phase = 2.0 * np.pi * request["frequency_hz"] * 1.5 / 1000.0
    expected = np.asarray(source_q * (-1.0) * 10.0 ** (-6.0 / 20.0) * np.exp(1j * phase), dtype=np.complex64)
    actual = np.asarray(request["boundary_neumann"]["real"][:2576], dtype=np.float32) + 1j * np.asarray(
        request["boundary_neumann"]["imag"][:2576], dtype=np.float32
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)


def test_prepare_deploy_solve_request_can_select_operator_matrix_fallback(tmp_path: Path) -> None:
    payload = _payload()
    payload["burtonMillerAssembly"] = "operator_matrices"

    _, request = prepare_deploy_solve_request(payload, tmp_path)

    assert request["burton_miller_assembly"] == "operator_matrices"


def test_prepare_deploy_solve_request_reuses_package_and_ground_pair_cache(tmp_path: Path) -> None:
    cache = DeploySolveCache()

    _, first = prepare_deploy_solve_request(_payload(), tmp_path / "first", cache=cache)
    package_data = next(iter(cache.packages.values()))
    ground_pairs = dict(cache.ground_image_pairs)
    _, second = prepare_deploy_solve_request(_payload(), tmp_path / "second", cache=cache)

    assert next(iter(cache.packages.values())) is package_data
    assert cache.ground_image_pairs.keys() == ground_pairs.keys()
    assert all(cache.ground_image_pairs[key] is value for key, value in ground_pairs.items())
    assert first["proximity"]["ground_image_close_face_pairs"] == second["proximity"][
        "ground_image_close_face_pairs"
    ]


def test_prepare_deploy_field_request_contains_only_observation_data(tmp_path: Path) -> None:
    payload = _payload()
    payload["solutionKey"] = "package-frequency-and-source-state"
    payload["includeComplexPressure"] = False

    request_path, request = prepare_deploy_field_request(payload, tmp_path)

    assert json.loads(request_path.read_text(encoding="utf-8")) == request
    assert request["schema"] == DEPLOY_FIELD_SCHEMA
    assert request["solution_key"] == payload["solutionKey"]
    assert request["observation_plane"] == {
        "width_m": 12.0,
        "depth_m": 10.0,
        "center_x_m": 0.0,
        "near_m": 2.0,
        "height_m": 1.2,
        "pitch_deg": 0.0,
        "yaw_deg": 0.0,
        "roll_deg": 0.0,
        "columns": 7,
        "rows": 5,
        "ground_tolerance_m": 1e-6,
    }
    assert request["include_complex_pressure"] is False
    assert "observation_points_m" not in request
    assert "observation_sample_indices" not in request
    assert "mesh_file" not in request
    assert "boundary_neumann" not in request
    assert "proximity" not in request


def test_prepare_deploy_cpu_field_request_keeps_explicit_points(tmp_path: Path) -> None:
    payload = _payload()
    payload.update({"solutionKey": "cpu-boundary", "backend": "cpu"})

    _, request = prepare_deploy_field_request(payload, tmp_path)

    assert len(request["observation_points_m"]) == 35
    assert request["observation_shape"] == [5, 7]
    assert request["observation_sample_indices"] == list(range(35))
    assert "observation_plane" not in request


def test_prepare_deploy_solve_request_requires_exact_exported_frequency(tmp_path: Path) -> None:
    payload = _payload()
    payload["frequencyHz"] = 81.25

    with pytest.raises(ValueError, match="exact exported package frequency"):
        prepare_deploy_solve_request(payload, tmp_path)


def test_prepare_deploy_solve_request_rejects_invalid_observation_shape(tmp_path: Path) -> None:
    payload = _payload()
    payload["observation"]["columns"] = 1

    with pytest.raises(ValueError, match="at least two rows and columns"):
        prepare_deploy_solve_request(payload, tmp_path)


def test_prepare_deploy_solve_request_transforms_observation_plane(tmp_path: Path) -> None:
    payload = _payload()
    payload["observation"].update({"centerXM": 3.0, "yawDeg": 90.0})

    request_path, _ = prepare_deploy_solve_request(payload, tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))

    assert request["observation_points_m"][0] == pytest.approx([-2.0, 1.2, 13.0])
    assert request["observation_points_m"][-1] == pytest.approx([8.0, 1.2, 1.0])


def test_prepare_deploy_solve_request_pitches_observation_plane(tmp_path: Path) -> None:
    payload = _payload()
    payload["observation"].update({"heightM": 6.0, "pitchDeg": 90.0})

    request_path, _ = prepare_deploy_solve_request(payload, tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))

    assert request["observation_points_m"][0] == pytest.approx([-6.0, 11.0, 7.0])
    assert request["observation_points_m"][-1] == pytest.approx([6.0, 1.0, 7.0])


def test_prepare_deploy_solve_request_omits_points_below_ground(tmp_path: Path) -> None:
    payload = _payload()
    payload["observation"].update({"heightM": 0.0, "pitchDeg": 30.0})

    request_path, _ = prepare_deploy_solve_request(payload, tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))

    assert len(request["observation_points_m"]) < 35
    assert len(request["observation_points_m"]) == len(request["observation_sample_indices"])
    assert all(point[1] >= -1e-6 for point in request["observation_points_m"])


def test_prepare_deploy_solve_request_rolls_observation_plane(tmp_path: Path) -> None:
    payload = _payload()
    payload["observation"].update({"heightM": 6.0, "rollDeg": 90.0})

    request_path, _ = prepare_deploy_solve_request(payload, tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))

    assert request["observation_points_m"][0] == pytest.approx([0.0, 0.0, 2.0], abs=1e-6)
    assert request["observation_points_m"][-1] == pytest.approx([0.0, 12.0, 12.0], abs=1e-6)


def test_prepare_deploy_solve_request_rejects_geometry_below_ground(tmp_path: Path) -> None:
    payload = _payload()
    payload["sources"][0]["positionHeightM"] = 0.1

    with pytest.raises(ValueError, match="below the ground plane"):
        prepare_deploy_solve_request(payload, tmp_path)


def test_prepare_deploy_solve_request_rejects_plane_with_no_above_ground_samples(tmp_path: Path) -> None:
    payload = _payload()
    payload["observation"]["heightM"] = -0.01

    with pytest.raises(ValueError, match="no sampling points on or above"):
        prepare_deploy_solve_request(payload, tmp_path)


def test_prepare_deploy_solve_request_enforces_surface_padding(tmp_path: Path) -> None:
    payload = _payload()
    payload["sources"][1].update(payload["sources"][0])
    payload["sources"][1]["id"] = "subwoofer-2"

    with pytest.raises(ValueError, match="surface spacing"):
        prepare_deploy_solve_request(payload, tmp_path)


def test_prepare_deploy_solve_request_rejects_old_two_mm_padding_target(tmp_path: Path) -> None:
    payload = _payload()
    payload["sources"][0].update(
        {"positionX": 0.0, "positionHeightM": 0.4, "positionZ": 0.0, "yawDeg": 0.0}
    )
    payload["sources"][1].update(
        {"positionX": 1.202, "positionHeightM": 0.4, "positionZ": 0.0, "yawDeg": 0.0}
    )

    with pytest.raises(ValueError, match="at least 10.0 mm"):
        prepare_deploy_solve_request(payload, tmp_path)


def test_bvh_surface_distance_for_parallel_triangles() -> None:
    first = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    second = first + np.asarray([0, 0, 0.125])
    triangles = np.asarray([[0, 1, 2]], dtype=np.int64)

    result = minimum_surface_distance(first, triangles, second, triangles)

    assert result.distance_m == pytest.approx(0.125)
    assert (result.face_a, result.face_b) == (0, 0)


def test_bvh_emits_every_pair_at_inclusive_close_pair_threshold() -> None:
    first = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]],
        dtype=float,
    )
    triangles = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    second = first + np.asarray([0, 0, CLOSE_PAIR_DISTANCE_M])

    pairs = surface_face_pairs_within(first, triangles, second, triangles, CLOSE_PAIR_DISTANCE_M)

    assert [(pair.face_a, pair.face_b) for pair in pairs] == [(0, 0), (1, 1)]
    assert all(pair.distance_m == pytest.approx(CLOSE_PAIR_DISTANCE_M) for pair in pairs)


def test_bvh_close_pair_threshold_excludes_faces_just_outside() -> None:
    first = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    triangles = np.asarray([[0, 1, 2]], dtype=np.int64)
    second = first + np.asarray([0, 0, CLOSE_PAIR_DISTANCE_M + 1e-5])

    assert surface_face_pairs_within(first, triangles, second, triangles, CLOSE_PAIR_DISTANCE_M) == []


def test_prepare_deploy_solve_request_emits_directed_close_face_pairs(tmp_path: Path) -> None:
    payload = _payload()
    # Package X bounds are +/- 0.6 m; these centers produce a 10 mm surface gap.
    payload["sources"][0].update(
        {"positionX": 0.0, "positionHeightM": 0.4, "positionZ": 0.0, "yawDeg": 0.0}
    )
    payload["sources"][1].update(
        {"positionX": 1.21, "positionHeightM": 0.4, "positionZ": 0.0, "yawDeg": 0.0}
    )

    _, request = prepare_deploy_solve_request(payload, tmp_path)

    cabinet_pair = request["proximity"]["pairs"][0]
    directed_pairs = request["proximity"]["close_face_pairs"]
    assert cabinet_pair["close"] is True
    assert cabinet_pair["near_face_pair_count"] > 1
    assert len(directed_pairs) == 2 * cabinet_pair["near_face_pair_count"]
    directed_set = {tuple(pair) for pair in directed_pairs}
    assert all((trial, test, order) in directed_set for test, trial, order in directed_set)
    assert {order for _, _, order in directed_set}.issubset({4, 6, 8})


def test_prepare_deploy_solve_request_emits_ground_image_close_pairs(tmp_path: Path) -> None:
    payload = _payload()
    payload["sources"] = [payload["sources"][0]]
    # The LOD package's local floor is -0.2695 m. A 0.2795 m center height
    # creates a 10 mm ground clearance and a 20 mm boundary-to-image gap.
    payload["sources"][0]["positionHeightM"] = 0.2795

    _, request = prepare_deploy_solve_request(payload, tmp_path)

    ground_pairs = request["proximity"]["ground_image_close_face_pairs"]
    assert ground_pairs
    assert {order for _, _, order in ground_pairs}.issubset({4, 6, 8})
    assert 6 in {order for _, _, order in ground_pairs}
