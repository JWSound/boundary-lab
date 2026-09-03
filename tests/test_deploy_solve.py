from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import blab.deploy_solve as deploy_solve_module
from blab.deploy_geometry import minimum_surface_distance, surface_face_pairs_within
from blab.deploy_solve import (
    CLOSE_PAIR_DISTANCE_M,
    CLOSE_PAIR_QUADRATURE_ORDER,
    DEPLOY_FIELD_SCHEMA,
    DEPLOY_MICROPHONE_SWEEP_SCHEMA,
    DEPLOY_SOLVE_SCHEMA,
    SOURCE_SURFACE_PADDING_M,
    DeploySolveCache,
    _combined_excitation_trace,
    _logical_excitation_indices,
    prepare_deploy_coupled_request,
    prepare_deploy_field_request,
    prepare_deploy_microphone_sweep_request,
    prepare_deploy_rom_microphone_sweep_request,
    prepare_deploy_rom_request,
    prepare_deploy_solve_request,
)

PACKAGE_PATH = Path(__file__).parents[1] / "deploy" / "library" / "S218BP_LOD.blabsp"
RIGID_MESH_PATH = Path(__file__).parents[1] / "deploy" / "library" / "RigidStage_LOD.msh"
with zipfile.ZipFile(PACKAGE_PATH, "r") as _package_archive:
    PACKAGE_FREQUENCIES = json.loads(_package_archive.read("manifest.json"))["frequencies_hz"]
TEST_FREQUENCY_INDEX = 43
TEST_FREQUENCY_HZ = float(PACKAGE_FREQUENCIES[TEST_FREQUENCY_INDEX])


def _payload() -> dict:
    return {
        "packagePath": str(PACKAGE_PATH),
        "frequencyHz": TEST_FREQUENCY_HZ,
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
    assert request["provenance"]["package_name"] == "S218BP"
    assert request["provenance"]["source_count"] == 2
    assert request["provenance"]["node_count"] == 2580
    assert request["provenance"]["face_count"] == 5152
    assert Path(request["mesh_file"]).is_file()
    assert len(request["boundary_neumann"]["real"]) == 5152
    assert len(request["reference_boundary_pressure"]["real"]) == 2580
    assert "observation_points_m" not in request
    assert "observation_shape" not in request
    assert request["observation_plane"]["columns"] == 7
    assert request["observation_plane"]["rows"] == 5
    assert request["mesh_is_world_space"] is True
    assert request["source_transforms"][0] == {
        "id": "combined-boundary",
        "position_m": [0.0, 0.0, 0.0],
        "pitch_deg": 0.0,
        "yaw_deg": 0.0,
        "roll_deg": 0.0,
    }
    assert [component["id"] for component in request["boundary_components"]] == [
        "subwoofer-1",
        "subwoofer-2",
    ]
    assert set(request["reference_boundary_pressure_mask"]) == {1}
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
            source_q = np.sum(
                np.asarray(fixed["normal_derivative_pa_per_m"])[TEST_FREQUENCY_INDEX, [0, 1]],
                axis=0,
            )
    phase = 2.0 * np.pi * request["frequency_hz"] * 1.5 / 1000.0
    expected = np.asarray(source_q * (-1.0) * 10.0 ** (-6.0 / 20.0) * np.exp(1j * phase), dtype=np.complex64)
    actual = np.asarray(request["boundary_neumann"]["real"][:2576], dtype=np.float32) + 1j * np.asarray(
        request["boundary_neumann"]["imag"][:2576], dtype=np.float32
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    assert request["provenance"]["excitation_indices"] == [0, 1]
    assert request["provenance"]["excitation_port_ids"] == [
        "excitation:component-18ds115-4",
        "excitation:component-18ds115-4__reflect_x",
    ]


def test_prepare_deploy_solve_request_mutes_source_drive(tmp_path: Path) -> None:
    payload = _payload()
    payload["sources"][0]["muted"] = True

    _, request = prepare_deploy_solve_request(payload, tmp_path)

    assert request["boundary_neumann"]["real"][:2576] == [0.0] * 2576
    assert request["boundary_neumann"]["imag"][:2576] == [0.0] * 2576


def test_logical_excitation_grouping_preserves_independent_and_legacy_ports() -> None:
    manifest = {
        "excitation_port_ids": ["port:a", "port:a-reflected", "port:b"],
        "physical_system": {
            "metadata": {
                "speaker_export_symmetry_expansion": {
                    "excitation_port_source_ids": {
                        "port:a": "port:a",
                        "port:a-reflected": "port:a",
                        "port:b": "port:b",
                    }
                }
            }
        },
    }

    assert _logical_excitation_indices(manifest, 3) == (0, 1)
    assert _logical_excitation_indices(manifest, 3, selected_index=2) == (2,)
    assert _logical_excitation_indices({"excitation_port_ids": ["port:a", "port:b"]}, 2) == (0,)

    traces = np.asarray([[[1, 2], [10, 20], [100, 200]]], dtype=np.complex64)
    np.testing.assert_array_equal(_combined_excitation_trace(traces, 0, (0, 1)), [11, 22])
    np.testing.assert_array_equal(_combined_excitation_trace(traces, 0, (2,)), [100, 200])


def test_prepare_deploy_solve_request_adds_rigid_zero_neumann_boundary(tmp_path: Path) -> None:
    payload = _payload()
    payload["rigidObjects"] = [
        {
            "id": "stage-1",
            "meshPath": str(RIGID_MESH_PATH),
            "positionX": 8.0,
            "positionHeightM": 0.01,
            "positionZ": 0.0,
            "pitchDeg": 0.0,
            "yawDeg": 0.0,
            "rollDeg": 0.0,
        }
    ]

    _, request = prepare_deploy_solve_request(payload, tmp_path)

    assert request["provenance"]["rigid_object_count"] == 1
    assert request["provenance"]["boundary_component_count"] == 3
    assert request["provenance"]["node_count"] == 2588
    assert request["provenance"]["face_count"] == 5164
    assert request["boundary_components"][-1]["id"] == "stage-1"
    assert request["boundary_components"][-1]["kind"] == "rigid"
    assert request["boundary_components"][-1]["vertex_count"] == 8
    assert request["boundary_components"][-1]["face_count"] == 12
    assert request["boundary_neumann"]["real"][-12:] == [0.0] * 12
    assert request["boundary_neumann"]["imag"][-12:] == [0.0] * 12
    assert request["reference_boundary_pressure_mask"][-8:] == [0] * 8
    assert any(pair["kind_b"] == "rigid" for pair in request["proximity"]["pairs"])


def test_prepare_deploy_solve_request_enforces_speaker_rigid_padding(tmp_path: Path) -> None:
    payload = _payload()
    payload["sources"] = [payload["sources"][1]]
    payload["rigidObjects"] = [
        {
            "id": "stage-1",
            "meshPath": str(RIGID_MESH_PATH),
            "scaleToMeters": 0.001,
            # The speaker's right side is x=5.6 m and the stage's left side is
            # x=5.609 m, leaving only 9 mm between overlapping side surfaces.
            "positionX": 7.609,
            "positionHeightM": 0.01,
            "positionZ": -0.5,
        }
    ]

    with pytest.raises(ValueError, match=r"stage-1.*9\.000 mm surface spacing"):
        prepare_deploy_solve_request(payload, tmp_path)


def test_prepare_deploy_solve_request_rejects_open_rigid_mesh(tmp_path: Path) -> None:
    lines = RIGID_MESH_PATH.read_text(encoding="utf-8").splitlines()
    element_header = lines.index("$Elements")
    lines[element_header + 1] = "11"
    lines.pop(lines.index("12 2 2 1 1 4 5 8"))
    mesh_path = tmp_path / "open-stage.msh"
    mesh_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = _payload()
    payload["rigidObjects"] = [{"id": "stage-1", "meshPath": str(mesh_path)}]

    with pytest.raises(ValueError, match="closed two-manifold"):
        prepare_deploy_solve_request(payload, tmp_path)


def test_prepare_deploy_solve_request_accepts_microphone_observation_points(tmp_path: Path) -> None:
    payload = _payload()
    payload["observationPointsM"] = [[1.0, 1.2, 4.0], [-2.0, 0.0, 8.0]]
    payload["includeComplexPressure"] = True

    request_path, _ = prepare_deploy_solve_request(payload, tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))

    np.testing.assert_allclose(request["observation_points_m"], payload["observationPointsM"])
    assert request["observation_shape"] == [1, 2]
    assert request["observation_sample_indices"] == [0, 1]
    assert request["include_complex_pressure"] is True
    assert "observation_plane" not in request


def test_prepare_deploy_solve_request_rejects_microphone_below_ground(tmp_path: Path) -> None:
    payload = _payload()
    payload["observationPointsM"] = [[0.0, -0.01, 4.0]]

    with pytest.raises(ValueError, match="below the ground"):
        prepare_deploy_solve_request(payload, tmp_path)


def test_prepare_microphone_sweep_stages_geometry_and_all_frequencies_once(tmp_path: Path) -> None:
    payload = _payload()
    payload["observationPointsM"] = [[1.0, 1.2, 4.0], [-2.0, 0.0, 8.0]]
    statuses: list[str] = []
    cache = DeploySolveCache()

    request_path, request = prepare_deploy_microphone_sweep_request(
        payload,
        tmp_path,
        cache=cache,
        status_callback=statuses.append,
    )

    assert json.loads(request_path.read_text(encoding="utf-8"))["schema"] == DEPLOY_MICROPHONE_SWEEP_SCHEMA
    assert len(request["frequencies_hz"]) == len(PACKAGE_FREQUENCIES)
    assert len(request["boundary_neumann_sweep"]["real"]) == len(PACKAGE_FREQUENCIES)
    assert len(request["boundary_neumann_sweep"]["real"][0]) == 5152
    assert len(request["reference_boundary_pressure_sweep"]["real"][0]) == 2580
    assert "boundary_neumann" not in request
    assert "reference_boundary_pressure" not in request
    np.testing.assert_allclose(request["observation_points_m"], payload["observationPointsM"])
    assert statuses[:3] == [
        "Preparing scene geometry",
        "Validating boundary spacing",
        "Building close-pair correction map",
    ]
    assert statuses[-1] == "Serializing multi-frequency BEAT request"

    repeated_statuses: list[str] = []
    repeated_payload = {**payload, "observationPointsM": [[3.0, 1.0, 5.0]]}
    _, repeated = prepare_deploy_microphone_sweep_request(
        repeated_payload,
        tmp_path / "repeated",
        cache=cache,
        status_callback=repeated_statuses.append,
    )
    assert repeated_statuses[0] == "Reusing prepared scene geometry"
    np.testing.assert_allclose(repeated["observation_points_m"], [[3.0, 1.0, 5.0]])
    assert Path(repeated["mesh_file"]).is_file()


def test_prepare_rom_microphone_sweep_batches_frequency_arrays_and_delay_drives(
    monkeypatch, tmp_path: Path
) -> None:
    array_names = ("k", "c", "d", "b", "e", "velocity", "current", "velocity_drive", "current_drive")
    arrays = {name: np.asarray([[[1.0 + 0.0j]], [[2.0 + 0.0j]]], dtype=np.complex64) for name in array_names}
    arrays["frequencies_hz"] = np.asarray([20.0, 40.0])
    package = type("Package", (), {
        "fingerprint": ("speaker.blabsp", 1, 1),
        "frequencies": np.asarray([10.0, 20.0, 40.0, 80.0]),
        "coupled_model": {
            "representation": "parity_petrov_galerkin_rom",
            "arrays": arrays,
        },
    })()
    cache = DeploySolveCache()
    monkeypatch.setattr(cache, "load_package", lambda _path: package)

    def prepare_single(payload, work_dir, **_kwargs):
        path = Path(work_dir) / "request.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        request = {
            "schema": "boundary_lab_deploy_rom",
            "frequency_hz": payload["frequencyHz"],
            "boundary_neumann": {"real": [0.0], "imag": [0.0]},
            "reference_boundary_pressure": {"real": [0.0], "imag": [0.0]},
            "rom": {
                "format_version": 1,
                "representation": "parity_petrov_galerkin_rom",
                "rank_per_sector": 1,
                "sector_signs": [[1, 1], [-1, 1], [1, -1], [-1, -1]],
                "node_orbits": [[0, 0, 0, 0]],
                "face_orbits": [[0, 0, 0, 0]],
                "instances": [{"id": "source", "node_offset": 0, "face_offset": 0}],
                "binary_arrays": {},
                "gmres_tolerance": 1e-4,
                "gmres_max_iterations": 30,
            },
            "provenance": {},
        }
        path.write_text(json.dumps(request), encoding="utf-8")
        return path, request

    monkeypatch.setattr(deploy_solve_module, "prepare_deploy_rom_request", prepare_single)
    request_path, request = prepare_deploy_rom_microphone_sweep_request(
        {
            "packagePath": "speaker.blabsp",
            "sources": [{"id": "source", "delayMs": 12.5}],
            "observationPointsM": [[0.0, 1.2, 4.0]],
        },
        tmp_path,
        cache=cache,
    )

    assert json.loads(request_path.read_text(encoding="utf-8"))["schema"] == DEPLOY_MICROPHONE_SWEEP_SCHEMA
    assert request["frequencies_hz"] == [20.0, 40.0]
    assert len(request["rom_sweep"]["frequencies"]) == 2
    first_drive = complex(
        request["rom_sweep"]["frequencies"][0]["instances"][0]["input_real"][0],
        request["rom_sweep"]["frequencies"][0]["instances"][0]["input_imag"][0],
    )
    second_drive = complex(
        request["rom_sweep"]["frequencies"][1]["instances"][0]["input_real"][0],
        request["rom_sweep"]["frequencies"][1]["instances"][0]["input_imag"][0],
    )
    assert first_drive == pytest.approx(2.83j, abs=1e-6)
    assert second_drive == pytest.approx(-2.83 + 0j, abs=1e-6)
    staged_path = Path(request["rom_sweep"]["frequencies"][1]["binary_arrays"]["k"]["file"])
    assert staged_path.is_file()
    assert request["provenance"]["rom_sweep_stage_cache_hit"] == 0
    assert request["provenance"]["rom_sweep_stage_binary_bytes_written"] > 0

    repeated_statuses: list[str] = []
    _, repeated = prepare_deploy_rom_microphone_sweep_request(
        {
            "packagePath": "speaker.blabsp",
            "sources": [{"id": "source", "delayMs": 25.0}],
            "observationPointsM": [[2.0, 1.2, 6.0]],
        },
        tmp_path / "repeated",
        cache=cache,
        status_callback=repeated_statuses.append,
    )

    repeated_path = Path(repeated["rom_sweep"]["frequencies"][1]["binary_arrays"]["k"]["file"])
    assert repeated_path == staged_path
    assert repeated["provenance"]["rom_sweep_stage_cache_hit"] == 1
    assert repeated["provenance"]["rom_sweep_stage_binary_bytes_written"] == 0
    assert "Reusing staged Level 3 ROM sweep data" in repeated_statuses
    cache.close()


def test_prepare_rom_request_retains_scene_speaker_and_transducer_identity(tmp_path: Path) -> None:
    _path, request = prepare_deploy_rom_request(_payload(), tmp_path)

    assert request["speakers"] == [
        {"id": "subwoofer-1", "name": "subwoofer-1"},
        {"id": "subwoofer-2", "name": "subwoofer-2"},
    ]
    assert len(request["transducers"]) == 4
    assert request["transducers"][0]["id"] == "subwoofer-1:component:18ds115-4"
    assert request["transducers"][0]["name"] == "subwoofer-1 / 18DS115-8"


def test_prepare_exact_coupled_request_batches_microphones_and_frequency_weights(
    monkeypatch, tmp_path: Path
) -> None:
    package = type("Package", (), {
        "coupled_model": {
            "representation": "exact_frequency_parametric_fem",
            "frequency_band_hz": [20.0, 80.0],
        },
        "manifest": {"files": {"coupled_model": {"representation": "exact_frequency_parametric_fem"}}},
    })()
    cache = type("Cache", (), {
        "load_package": lambda self, _path: package,
        "load_rigid_mesh": lambda self, _path: None,
    })()
    compiled_system = {
        "id": "base",
        "name": "Base",
        "meshes": [{"id": "mesh:exterior", "file": str(tmp_path / "base.msh"), "scale_to_m": 1.0}],
        "regions": [{"id": "region:exterior", "kind": "unbounded_air", "mesh_ids": ["mesh:exterior"]}],
        "boundaries": [{
            "id": "boundary:exterior",
            "region_id": "region:exterior",
            "kind": "rigid",
            "group": {"mesh_id": "mesh:exterior", "dimension": 2, "tag": 1},
            "parameters": {},
        }],
        "interfaces": [],
        "components": [{"id": "component:driver", "boundary_ids": ["boundary:exterior"], "parameters": {}}],
        "excitation_ports": [{"id": "port:driver", "component_id": "component:driver"}],
    }
    monkeypatch.setattr(
        deploy_solve_module,
        "stage_exact_coupled_system",
        lambda _package, _work_dir: {"compiled_system": compiled_system},
    )
    monkeypatch.setattr(
        deploy_solve_module,
        "_write_transformed_coupled_mesh",
        lambda _source, target, _base_mesh, _placement: target.write_text("mesh", encoding="utf-8"),
    )

    _path, request = prepare_deploy_coupled_request(
        {
            "packagePath": "speaker.blabsp",
            "backend": "cuda",
            "frequenciesHz": [20.0, 40.0],
            "sources": [{"id": "source", "delayMs": 12.5}],
            "observationPointsM": [[0.0, 1.2, 4.0], [1.0, 1.2, 5.0]],
        },
        tmp_path,
        cache=cache,
    )

    assert request["frequencies_hz"] == [20.0, 40.0]
    assert request["deploy"]["rows"] == 1
    assert request["deploy"]["columns"] == 2
    weights = request["outputs"][0]["options"]["excitation_weights_sweep"]
    assert complex(weights[0][0]["real"], weights[0][0]["imag"]) == pytest.approx(1j, abs=1e-7)
    assert complex(weights[1][0]["real"], weights[1][0]["imag"]) == pytest.approx(-1 + 0j, abs=1e-7)


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
    payload["backend"] = "cpu"
    payload["observation"].update({"centerXM": 3.0, "yawDeg": 90.0})

    request_path, _ = prepare_deploy_solve_request(payload, tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))

    assert request["observation_points_m"][0] == pytest.approx([-2.0, 1.2, 13.0])
    assert request["observation_points_m"][-1] == pytest.approx([8.0, 1.2, 1.0])


def test_prepare_deploy_solve_request_pitches_observation_plane(tmp_path: Path) -> None:
    payload = _payload()
    payload["backend"] = "cpu"
    payload["observation"].update({"heightM": 6.0, "pitchDeg": 90.0})

    request_path, _ = prepare_deploy_solve_request(payload, tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))

    assert request["observation_points_m"][0] == pytest.approx([-6.0, 11.0, 7.0])
    assert request["observation_points_m"][-1] == pytest.approx([6.0, 1.0, 7.0])


def test_prepare_deploy_solve_request_omits_points_below_ground(tmp_path: Path) -> None:
    payload = _payload()
    payload["backend"] = "cpu"
    payload["observation"].update({"heightM": 0.0, "pitchDeg": 30.0})

    request_path, _ = prepare_deploy_solve_request(payload, tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))

    assert len(request["observation_points_m"]) < 35
    assert len(request["observation_points_m"]) == len(request["observation_sample_indices"])
    assert all(point[1] >= -1e-6 for point in request["observation_points_m"])


def test_prepare_deploy_solve_request_rolls_observation_plane(tmp_path: Path) -> None:
    payload = _payload()
    payload["backend"] = "cpu"
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
