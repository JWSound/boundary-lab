from dataclasses import replace

import numpy as np
import pytest

from blab.deploy.assets import DeployPackageData, DeploySolveCache
from blab.deploy.packages import common_frequencies, scene_packages
from blab.deploy.solve import (
    prepare_deploy_microphone_sweep_request,
    prepare_deploy_rom_microphone_sweep_request,
    prepare_deploy_rom_request,
)


def package(path, rank, inputs, frequencies):
    # Different closed meshes, ranks, input counts and driver counts.
    if rank == 1:
        points = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        triangles = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    else:
        points = np.array([[0.1, 0, 0], [-0.1, 0, 0], [0, 0.1, 0], [0, -0.1, 0], [0, 0, 0.1], [0, 0, -0.1]])
        triangles = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    nf, nn, nt = len(frequencies), len(points), len(triangles)
    arrays = {"frequencies_hz": np.array(frequencies)}
    for name, shape in {
        "k": (rank, rank),
        "c": (rank, nn),
        "d": (nt, rank),
        "b": (rank, inputs),
        "e": (nt, inputs),
        "velocity": (inputs, rank),
        "current": (inputs, rank),
        "velocity_drive": (inputs, inputs),
        "current_drive": (inputs, inputs),
    }.items():
        arrays[name] = np.full((nf, 1, *shape), 0.01, dtype=np.complex64)
    arrays["k"][:] = np.eye(rank)
    model = {
        "representation": "parity_petrov_galerkin_rom",
        "rank_per_sector": rank,
        "symmetry_mode": "off",
        "image_count": 1,
        "sector_signs": [[1, 1]],
        "node_orbits": [[i] for i in range(nn)],
        "face_orbits": [[i] for i in range(nt)],
        "arrays": arrays,
    }
    return DeployPackageData(
        path,
        (str(path), 1, 1),
        {"fidelity_level": 3, "name": path.stem, "excitation_port_ids": [f"input:{i}" for i in range(inputs)]},
        np.array(frequencies),
        triangles,
        points,
        np.ones((nf, inputs, nn), dtype=np.complex64),
        np.ones((nf, inputs, nt), dtype=np.complex64),
        b"",
        model,
    )


@pytest.fixture
def mixed(tmp_path, monkeypatch):
    a = package(tmp_path / "a.blabsp", 1, 1, [20, 40, 80])
    b = package(tmp_path / "b.blabsp", 2, 2, [10, 40, 80])
    cache = DeploySolveCache()
    monkeypatch.setattr(cache, "load_package", lambda path: a if path == a.path else b)
    payload = {
        "packagePaths": {"a": str(a.path), "b": str(b.path)},
        "frequencyHz": 40,
        "sources": [
            {"id": "a1", "packageId": "a", "positionX": -2, "positionHeightM": 1},
            {"id": "b1", "packageId": "b", "positionX": 0, "positionHeightM": 1},
            {"id": "a2", "packageId": "a", "positionX": 2, "positionHeightM": 1, "delayMs": 2},
        ],
        "observationPointsM": [[0, 1, 4]],
        "backend": "cuda",
    }
    yield payload, cache, a, b
    cache.close()


def test_mixed_models_preserve_interleaved_mesh_and_output_order(mixed, tmp_path):
    payload, cache, _, _ = mixed
    _, request = prepare_deploy_rom_request(payload, tmp_path / "single", cache=cache)
    assert request["schema_version"] == 3
    assert [c["vertex_count"] for c in request["boundary_components"]] == [4, 6, 4]
    assert [c["vertex_offset"] for c in request["boundary_components"]] == [0, 4, 10]
    assert [c["face_offset"] for c in request["boundary_components"]] == [0, 4, 12]
    assert [i["id"] for i in request["rom"]["instances"]] == ["a1", "b1", "a2"]
    assert [i["model_id"] for i in request["rom"]["instances"]] == ["a", "b", "a"]
    assert [t["source_id"] for t in request["transducers"]] == ["a1", "b1", "b1", "a2"]
    assert len(request["rom"]["models"]["b"]["instances"][0]["input_real"]) == 2
    assert len(request["boundary_neumann"]["real"]) == 16
    assert not any(request["boundary_neumann"]["real"])


def test_mixed_sweeps_intersect_frequencies_and_reuse_both_models(mixed, tmp_path):
    payload, cache, _, _ = mixed
    _, first = prepare_deploy_rom_microphone_sweep_request(payload, tmp_path / "first", cache=cache)
    _, second = prepare_deploy_rom_microphone_sweep_request(payload, tmp_path / "second", cache=cache)
    assert first["frequencies_hz"] == [40, 80]
    assert second["provenance"]["rom_sweep_stage_binary_bytes_written"] == 0
    assert len(cache.rom_sweep_stages) == 2
    entry = first["rom_sweep"]["frequencies"][1]
    assert [i["id"] for i in entry["instances"]] == ["a1", "b1", "a2"]
    drive = entry["instances"][2]
    assert complex(drive["input_real"][0], drive["input_imag"][0]) == pytest.approx(
        2.83 * np.exp(-2j * np.pi * 80 * 0.002)
    )
    _, boundary = prepare_deploy_microphone_sweep_request(payload, tmp_path / "boundary", cache=cache)
    assert boundary["frequencies_hz"] == [40, 80]
    assert len(boundary["boundary_neumann_sweep"]["real"][0]) == 16


def test_mixed_package_validation(mixed):
    payload, cache, a, b = mixed
    assert common_frequencies({"a": a, "b": b}, coupled=True) == [40, 80]
    with pytest.raises(ValueError, match="missing package"):
        scene_packages({**payload, "packagePaths": {"a": str(a.path)}}, cache)
    changed = replace(b, manifest={**b.manifest, "medium": {"sound_speed_m_per_s": 100}})
    cache.load_package = lambda path: a if path == a.path else changed
    with pytest.raises(ValueError, match="same exterior"):
        scene_packages(payload, cache)


def test_cache_keeps_distinct_packages(monkeypatch, tmp_path):
    import blab.deploy.assets as assets

    a = package(tmp_path / "a.blabsp", 1, 1, [40])
    b = package(tmp_path / "b.blabsp", 2, 2, [40])
    a.path.touch()
    b.path.touch()
    monkeypatch.setattr(
        assets,
        "_load_deploy_package_data",
        lambda path, fingerprint: replace(a if path == a.path else b, fingerprint=fingerprint),
    )
    cache = DeploySolveCache()
    try:
        first = cache.load_package(a.path)
        cache.load_package(b.path)
        assert cache.load_package(a.path) is first
        assert len(cache.packages) == 2
    finally:
        cache.close()
