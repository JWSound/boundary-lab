from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import blab.speaker_package as speaker_package_module
from blab.physical_model import (
    AcousticRegionKind,
    CompiledPhysicalSystem,
    CompiledRegion,
    ComponentKind,
    ExcitationPort,
    ExcitationPortKind,
    PhysicalComponent,
    PhysicalSolveKind,
)
from blab.solve_results import (
    BEM_BOUNDARY_DOMAIN_ID,
    BEM_BOUNDARY_NEUMANN_ID,
    BEM_BOUNDARY_PRESSURE_ID,
    SPHERE_DOMAIN_ID,
    SPHERE_PRESSURE_ID,
    ResultDomain,
    SolvedQuantity,
    SolvedSystem,
    SolveProvenance,
)
from blab.speaker_package import (
    SOURCE_TO_PACKAGE_ROTATION,
    SPEAKER_MACRO_B_ID,
    SPEAKER_MACRO_C_ID,
    SPEAKER_MACRO_D_ID,
    SPEAKER_MACRO_E_ID,
    SPEAKER_MACRO_K_ID,
    SpeakerPackageConfig,
    SpeakerPackageFidelity,
    expand_bem_boundary_symmetry,
    export_speaker_package,
    prepare_speaker_package_solve,
    rotate_source_points,
    speaker_package_issues,
    validate_speaker_package,
)
from blab.system_contract import OutputRequest, SystemSolveRequest
from blab.system_solve import SystemUiSolveRequest


def _solved_system(*, include_bem: bool = True, symmetry: str = "off") -> SolvedSystem:
    frequencies = np.asarray([100.0, 1000.0])
    excitation_ids = ("port:a", "port:b")
    sphere_points = np.asarray([[0.0, 0.0, 2.0], [0.0, 2.0, 0.0], [0.0, -2.0, 0.0], [2.0, 0.0, 0.0]])
    sphere_pressure = (np.arange(16).reshape(2, 2, 4) + 1j * np.arange(101, 117).reshape(2, 2, 4)).astype(np.complex64)
    domains = {
        SPHERE_DOMAIN_ID: ResultDomain(
            id=SPHERE_DOMAIN_ID,
            kind="spherical_observation",
            dimensions=("observation",),
            coordinates={"points_m": sphere_points},
        )
    }
    quantities = {
        SPHERE_PRESSURE_ID: SolvedQuantity(
            id=SPHERE_PRESSURE_ID,
            quantity="exterior_pressure",
            unit="Pa",
            dimensions=("frequency", "excitation", "observation"),
            values=sphere_pressure,
            domain_id=SPHERE_DOMAIN_ID,
            available_frequency_mask=np.ones(2, dtype=bool),
        )
    }
    if include_bem:
        bem_points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        triangles = np.asarray([[0, 1, 2], [0, 3, 1]], dtype=np.int64)
        domains[BEM_BOUNDARY_DOMAIN_ID] = ResultDomain(
            id=BEM_BOUNDARY_DOMAIN_ID,
            kind="bem_boundary",
            dimensions=("bem_node", "bem_face"),
            coordinates={"points_m": bem_points, "mesh_index": np.zeros(4, dtype=np.int32)},
            topology={
                "triangles": triangles,
                "mesh_index": np.zeros(2, dtype=np.int32),
                "source_physical_tag": np.asarray([7, 9], dtype=np.int32),
            },
            metadata={"pressure_space": "P1", "normal_derivative_space": "DP0", "symmetry": symmetry},
        )
        quantities[BEM_BOUNDARY_PRESSURE_ID] = SolvedQuantity(
            id=BEM_BOUNDARY_PRESSURE_ID,
            quantity="bem_boundary_pressure",
            unit="Pa",
            dimensions=("frequency", "excitation", "bem_node"),
            values=(np.arange(16).reshape(2, 2, 4) + 1j * np.arange(201, 217).reshape(2, 2, 4)).astype(np.complex64),
            domain_id=BEM_BOUNDARY_DOMAIN_ID,
            available_frequency_mask=np.ones(2, dtype=bool),
        )
        quantities[BEM_BOUNDARY_NEUMANN_ID] = SolvedQuantity(
            id=BEM_BOUNDARY_NEUMANN_ID,
            quantity="bem_boundary_neumann",
            unit="Pa/m",
            dimensions=("frequency", "excitation", "bem_face"),
            values=(np.arange(8).reshape(2, 2, 2) + 1j * np.arange(301, 309).reshape(2, 2, 2)).astype(np.complex64),
            domain_id=BEM_BOUNDARY_DOMAIN_ID,
            available_frequency_mask=np.ones(2, dtype=bool),
        )
    exterior = CompiledRegion(
        id="region:exterior",
        name="Exterior",
        kind=AcousticRegionKind.UNBOUNDED_AIR,
        mesh_ids=(),
        volume_groups=(),
        sound_speed_m_per_s=343.0,
        density_kg_per_m3=1.21,
        loss_model={},
    )
    compiled = CompiledPhysicalSystem(
        id="system:test",
        name="Test speaker",
        meshes=(),
        regions=(exterior,),
        boundaries=(),
        interfaces=(),
        components=(),
        excitation_ports=(),
        assumptions=(),
    )
    return SolvedSystem(
        run_id="test-run",
        provenance=SolveProvenance(
            backend_id="beat_cpu",
            solve_kind="exterior_bem",
            solver_options={"symmetry": symmetry},
        ),
        frequencies_hz=frequencies,
        excitation_ids=excitation_ids,
        domains=domains,
        quantities=quantities,
        completion_mask=np.ones(2, dtype=bool),
        diagnostics_by_frequency=({}, {}),
        status="completed",
        compiled_system=compiled,
    )


def _coupled_macro_solved_system() -> SolvedSystem:
    solved = _solved_system()
    frequency_count = solved.frequencies_hz.size
    excitation_count = len(solved.excitation_ids)
    state_count = 3
    bem_node_count = solved.domains[BEM_BOUNDARY_DOMAIN_ID].coordinates["points_m"].shape[0]
    bem_face_count = solved.domains[BEM_BOUNDARY_DOMAIN_ID].topology["triangles"].shape[0]
    quantities = dict(solved.quantities)
    definitions = (
        (
            SPEAKER_MACRO_K_ID,
            "speaker_macro_k",
            (frequency_count, state_count, state_count),
            ("frequency", "state_row", "state_column"),
        ),
        (
            SPEAKER_MACRO_C_ID,
            "speaker_macro_c",
            (frequency_count, state_count, bem_node_count),
            ("frequency", "state_row", "bem_node"),
        ),
        (
            SPEAKER_MACRO_D_ID,
            "speaker_macro_d",
            (frequency_count, bem_face_count, state_count),
            ("frequency", "bem_face", "state_column"),
        ),
        (
            SPEAKER_MACRO_B_ID,
            "speaker_macro_b",
            (frequency_count, state_count, excitation_count),
            ("frequency", "state_row", "excitation"),
        ),
        (
            SPEAKER_MACRO_E_ID,
            "speaker_macro_e",
            (frequency_count, bem_face_count, excitation_count),
            ("frequency", "bem_face", "excitation"),
        ),
    )
    for index, (identifier, quantity, shape, dimensions) in enumerate(definitions, start=1):
        values = np.full(shape, complex(index, -index), dtype=np.complex64)
        quantities[identifier] = SolvedQuantity(
            id=identifier,
            quantity=quantity,
            unit="mixed",
            dimensions=dimensions,
            values=values,
            metadata={
                "format_version": 1,
                "state_count": state_count,
                "state_blocks": [{"name": "test", "offset": 0, "count": state_count}],
            },
            available_frequency_mask=np.ones(frequency_count, dtype=bool),
        )
    return replace(
        solved,
        quantities=quantities,
        provenance=replace(solved.provenance, solve_kind="coupled_bem_fem"),
    )


def _read_npz(archive: zipfile.ZipFile, member: str) -> dict[str, np.ndarray]:
    with np.load(archive.open(member)) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def test_level_one_archive_is_versioned_and_preserves_complex_pattern(tmp_path: Path) -> None:
    solved = _solved_system(include_bem=False)
    output = tmp_path / "speaker.blabsp"

    result = export_speaker_package(
        solved,
        SpeakerPackageConfig(output, "Test speaker", SpeakerPackageFidelity.PATTERN),
    )

    assert result.path == output.resolve()
    manifest = validate_speaker_package(output)
    assert manifest["schema"] == "boundary-lab-speaker-package"
    assert manifest["schema_version"] == 1
    assert manifest["fidelity_level"] == 1
    assert manifest["capabilities"] == ["complex_spherical_pattern"]
    assert manifest["excitation_port_ids"] == ["port:a", "port:b"]
    assert manifest["phasor_convention"] == "exp(-i omega t)"
    assert manifest["coordinate_system"]["forward_axis"] == "+Y"
    assert manifest["coordinate_system"]["source_forward_axis"] == "+Z"
    with zipfile.ZipFile(output) as archive:
        pattern = _read_npz(archive, "data/patterns.npz")
    original = solved.quantities[SPHERE_PRESSURE_ID].values
    np.testing.assert_array_equal(pattern["pressure_pa"], original)
    assert np.iscomplexobj(pattern["pressure_pa"])


def test_level_two_contains_progressive_pattern_and_fixed_source_data(tmp_path: Path) -> None:
    solved = _solved_system()
    output = tmp_path / "speaker.blabsp"

    export_speaker_package(
        solved,
        SpeakerPackageConfig(output, "Test speaker", SpeakerPackageFidelity.FIXED_SOURCES),
    )

    manifest = validate_speaker_package(output)
    assert manifest["capabilities"] == ["complex_spherical_pattern", "fixed_distributed_sources"]
    assert manifest["files"]["fixed_sources"]["pressure_space"] == "P1"
    assert manifest["files"]["fixed_sources"]["normal_derivative_space"] == "DP0"
    with zipfile.ZipFile(output) as archive:
        assert "geometry/exterior.msh" in archive.namelist()
        fixed = _read_npz(archive, "data/fixed-sources.npz")
    np.testing.assert_array_equal(fixed["triangles"], solved.domains[BEM_BOUNDARY_DOMAIN_ID].topology["triangles"])
    np.testing.assert_array_equal(fixed["source_physical_tag"], np.asarray([7, 9], dtype=np.int32))
    np.testing.assert_array_equal(fixed["pressure_pa"], solved.quantities[BEM_BOUNDARY_PRESSURE_ID].values)
    np.testing.assert_array_equal(
        fixed["normal_derivative_pa_per_m"], solved.quantities[BEM_BOUNDARY_NEUMANN_ID].values
    )


def test_level_three_contains_dynamic_macro_model(tmp_path: Path) -> None:
    solved = _coupled_macro_solved_system()
    output = tmp_path / "speaker-coupled.blabsp"

    export_speaker_package(
        solved,
        SpeakerPackageConfig(output, "Coupled speaker", SpeakerPackageFidelity.COUPLED),
    )

    manifest = validate_speaker_package(output)
    assert manifest["fidelity_level"] == 3
    assert manifest["fidelity"] == "coupled"
    assert manifest["capabilities"] == [
        "complex_spherical_pattern",
        "fixed_distributed_sources",
        "dynamic_boundary_macro_model",
    ]
    assert manifest["files"]["coupled_model"]["equations"] == [
        "K z + C p = B u",
        "q = D z + E u",
    ]
    with zipfile.ZipFile(output) as archive:
        model = _read_npz(archive, "data/coupled-model.npz")
    np.testing.assert_array_equal(model["matrix_k"], solved.quantities[SPEAKER_MACRO_K_ID].values)
    np.testing.assert_array_equal(model["matrix_e"], solved.quantities[SPEAKER_MACRO_E_ID].values)


def test_export_rotation_maps_plus_z_to_plus_y_without_reflection(tmp_path: Path) -> None:
    solved = _solved_system()
    output = tmp_path / "speaker.blabsp"
    export_speaker_package(
        solved,
        SpeakerPackageConfig(output, "Test speaker", SpeakerPackageFidelity.FIXED_SOURCES),
    )

    assert np.linalg.det(SOURCE_TO_PACKAGE_ROTATION) == pytest.approx(1.0)
    np.testing.assert_array_equal(rotate_source_points(np.asarray([[0.0, 0.0, 2.0]])), [[0.0, 2.0, 0.0]])
    with zipfile.ZipFile(output) as archive:
        pattern = _read_npz(archive, "data/patterns.npz")
        fixed = _read_npz(archive, "data/fixed-sources.npz")
    source_sphere = solved.domains[SPHERE_DOMAIN_ID].coordinates["points_m"]
    source_bem = solved.domains[BEM_BOUNDARY_DOMAIN_ID].coordinates["points_m"]
    np.testing.assert_allclose(pattern["points_m"], source_sphere @ SOURCE_TO_PACKAGE_ROTATION.T)
    np.testing.assert_allclose(fixed["points_m"], source_bem @ SOURCE_TO_PACKAGE_ROTATION.T)
    source_triangle = source_bem[fixed["triangles"][0]]
    target_triangle = fixed["points_m"][fixed["triangles"][0]]
    source_normal = np.cross(source_triangle[1] - source_triangle[0], source_triangle[2] - source_triangle[0])
    target_normal = np.cross(target_triangle[1] - target_triangle[0], target_triangle[2] - target_triangle[0])
    np.testing.assert_allclose(source_normal, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(target_normal, [0.0, 1.0, 0.0])


def test_readiness_is_progressive_and_level_two_accepts_reduced_symmetry() -> None:
    level_one = _solved_system(include_bem=False)
    assert speaker_package_issues(level_one, SpeakerPackageFidelity.PATTERN) == ()
    assert {issue.code for issue in speaker_package_issues(level_one, SpeakerPackageFidelity.FIXED_SOURCES)} == {
        "missing_bem_domain",
        "missing_bem_pressure",
        "missing_bem_neumann",
    }
    reduced = _solved_system(symmetry="x")
    assert speaker_package_issues(reduced, SpeakerPackageFidelity.FIXED_SOURCES) == ()


@pytest.mark.parametrize(
    ("symmetry", "expected_nodes", "expected_faces", "expected_images"),
    (("x", 5, 4, 2), ("xy", 6, 6, 4)),
)
def test_bem_symmetry_expansion_welds_seams_and_preserves_trace_parity(
    symmetry: str,
    expected_nodes: int,
    expected_faces: int,
    expected_images: int,
) -> None:
    solved = _solved_system(symmetry=symmetry)
    domain = solved.domains[BEM_BOUNDARY_DOMAIN_ID]
    source_points = np.asarray(domain.coordinates["points_m"])
    source_faces = np.asarray(domain.topology["triangles"])
    source_pressure = np.asarray(solved.quantities[BEM_BOUNDARY_PRESSURE_ID].values)
    source_neumann = np.asarray(solved.quantities[BEM_BOUNDARY_NEUMANN_ID].values)

    expanded = expand_bem_boundary_symmetry(
        points_m=source_points,
        triangles=source_faces,
        pressure_pa=source_pressure,
        normal_derivative_pa_per_m=source_neumann,
        symmetry=symmetry,
    )

    assert expanded.points_m.shape == (expected_nodes, 3)
    assert expanded.triangles.shape == (expected_faces, 3)
    assert len(expanded.symmetry_images) == expected_images
    np.testing.assert_array_equal(expanded.pressure_pa, source_pressure[..., expanded.source_node_index])
    np.testing.assert_array_equal(
        expanded.normal_derivative_pa_per_m,
        source_neumann[..., expanded.source_face_index],
    )
    assert (
        len(
            {
                (int(source_index), *sorted(int(node) for node in face))
                for source_index, face in zip(expanded.source_face_index, expanded.triangles, strict=True)
            }
        )
        == expected_faces
    )

    for face, source_index, image_index in zip(
        expanded.triangles,
        expanded.source_face_index,
        expanded.face_symmetry_image_index,
        strict=True,
    ):
        source_triangle = source_points[source_faces[source_index]]
        target_triangle = expanded.points_m[face]
        source_normal = np.cross(source_triangle[1] - source_triangle[0], source_triangle[2] - source_triangle[0])
        target_normal = np.cross(target_triangle[1] - target_triangle[0], target_triangle[2] - target_triangle[0])
        transform = np.asarray(expanded.symmetry_images[image_index]["source_frame_transform"])
        np.testing.assert_allclose(target_normal, transform @ source_normal)


def test_bem_symmetry_expansion_snaps_scale_relative_plane_noise() -> None:
    expanded = expand_bem_boundary_symmetry(
        points_m=np.asarray(
            [
                [-1.2e-8, 0.0, 0.8],
                [0.5, 0.0, 0.8],
                [0.0, 0.5, 0.8],
            ]
        ),
        triangles=np.asarray([[0, 1, 2]]),
        pressure_pa=np.ones(3),
        normal_derivative_pa_per_m=np.ones(1),
        symmetry="x",
    )

    source_zero_indices = np.flatnonzero(expanded.source_node_index == 0)
    assert source_zero_indices.tolist() == [0]
    assert expanded.points_m[source_zero_indices[0], 0] == 0.0


def test_level_two_xy_symmetry_is_materialized_before_package_rotation(tmp_path: Path) -> None:
    solved = _solved_system(symmetry="xy")
    output = tmp_path / "speaker.blabsp"

    export_speaker_package(
        solved,
        SpeakerPackageConfig(output, "Symmetric speaker", SpeakerPackageFidelity.FIXED_SOURCES),
    )

    manifest = validate_speaker_package(output)
    metadata = manifest["files"]["fixed_sources"]
    assert metadata["geometry_expansion"] == "full"
    assert metadata["source_symmetry"] == "xy"
    assert metadata["source_node_count"] == 4
    assert metadata["exported_node_count"] == 6
    assert metadata["source_face_count"] == 2
    assert metadata["exported_face_count"] == 6
    assert [image["name"] for image in metadata["symmetry_images"]] == [
        "identity",
        "reflect_x",
        "reflect_y",
        "reflect_xy",
    ]
    with zipfile.ZipFile(output) as archive:
        fixed = _read_npz(archive, "data/fixed-sources.npz")
    source_pressure = solved.quantities[BEM_BOUNDARY_PRESSURE_ID].values
    source_neumann = solved.quantities[BEM_BOUNDARY_NEUMANN_ID].values
    np.testing.assert_array_equal(fixed["pressure_pa"], source_pressure[..., fixed["source_bem_node_index"]])
    np.testing.assert_array_equal(
        fixed["normal_derivative_pa_per_m"], source_neumann[..., fixed["source_bem_face_index"]]
    )
    # Source-frame X remains package X; source-frame Y is package -Z.
    assert set(np.sign(fixed["points_m"][:, 0])) == {-1.0, 0.0, 1.0}
    assert set(np.sign(fixed["points_m"][:, 2])) == {-1.0, 0.0, 1.0}


def test_archive_paths_are_portable_and_atomic_failure_preserves_destination(tmp_path: Path, monkeypatch) -> None:
    solved = _solved_system(include_bem=False)
    output = tmp_path / "speaker.blabsp"
    output.write_bytes(b"old package")
    before = set(tmp_path.iterdir())

    def fail(_path, _solved, _config):
        raise RuntimeError("archive failed")

    monkeypatch.setattr(speaker_package_module, "_write_archive", fail)
    with pytest.raises(RuntimeError, match="archive failed"):
        export_speaker_package(
            solved,
            SpeakerPackageConfig(output, "Test speaker", SpeakerPackageFidelity.PATTERN),
        )
    assert output.read_bytes() == b"old package"
    assert set(tmp_path.iterdir()) == before


def test_package_validator_reports_missing_payload(tmp_path: Path) -> None:
    solved = _solved_system(include_bem=False)
    good = tmp_path / "good.blabsp"
    broken = tmp_path / "broken.blabsp"
    export_speaker_package(solved, SpeakerPackageConfig(good, "Test speaker"))
    with zipfile.ZipFile(good) as source, zipfile.ZipFile(broken, "w") as target:
        for name in source.namelist():
            if name != "data/patterns.npz":
                target.writestr(name, source.read(name))

    with pytest.raises(ValueError, match="missing referenced member"):
        validate_speaker_package(broken)


def test_manifest_is_json_and_contains_no_absolute_paths(tmp_path: Path) -> None:
    output = tmp_path / "speaker.blabsp"
    export_speaker_package(_solved_system(), SpeakerPackageConfig(output, "Test speaker", 2))
    with zipfile.ZipFile(output) as archive:
        manifest_text = archive.read("manifest.json").decode("utf-8")
        json.loads(manifest_text)
        assert str(tmp_path.resolve()) not in manifest_text
        for name in archive.namelist():
            assert not Path(name).is_absolute()
            assert ".." not in Path(name).parts
            assert "\\" not in name


def test_export_solve_preparation_forces_sphere_and_level_two_traces() -> None:
    solved = _solved_system()
    component = PhysicalComponent("component:source", "Source", ComponentKind.IDEAL_VELOCITY_SOURCE, ())
    port = ExcitationPort("port:source", "Source", component.id, ExcitationPortKind.NORMAL_VELOCITY)
    assert solved.compiled_system is not None
    compiled = replace(solved.compiled_system, components=(component,), excitation_ports=(port,))
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(100.0,),
        excitation_port_ids=(port.id,),
        outputs=(
            OutputRequest(
                id="ui:exterior-pressure",
                quantity="exterior_pressure",
                options={"points_m": [[0.0, 0.0, 1.0]], "observation_domains": []},
            ),
        ),
        solver_options={"symmetry": "x"},
    )
    prepared = SystemUiSolveRequest(
        request=request,
        backend_id="beat_cpu",
        solve_kind=PhysicalSolveKind.EXTERIOR_BEM,
        polar_angle_deg=np.empty(0),
        excitation_channel_names=np.asarray(["main"]),
        excitation_component_names=np.asarray(["Source"]),
        horizontal_count=0,
        vertical_count=0,
        result_domains=(solved.domains[BEM_BOUNDARY_DOMAIN_ID],),
    )

    updated = prepare_speaker_package_solve(
        prepared,
        fidelity="fixed",
        sphere_point_count=32,
        sphere_radius_m=2.0,
    )

    output_ids = {output.id for output in updated.request.outputs}
    domain_ids = {domain.id for domain in updated.result_domains}
    assert SPHERE_DOMAIN_ID in domain_ids
    assert BEM_BOUNDARY_DOMAIN_ID in domain_ids
    assert BEM_BOUNDARY_PRESSURE_ID in output_ids
    assert BEM_BOUNDARY_NEUMANN_ID in output_ids
    assert updated.request.solver_options["symmetry"] == "x"
    compact = next(output for output in updated.request.outputs if output.id == "ui:exterior-pressure")
    sphere_block = next(
        item for item in compact.options["observation_domains"] if item["quantity_id"] == SPHERE_PRESSURE_ID
    )
    assert sphere_block["count"] == 32
    assert len(compact.options["points_m"]) == 33
