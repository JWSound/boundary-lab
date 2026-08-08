from pathlib import Path

import meshio
import numpy as np

from blab.ath import AthRunResult
from blab.generators.ath import (
    ATH_PROVIDER_ID,
    AthGeneratorBackend,
    ath_result_to_generated_geometry,
    ath_source,
    generated_geometry_to_ath_result,
)
from blab.generators.base import GeneratedGeometryReference, GeneratorDocument
from blab.generators.registry import available_generator_infos, create_generator, generator_info


def _write_triangle_mesh(path: Path) -> None:
    mesh = meshio.Mesh(
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        cells=[("triangle", np.array([[0, 1, 2]], dtype=np.int64))],
        cell_data={"gmsh:physical": [np.array([2], dtype=np.int32)]},
        field_data={"SD1D1001": np.array([2, 2], dtype=np.int32)},
    )
    meshio.write(path, mesh, file_format="gmsh22", binary=False)


def test_generator_registry_exposes_ath_provider_contract() -> None:
    info = generator_info("ath4")

    assert info.provider_id == ATH_PROVIDER_ID
    assert info.capabilities.source_formats == ("ath_cfg",)
    assert {item.provider_id for item in available_generator_infos()} == {"ath"}
    assert isinstance(create_generator("ath", ath_exe="ath.exe"), AthGeneratorBackend)


def test_ath_result_conversion_preserves_solver_artifacts(tmp_path: Path) -> None:
    mesh_path = tmp_path / "waveguide.msh"
    _write_triangle_mesh(mesh_path)
    ath_result = AthRunResult(
        output_dir=tmp_path,
        msh_path=mesh_path,
        config_path=tmp_path / "waveguide.cfg",
        driven_tag=2,
        radiators=(),
        cleaned_msh_path=mesh_path,
    )

    generated = ath_result_to_generated_geometry(ath_result)
    restored = generated_geometry_to_ath_result(generated)

    assert generated.provider_id == "ath"
    assert generated.solver_mesh_path == mesh_path
    assert generated.provider_metadata["driven_tag"] == 2
    assert restored == ath_result


def test_ath_backend_restores_generic_geometry_from_document_artifact(tmp_path: Path) -> None:
    mesh_path = tmp_path / "waveguide.msh"
    _write_triangle_mesh(mesh_path)
    document = GeneratorDocument(
        id="design1",
        name="waveguide",
        provider_id="ath",
        provider_schema_version=1,
        source=ath_source("Length = 100"),
        artifact=GeneratedGeometryReference(
            output_dir=str(tmp_path),
            mesh_path=str(mesh_path),
            source_path=str(tmp_path / "waveguide.cfg"),
        ),
    )

    restored = create_generator("ath").restore(document)

    assert restored is not None
    assert restored.provider_id == "ath"
    assert restored.mesh_path == mesh_path
    assert restored.provider_metadata["driven_tag"] == 2
