from pathlib import Path

import numpy as np

from blab.cloud.bundle import load_solve_bundle, write_solve_bundle
from blab.config import MeshConfig, RadiatorConfig, SimulationConfig


def test_solve_bundle_round_trips_config_frequencies_and_meshes(tmp_path: Path) -> None:
    mesh_a = tmp_path / "waveguide.msh"
    mesh_b = tmp_path / "woofer.msh"
    mesh_a.write_text("mesh a", encoding="utf-8")
    mesh_b.write_text("mesh b", encoding="utf-8")
    config = SimulationConfig(
        mesh_file=str(mesh_a),
        freq_min=200.0,
        freq_max=2000.0,
        freq_count=3,
        meshes=(
            MeshConfig(name="waveguide", file=str(mesh_a), scale_factor=0.001),
            MeshConfig(name="woofer", file=str(mesh_b), translation_m=(0.0, 0.0, 0.1)),
        ),
        radiators=(RadiatorConfig(name="HF", tag=2),),
    )
    frequencies = np.array([200.0, 2000.0, 632.5], dtype=np.float32)

    bundle_path = write_solve_bundle(tmp_path / "job.blabsolve.zip", config=config, frequencies=frequencies)
    restored_config, restored_frequencies = load_solve_bundle(bundle_path, tmp_path / "workspace")

    np.testing.assert_allclose(restored_frequencies, frequencies)
    assert Path(restored_config.mesh_file).read_text(encoding="utf-8") == "mesh a"
    assert Path(restored_config.meshes[0].file).read_text(encoding="utf-8") == "mesh a"
    assert Path(restored_config.meshes[1].file).read_text(encoding="utf-8") == "mesh b"
    assert restored_config.meshes[1].translation_m == (0.0, 0.0, 0.1)
    assert restored_config.radiators[0].name == "HF"
