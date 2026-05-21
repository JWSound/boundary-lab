import numpy as np

from blab.cloud.protocol import (
    config_from_payload,
    config_to_payload,
    frequency_result_from_payload,
    frequency_result_to_payload,
)
from blab.config import MeshConfig, RadiatorConfig, SimulationConfig
from blab.live import FrequencyResult


def test_simulation_config_round_trips_through_cloud_payload() -> None:
    config = SimulationConfig(
        mesh_file="case.msh",
        freq_min=200.0,
        freq_max=2000.0,
        freq_count=3,
        meshes=(MeshConfig(name="waveguide", file="waveguide.msh", translation_m=(0.0, 0.1, 0.0)),),
        radiators=(RadiatorConfig(name="HF", tag=2, level_db=-1.5, delay_ms=0.1),),
    )

    restored = config_from_payload(config_to_payload(config))

    assert restored.mesh_file == "case.msh"
    assert restored.meshes[0].translation_m == (0.0, 0.1, 0.0)
    assert restored.radiators[0].name == "HF"
    assert restored.radiators[0].level_db == -1.5


def test_frequency_result_round_trips_through_cloud_payload() -> None:
    result = FrequencyResult(
        freq_hz=1000.0,
        horizontal_spl_norm_db=np.array([-6.0, 0.0, -6.0], dtype=np.float32),
        vertical_spl_norm_db=np.array([-8.0, 0.0, -8.0], dtype=np.float32),
        impedance=np.array([[1.0, 0.2], [2.0, 0.4]], dtype=np.float32),
        sphere_spl_norm_db=np.array([-3.0, 0.0], dtype=np.float32),
    )

    restored = frequency_result_from_payload(frequency_result_to_payload(result))

    assert restored.freq_hz == 1000.0
    np.testing.assert_allclose(restored.horizontal_spl_norm_db, result.horizontal_spl_norm_db)
    np.testing.assert_allclose(restored.impedance, result.impedance)
    np.testing.assert_allclose(restored.sphere_spl_norm_db, result.sphere_spl_norm_db)
