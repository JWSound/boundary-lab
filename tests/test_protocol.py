import numpy as np

from blab.config import CrossoverConfig, MeshConfig, RadiatorConfig, SimulationConfig
from blab.live import FrequencyResult
from blab.protocol import (
    frequency_result_from_dict,
    frequency_result_to_dict,
    simulation_config_from_dict,
    simulation_config_to_dict,
    solve_request_from_config_and_frequencies,
    solve_request_to_config_and_frequencies,
)


def test_simulation_config_round_trips_through_wire_dict() -> None:
    config = SimulationConfig(
        mesh_file="fallback.msh",
        freq_min=100.0,
        freq_max=10000.0,
        freq_count=5,
        meshes=(
            MeshConfig(
                name="waveguide",
                file="waveguide.msh",
                scale_factor=0.001,
                translation_m=(0.1, 0.0, -0.2),
            ),
        ),
        radiators=(
            RadiatorConfig(
                name="HF",
                mesh="waveguide",
                tag=4,
                level_db=-3.0,
                polarity=-1,
                delay_ms=0.25,
                hpf=CrossoverConfig(
                    type="highpass",
                    filter="linkwitz_riley",
                    order=4,
                    frequency_hz=900.0,
                ),
            ),
        ),
        spherical_sampling_enabled=True,
        spherical_sampling_points=32,
    )

    restored = simulation_config_from_dict(simulation_config_to_dict(config))

    assert restored.mesh_file == "fallback.msh"
    assert restored.meshes[0].translation_m == (0.1, 0.0, -0.2)
    assert restored.radiators[0].hpf.filter == "linkwitz_riley"
    assert restored.radiators[0].polarity == -1
    assert restored.spherical_sampling_enabled is True


def test_frequency_result_round_trips_through_wire_dict() -> None:
    result = FrequencyResult(
        freq_hz=1000.0,
        horizontal_spl_norm_db=np.array([-6.0, 0.0, -3.0], dtype=np.float32),
        vertical_spl_norm_db=np.array([-8.0, 0.0, -4.0], dtype=np.float32),
        impedance=np.array([[1.0, 0.2], [2.0, 0.4]], dtype=np.float32),
        horizontal_spl_db=np.array([82.0, 88.0, 85.0], dtype=np.float32),
        vertical_spl_db=np.array([80.0, 88.0, 84.0], dtype=np.float32),
        sphere_spl_norm_db=np.array([-12.0, -3.0], dtype=np.float32),
    )

    restored = frequency_result_from_dict(frequency_result_to_dict(result))

    assert restored.freq_hz == 1000.0
    assert np.allclose(restored.horizontal_spl_norm_db, result.horizontal_spl_norm_db)
    assert np.allclose(restored.impedance, result.impedance)
    assert np.allclose(restored.sphere_spl_norm_db, result.sphere_spl_norm_db)


def test_solve_request_round_trips_config_and_frequencies() -> None:
    config = SimulationConfig(mesh_file="mesh.msh")
    frequencies = np.array([200.0, 1000.0, 5000.0], dtype=np.float32)

    restored_config, restored_freqs = solve_request_to_config_and_frequencies(
        solve_request_from_config_and_frequencies(config, frequencies)
    )

    assert restored_config.mesh_file == "mesh.msh"
    assert np.allclose(restored_freqs, frequencies)
