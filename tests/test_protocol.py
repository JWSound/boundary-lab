import numpy as np

from blab.config import ChannelConfig, CrossoverConfig, MeshConfig, RadiatorConfig, SimulationConfig
from blab.protocol import (
    build_mesh_assets,
    frequency_result_from_dict,
    frequency_result_to_dict,
    mesh_asset_references,
    simulation_config_from_dict,
    simulation_config_to_dict,
    solve_request_from_config_and_frequencies,
    solve_request_to_job_inputs,
    solve_request_to_config_and_frequencies,
)
from blab.solvers.base import FrequencyResult, FrequencySolveTimings


def test_simulation_config_round_trips_through_wire_dict() -> None:
    config = SimulationConfig(
        mesh_file="fallback.msh",
        freq_min=100.0,
        freq_max=10000.0,
        freq_count=5,
        distance=3.5,
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
                channel="tweeter",
                velocity_offset_db=-1.5,
                hpf=CrossoverConfig(
                    type="highpass",
                    filter="linkwitz_riley",
                    order=4,
                    frequency_hz=900.0,
                ),
            ),
        ),
        channels=(
            ChannelConfig(
                name="tweeter",
                level_db=-3.0,
                polarity=-1,
                delay_ms=0.25,
                lpf=CrossoverConfig(type="lowpass", filter="butterworth", order=2, frequency_hz=20000.0),
            ),
        ),
        spherical_sampling_enabled=True,
        spherical_sampling_points=32,
        symmetry="xy",
    )

    restored = simulation_config_from_dict(simulation_config_to_dict(config))

    assert restored.mesh_file == "fallback.msh"
    assert restored.distance == 3.5
    assert restored.meshes[0].translation_m == (0.1, 0.0, -0.2)
    assert restored.radiators[0].hpf.filter == "linkwitz_riley"
    assert restored.radiators[0].channel == "tweeter"
    assert restored.radiators[0].velocity_offset_db == -1.5
    assert restored.channels[0].polarity == -1
    assert restored.channels[0].lpf.frequency_hz == 20000.0
    assert restored.spherical_sampling_enabled is True
    assert restored.symmetry == "xy"


def test_frequency_result_round_trips_through_wire_dict() -> None:
    result = FrequencyResult(
        freq_hz=1000.0,
        horizontal_spl_norm_db=np.array([-6.0, 0.0, -3.0], dtype=np.float32),
        vertical_spl_norm_db=np.array([-8.0, 0.0, -4.0], dtype=np.float32),
        impedance=np.array([[1.0, 0.2], [2.0, 0.4]], dtype=np.float32),
        horizontal_spl_db=np.array([82.0, 88.0, 85.0], dtype=np.float32),
        vertical_spl_db=np.array([80.0, 88.0, 84.0], dtype=np.float32),
        sphere_spl_norm_db=np.array([-12.0, -3.0], dtype=np.float32),
        channel_names=np.array(["LF", "HF"]),
        horizontal_pressure=np.array(
            [[1.0 + 0.0j, 2.0 + 0.5j, 1.0 - 0.2j], [0.5 + 0.1j, 1.5 + 0.0j, 0.3 - 0.4j]],
            dtype=np.complex64,
        ),
        vertical_pressure=np.array(
            [[0.8 + 0.0j, 2.0 + 0.2j, 0.8 - 0.1j], [0.4 + 0.1j, 1.5 + 0.0j, 0.2 - 0.3j]],
            dtype=np.complex64,
        ),
        sphere_pressure=np.array([[0.4 + 0.0j, 0.8 + 0.1j], [0.2 + 0.1j, 0.6 + 0.0j]], dtype=np.complex64),
        timings=FrequencySolveTimings(assembly_s=1.25, solve_s=2.5, field_s=0.75),
    )

    restored = frequency_result_from_dict(frequency_result_to_dict(result))

    assert restored.freq_hz == 1000.0
    assert np.allclose(restored.horizontal_spl_norm_db, result.horizontal_spl_norm_db)
    assert np.allclose(restored.impedance, result.impedance)
    assert np.allclose(restored.sphere_spl_norm_db, result.sphere_spl_norm_db)
    assert restored.channel_names.tolist() == ["LF", "HF"]
    assert np.allclose(restored.horizontal_pressure, result.horizontal_pressure)
    assert np.allclose(restored.vertical_pressure, result.vertical_pressure)
    assert np.allclose(restored.sphere_pressure, result.sphere_pressure)
    assert restored.timings == result.timings


def test_solve_request_round_trips_config_and_frequencies() -> None:
    config = SimulationConfig(mesh_file="mesh.msh", symmetry="x")
    frequencies = np.array([200.0, 1000.0, 5000.0], dtype=np.float32)

    restored_config, restored_freqs = solve_request_to_config_and_frequencies(
        solve_request_from_config_and_frequencies(config, frequencies)
    )

    assert restored_config.mesh_file == "mesh.msh"
    assert restored_config.symmetry == "x"
    assert np.allclose(restored_freqs, frequencies)


def test_solve_request_can_embed_mesh_assets(tmp_path) -> None:
    fallback = tmp_path / "fallback.msh"
    mesh = tmp_path / "mesh.msh"
    fallback.write_text("fallback mesh", encoding="utf-8")
    mesh.write_text("real mesh", encoding="utf-8")
    config = SimulationConfig(
        mesh_file=str(fallback),
        meshes=(MeshConfig(name="mesh", file=str(mesh), scale_factor=0.001),),
    )

    references = mesh_asset_references(config)
    assets = build_mesh_assets(config)
    restored_config, restored_freqs, restored_assets = solve_request_to_job_inputs(
        solve_request_from_config_and_frequencies(
            config,
            np.array([1000.0], dtype=np.float32),
            include_assets=True,
        )
    )

    assert references == [str(fallback), str(mesh)]
    assert restored_config.meshes[0].file == str(mesh)
    assert restored_freqs.tolist() == [1000.0]
    assert [asset["original_path"] for asset in assets] == references
    assert [asset["original_path"] for asset in restored_assets] == references
    assert all(asset["content_base64"] for asset in restored_assets)
