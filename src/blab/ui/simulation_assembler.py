"""Application service for constructing validated solver requests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blab.config import ChannelConfig, MeshConfig, RadiatorConfig, SimulationConfig
from blab.live import build_log_frequencies, order_frequencies_for_live_plotting
from blab.symmetry import validate_reduced_mesh_configs


@dataclass(frozen=True)
class SimulationParameters:
    freq_min_hz: float
    freq_max_hz: float
    freq_count: int
    observation_distance_m: float
    polar_angle_step_deg: float
    use_burton_miller: bool
    gmres_tolerance: float
    normalized_channel_correction: bool
    horizontal_normalization_angle_deg: float
    spherical_sampling_enabled: bool
    spherical_sampling_points: int
    symmetry: str


@dataclass(frozen=True)
class PreparedSimulation:
    config: SimulationConfig
    ordered_frequencies: np.ndarray


class SimulationAssembler:
    """Build a solver-domain request without reading widgets or managing Qt."""

    def prepare(
        self,
        *,
        mesh_configs: tuple[MeshConfig, ...],
        radiators: tuple[RadiatorConfig, ...],
        channels: tuple[ChannelConfig, ...],
        parameters: SimulationParameters,
    ) -> PreparedSimulation:
        if not mesh_configs:
            raise ValueError("Enable at least one generated or imported mesh before solving.")
        if not radiators:
            raise ValueError("Mark at least one surface as driven before solving.")

        validate_reduced_mesh_configs(mesh_configs, parameters.symmetry)
        freq_min = float(min(parameters.freq_min_hz, parameters.freq_max_hz))
        freq_max = float(max(parameters.freq_min_hz, parameters.freq_max_hz))
        frequencies = build_log_frequencies(freq_min, freq_max, int(parameters.freq_count))
        ordered_frequencies = order_frequencies_for_live_plotting(frequencies)
        config = SimulationConfig(
            mesh_file=mesh_configs[0].file,
            freq_min=freq_min,
            freq_max=freq_max,
            freq_count=int(parameters.freq_count),
            tag_throat=radiators[0].tag,
            meshes=mesh_configs,
            radiators=radiators,
            channels=channels,
            distance=float(parameters.observation_distance_m),
            step_size=float(parameters.polar_angle_step_deg),
            use_burton_miller=bool(parameters.use_burton_miller),
            gmres_tolerance=float(parameters.gmres_tolerance),
            workers=1,
            flat_target_normalization_enabled=bool(parameters.normalized_channel_correction),
            flat_target_reference_angle_deg=float(parameters.horizontal_normalization_angle_deg),
            spherical_sampling_enabled=bool(parameters.spherical_sampling_enabled),
            spherical_sampling_points=int(parameters.spherical_sampling_points),
            symmetry=parameters.symmetry,
        )
        return PreparedSimulation(config=config, ordered_frequencies=ordered_frequencies)
