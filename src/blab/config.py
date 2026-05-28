"""Domain configuration used by the solver and protocol layers.

GUI-only workflow state, such as saved project editor text and imported mesh
choices, belongs in ``blab.ui.project_io`` until it becomes solver input.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from blab.defaults import SOLVER_OUTPUT_NPZ


@dataclass
class CrossoverConfig:
    type: str = "none"
    filter: str = "butterworth"
    order: int = 1
    frequency_hz: float | None = None


@dataclass
class RadiatorConfig:
    name: str
    tag: int
    mesh: str | None = None
    channel: str = "main"
    velocity_offset_db: float = 0.0
    level_db: float = 0.0
    polarity: int = 1
    delay_ms: float = 0.0
    hpf: CrossoverConfig = field(default_factory=CrossoverConfig)
    lpf: CrossoverConfig = field(default_factory=CrossoverConfig)


@dataclass
class ChannelConfig:
    name: str
    level_db: float = 0.0
    polarity: int = 1
    delay_ms: float = 0.0
    hpf: CrossoverConfig = field(default_factory=CrossoverConfig)
    lpf: CrossoverConfig = field(default_factory=CrossoverConfig)


@dataclass
class MeshConfig:
    name: str
    file: str
    scale_factor: float | None = None
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class SimulationConfig:
    mesh_file: str
    sound_speed: float = 343.0 # m/s
    rho: float = 1.21 # kg/m^3
    distance: float = 2.0 # meters
    axial_offset: float = 0 # meters
    step_size: float = 5
    min_angle: float = -180
    max_angle: float = 180
    freq_min: float = 400.0
    freq_max: float = 20000.0
    freq_count: int = 48
    tag_throat: int = 2
    meshes: tuple[MeshConfig, ...] = ()
    radiators: tuple[RadiatorConfig, ...] = ()
    channels: tuple[ChannelConfig, ...] = ()
    scale_factor: float = 0.001
    use_burton_miller: bool = True
    flat_target_normalization_enabled: bool = True
    gmres_tolerance: float = 1e-3
    workers: int = 3
    spherical_sampling_enabled: bool = False
    spherical_sampling_points: int = 6000
    output_npz: str = str(SOLVER_OUTPUT_NPZ)


def load_external_config(
    config_path: Path | None,
) -> tuple[tuple[MeshConfig, ...], tuple[RadiatorConfig, ...]]:
    if config_path is None:
        return (), ()

    raw_config = read_external_config(config_path)

    meshes = []
    seen_mesh_names = set()
    meshes_raw = raw_config.get("meshes", [])
    if meshes_raw is not None:
        if not isinstance(meshes_raw, list):
            raise ValueError(f"{config_path} 'meshes' must be a list.")
        for index, item in enumerate(meshes_raw, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Mesh entry {index} must be an object.")
            name = str(item.get("name", f"mesh_{index}"))
            if name in seen_mesh_names:
                raise ValueError(f"Duplicate mesh name in config: {name}")
            seen_mesh_names.add(name)
            if "file" not in item:
                raise ValueError(f"Mesh '{name}' must provide a file path.")

            mesh_path = Path(str(item["file"]))
            if not mesh_path.is_absolute():
                mesh_path = config_path.parent / mesh_path

            meshes.append(
                MeshConfig(
                    name=name,
                    file=str(mesh_path),
                    scale_factor=(
                        None if item.get("scale_factor") is None else float(item["scale_factor"])
                    ),
                    translation_m=parse_translation_m(name, item.get("translation_m")),
                )
            )

    radiators_raw = raw_config.get("radiators", [])
    if not isinstance(radiators_raw, list) or len(radiators_raw) == 0:
        raise ValueError(f"{config_path} must contain a non-empty 'radiators' list.")

    radiators = []
    seen_names = set()
    for index, item in enumerate(radiators_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Radiator entry {index} must be an object.")

        name = str(item.get("name", f"radiator_{index}"))
        if name in seen_names:
            raise ValueError(f"Duplicate radiator name in config: {name}")
        seen_names.add(name)

        hpf = _parse_crossover_config(item.get("hpf", {}) or {}, crossover_type="highpass")
        lpf = _parse_crossover_config(item.get("lpf", {}) or {}, crossover_type="lowpass")
        radiators.append(
            RadiatorConfig(
                name=name,
                tag=int(item["tag"]),
                mesh=None if item.get("mesh") is None else str(item.get("mesh")),
                channel=str(item.get("channel", "main")),
                velocity_offset_db=float(item.get("velocity_offset_db", 0.0)),
                level_db=float(item.get("level_db", 0.0)),
                polarity=int(item.get("polarity", 1)),
                delay_ms=float(item.get("delay_ms", 0.0)),
                hpf=hpf,
                lpf=lpf,
            )
        )

    return tuple(meshes), tuple(radiators)


def _parse_crossover_config(raw: dict, *, crossover_type: str | None = None) -> CrossoverConfig:
    if not isinstance(raw, dict):
        raise ValueError("Crossover config must be an object.")

    return CrossoverConfig(
        type=(
            str(raw.get("type", "none")).lower()
            if crossover_type is None or raw.get("frequency_hz") is None
            else crossover_type
        ),
        filter=str(raw.get("filter", "butterworth")).lower(),
        order=int(raw.get("order", 1)),
        frequency_hz=(
            None
            if raw.get("frequency_hz") is None
            else float(raw.get("frequency_hz"))
        ),
    )


def parse_translation_m(mesh_name: str, value) -> tuple[float, float, float]:
    if value is None:
        return (0.0, 0.0, 0.0)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Mesh '{mesh_name}' translation_m must be a 3-item list.")
    return tuple(float(v) for v in value)


def read_external_config(config_path: Path) -> dict:
    suffix = config_path.suffix.lower()
    if suffix == ".toml":
        with config_path.open("rb") as f:
            return tomllib.load(f)
    raise ValueError(f"Unsupported config extension for {config_path}. Use .toml.")
