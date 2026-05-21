"""Wire-format helpers for Boundary Lab solve jobs."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np

from blab.config import CrossoverConfig, MeshConfig, RadiatorConfig, SimulationConfig
from blab.live import FrequencyResult


PROTOCOL_VERSION = 1


def crossover_to_dict(crossover: CrossoverConfig) -> dict[str, Any]:
    return {
        "type": crossover.type,
        "filter": crossover.filter,
        "order": int(crossover.order),
        "frequency_hz": None if crossover.frequency_hz is None else float(crossover.frequency_hz),
    }


def crossover_from_dict(raw: dict[str, Any] | None) -> CrossoverConfig:
    raw = raw or {}
    return CrossoverConfig(
        type=str(raw.get("type", "none")),
        filter=str(raw.get("filter", "butterworth")),
        order=int(raw.get("order", 1)),
        frequency_hz=None if raw.get("frequency_hz") is None else float(raw["frequency_hz"]),
    )


def mesh_to_dict(mesh: MeshConfig) -> dict[str, Any]:
    return {
        "name": mesh.name,
        "file": mesh.file,
        "scale_factor": mesh.scale_factor,
        "translation_m": [float(value) for value in mesh.translation_m],
    }


def mesh_from_dict(raw: dict[str, Any]) -> MeshConfig:
    translation = raw.get("translation_m", (0.0, 0.0, 0.0))
    if len(translation) != 3:
        raise ValueError("Mesh translation_m must contain three values.")
    return MeshConfig(
        name=str(raw["name"]),
        file=str(raw["file"]),
        scale_factor=None if raw.get("scale_factor") is None else float(raw["scale_factor"]),
        translation_m=tuple(float(value) for value in translation),
    )


def radiator_to_dict(radiator: RadiatorConfig) -> dict[str, Any]:
    return {
        "name": radiator.name,
        "tag": int(radiator.tag),
        "mesh": radiator.mesh,
        "level_db": float(radiator.level_db),
        "polarity": int(radiator.polarity),
        "delay_ms": float(radiator.delay_ms),
        "crossover": crossover_to_dict(radiator.crossover),
        "hpf": crossover_to_dict(radiator.hpf),
        "lpf": crossover_to_dict(radiator.lpf),
    }


def radiator_from_dict(raw: dict[str, Any]) -> RadiatorConfig:
    return RadiatorConfig(
        name=str(raw["name"]),
        tag=int(raw["tag"]),
        mesh=None if raw.get("mesh") is None else str(raw["mesh"]),
        level_db=float(raw.get("level_db", 0.0)),
        polarity=int(raw.get("polarity", 1)),
        delay_ms=float(raw.get("delay_ms", 0.0)),
        crossover=crossover_from_dict(raw.get("crossover")),
        hpf=crossover_from_dict(raw.get("hpf")),
        lpf=crossover_from_dict(raw.get("lpf")),
    )


def simulation_config_to_dict(config: SimulationConfig) -> dict[str, Any]:
    payload = {
        field.name: getattr(config, field.name)
        for field in fields(SimulationConfig)
        if field.name not in {"meshes", "radiators"}
    }
    payload["meshes"] = [mesh_to_dict(mesh) for mesh in config.meshes]
    payload["radiators"] = [radiator_to_dict(radiator) for radiator in config.radiators]
    return payload


def simulation_config_from_dict(raw: dict[str, Any]) -> SimulationConfig:
    values = {
        field.name: raw[field.name]
        for field in fields(SimulationConfig)
        if field.name in raw and field.name not in {"meshes", "radiators"}
    }
    values["meshes"] = tuple(mesh_from_dict(mesh) for mesh in raw.get("meshes", ()))
    values["radiators"] = tuple(radiator_from_dict(radiator) for radiator in raw.get("radiators", ()))
    return SimulationConfig(**values)


def ndarray_to_wire(array: np.ndarray | None) -> list | None:
    if array is None:
        return None
    return np.asarray(array, dtype=np.float32).tolist()


def ndarray_from_wire(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    return np.asarray(raw, dtype=np.float32)


def frequency_result_to_dict(result: FrequencyResult) -> dict[str, Any]:
    return {
        "freq_hz": float(result.freq_hz),
        "horizontal_spl_norm_db": ndarray_to_wire(result.horizontal_spl_norm_db),
        "vertical_spl_norm_db": ndarray_to_wire(result.vertical_spl_norm_db),
        "impedance": ndarray_to_wire(result.impedance),
        "horizontal_spl_db": ndarray_to_wire(result.horizontal_spl_db),
        "vertical_spl_db": ndarray_to_wire(result.vertical_spl_db),
        "sphere_spl_norm_db": ndarray_to_wire(result.sphere_spl_norm_db),
    }


def frequency_result_from_dict(raw: dict[str, Any]) -> FrequencyResult:
    return FrequencyResult(
        freq_hz=float(raw["freq_hz"]),
        horizontal_spl_norm_db=ndarray_from_wire(raw["horizontal_spl_norm_db"]),
        vertical_spl_norm_db=ndarray_from_wire(raw["vertical_spl_norm_db"]),
        impedance=ndarray_from_wire(raw["impedance"]),
        horizontal_spl_db=ndarray_from_wire(raw.get("horizontal_spl_db")),
        vertical_spl_db=ndarray_from_wire(raw.get("vertical_spl_db")),
        sphere_spl_norm_db=ndarray_from_wire(raw.get("sphere_spl_norm_db")),
    )


def solve_request_to_config_and_frequencies(raw: dict[str, Any]) -> tuple[SimulationConfig, np.ndarray]:
    if int(raw.get("schema_version", PROTOCOL_VERSION)) != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported solve request schema_version {raw.get('schema_version')}.")
    if "config" not in raw:
        raise ValueError("Solve request must include config.")
    if "frequencies_hz" not in raw:
        raise ValueError("Solve request must include frequencies_hz.")
    return (
        simulation_config_from_dict(raw["config"]),
        np.asarray(raw["frequencies_hz"], dtype=np.float32),
    )


def solve_request_from_config_and_frequencies(
    config: SimulationConfig,
    frequencies_hz: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_VERSION,
        "config": simulation_config_to_dict(config),
        "frequencies_hz": ndarray_to_wire(np.asarray(frequencies_hz, dtype=np.float32)),
    }
