"""JSON-safe cloud solve payloads and event helpers."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from blab.config import CrossoverConfig, MeshConfig, RadiatorConfig, SimulationConfig
from blab.live import FrequencyResult


ArrayPayload = dict[str, Any]
EventPayload = dict[str, Any]


def array_to_payload(value: np.ndarray | None) -> ArrayPayload | None:
    """Encode an ndarray as plain JSON data for the prototype stream."""
    if value is None:
        return None
    array = np.asarray(value)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data": array.reshape(-1).tolist(),
    }


def array_from_payload(payload: ArrayPayload | None) -> np.ndarray | None:
    if payload is None:
        return None
    return np.asarray(payload["data"], dtype=np.dtype(payload["dtype"])).reshape(payload["shape"])


def config_to_payload(config: SimulationConfig) -> dict[str, Any]:
    return asdict(config)


def config_from_payload(payload: dict[str, Any]) -> SimulationConfig:
    data = dict(payload)
    data["meshes"] = tuple(_mesh_from_payload(item) for item in data.get("meshes", ()))
    data["radiators"] = tuple(_radiator_from_payload(item) for item in data.get("radiators", ()))
    return SimulationConfig(**data)


def _mesh_from_payload(payload: dict[str, Any]) -> MeshConfig:
    return MeshConfig(
        name=str(payload["name"]),
        file=str(payload["file"]),
        scale_factor=None if payload.get("scale_factor") is None else float(payload["scale_factor"]),
        translation_m=tuple(float(v) for v in payload.get("translation_m", (0.0, 0.0, 0.0))),
    )


def _radiator_from_payload(payload: dict[str, Any]) -> RadiatorConfig:
    return RadiatorConfig(
        name=str(payload["name"]),
        tag=int(payload["tag"]),
        mesh=None if payload.get("mesh") is None else str(payload["mesh"]),
        level_db=float(payload.get("level_db", 0.0)),
        polarity=int(payload.get("polarity", 1)),
        delay_ms=float(payload.get("delay_ms", 0.0)),
        crossover=_crossover_from_payload(payload.get("crossover", {})),
        hpf=_crossover_from_payload(payload.get("hpf", {})),
        lpf=_crossover_from_payload(payload.get("lpf", {})),
    )


def _crossover_from_payload(payload: dict[str, Any]) -> CrossoverConfig:
    return CrossoverConfig(
        type=str(payload.get("type", "none")),
        filter=str(payload.get("filter", "butterworth")),
        order=int(payload.get("order", 1)),
        frequency_hz=None if payload.get("frequency_hz") is None else float(payload["frequency_hz"]),
    )


def frequency_result_to_payload(result: FrequencyResult) -> dict[str, Any]:
    return {
        "freq_hz": float(result.freq_hz),
        "horizontal_spl_norm_db": array_to_payload(result.horizontal_spl_norm_db),
        "vertical_spl_norm_db": array_to_payload(result.vertical_spl_norm_db),
        "impedance": array_to_payload(result.impedance),
        "horizontal_spl_db": array_to_payload(result.horizontal_spl_db),
        "vertical_spl_db": array_to_payload(result.vertical_spl_db),
        "sphere_spl_norm_db": array_to_payload(result.sphere_spl_norm_db),
    }


def frequency_result_from_payload(payload: dict[str, Any]) -> FrequencyResult:
    horizontal = array_from_payload(payload["horizontal_spl_norm_db"])
    vertical = array_from_payload(payload["vertical_spl_norm_db"])
    impedance = array_from_payload(payload["impedance"])
    if horizontal is None or vertical is None or impedance is None:
        raise ValueError("Frequency result payload is missing required arrays.")
    return FrequencyResult(
        freq_hz=float(payload["freq_hz"]),
        horizontal_spl_norm_db=horizontal,
        vertical_spl_norm_db=vertical,
        impedance=impedance,
        horizontal_spl_db=array_from_payload(payload.get("horizontal_spl_db")),
        vertical_spl_db=array_from_payload(payload.get("vertical_spl_db")),
        sphere_spl_norm_db=array_from_payload(payload.get("sphere_spl_norm_db")),
    )


def initialized_event(
    *,
    job_id: str,
    polar_angle_deg: np.ndarray,
    radiator_names: np.ndarray,
    sphere_metadata: dict[str, np.ndarray] | None,
) -> EventPayload:
    sphere_metadata = sphere_metadata or {}
    return {
        "type": "initialized",
        "job_id": job_id,
        "polar_angle_deg": array_to_payload(polar_angle_deg),
        "radiator_names": array_to_payload(radiator_names),
        "sphere_metadata": {
            key: array_to_payload(value)
            for key, value in sphere_metadata.items()
        },
    }


def frequency_result_event(job_id: str, result: FrequencyResult) -> EventPayload:
    return {
        "type": "frequency_result",
        "job_id": job_id,
        "frequency_hz": float(result.freq_hz),
        "result": frequency_result_to_payload(result),
    }


def status_event(job_id: str, message: str) -> EventPayload:
    return {"type": "status", "job_id": job_id, "message": message}


def completed_event(job_id: str) -> EventPayload:
    return {"type": "completed", "job_id": job_id}


def failed_event(job_id: str, message: str) -> EventPayload:
    return {"type": "failed", "job_id": job_id, "message": message}
