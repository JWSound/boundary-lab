"""Built-in geometry-generator registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from blab.generators.base import GeneratorBackend, GeneratorCapabilities


@dataclass(frozen=True)
class GeneratorBackendInfo:
    provider_id: str
    label: str
    capabilities: GeneratorCapabilities
    factory: Callable[..., GeneratorBackend] | None = None
    available: bool = True
    description: str = ""


_BACKENDS: dict[str, GeneratorBackendInfo] = {
    "ath": GeneratorBackendInfo(
        provider_id="ath",
        label="Ath",
        capabilities=GeneratorCapabilities(source_formats=("ath_cfg",)),
        factory=lambda **kwargs: _create_ath_backend(**kwargs),
        description="Generate waveguide meshes with the bundled Ath executable.",
    ),
}


def available_generator_infos() -> tuple[GeneratorBackendInfo, ...]:
    return tuple(info for info in _BACKENDS.values() if info.available)


def generator_info(provider_id: str) -> GeneratorBackendInfo:
    normalized_id = normalize_generator_id(provider_id)
    try:
        return _BACKENDS[normalized_id]
    except KeyError as exc:
        raise ValueError(f"Unknown geometry generator: {provider_id}") from exc


def create_generator(provider_id: str, **kwargs: Any) -> GeneratorBackend:
    info = generator_info(provider_id)
    if info.factory is None:
        raise ValueError(f"Geometry generator {info.label!r} is not available.")
    return info.factory(**kwargs)


def normalize_generator_id(provider_id: str) -> str:
    text = str(provider_id or "").strip().lower()
    return {"ath4": "ath"}.get(text, text or "ath")


def _create_ath_backend(**kwargs: Any) -> GeneratorBackend:
    from blab.generators.ath import AthGeneratorBackend

    return AthGeneratorBackend(**kwargs)
