"""Built-in geometry-generator registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from blab.generators.base import GeneratedGeometry, GeneratorBackend, GeneratorCapabilities, GeneratorDocument


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


def register_generator(info: GeneratorBackendInfo) -> None:
    """Register an explicitly loaded provider without replacing an existing one.

    Package discovery is separate from registration. Hosts must explicitly load
    provider code; merely opening a project never imports a plugin.
    """
    if not info.provider_id or normalize_generator_id(info.provider_id) != info.provider_id:
        raise ValueError("Provider ID must be nonempty, normalized lowercase text.")
    if info.provider_id in _BACKENDS:
        raise ValueError(f"Geometry provider {info.provider_id!r} is already registered.")
    if not info.label.strip() or not callable(info.factory):
        raise ValueError("A geometry provider needs a label and callable factory.")
    _BACKENDS[info.provider_id] = info


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


def restore_generator_document(document: GeneratorDocument) -> GeneratedGeometry | None:
    artifact = document.artifact
    if artifact is not None and artifact.mesh_data is not None:
        return GeneratedGeometry(
            provider_id=document.provider_id,
            output_dir=Path(artifact.output_dir),
            mesh_path=None,
            radiators=(),
            mirror_axes=artifact.mirror_axes,
            mesh_data=artifact.mesh_data,
            reduced_mesh_data=artifact.reduced_mesh_data,
        )
    return create_generator(document.provider_id).restore(document)


def normalize_generator_id(provider_id: str) -> str:
    text = str(provider_id or "").strip().lower()
    return {"ath4": "ath"}.get(text, text or "ath")


def _create_ath_backend(**kwargs: Any) -> GeneratorBackend:
    from blab.generators.ath import AthGeneratorBackend

    return AthGeneratorBackend(**kwargs)
