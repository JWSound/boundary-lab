"""Geometry generator contracts and built-in providers."""

from blab.generators.base import (
    PROVIDER_API_VERSION,
    GeneratedGeometry,
    GeneratedGeometryReference,
    GenerationCancelledError,
    GenerationCompleted,
    GenerationRequest,
    GenerationResponse,
    GeneratorBackend,
    GeneratorCapabilities,
    GeneratorDocument,
    GeneratorSession,
)
from blab.mesh_data import MeshData

__all__ = [
    "MeshData",
    "PROVIDER_API_VERSION",
    "GeneratedGeometry",
    "GeneratedGeometryReference",
    "GenerationCancelledError",
    "GenerationCompleted",
    "GenerationRequest",
    "GenerationResponse",
    "GeneratorBackend",
    "GeneratorCapabilities",
    "GeneratorDocument",
    "GeneratorSession",
]
