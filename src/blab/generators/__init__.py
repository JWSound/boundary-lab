"""Geometry generator contracts and built-in providers."""

from blab.generators.base import (
    PROVIDER_API_VERSION,
    GeneratedGeometry,
    GeneratedGeometryReference,
    GeneratedMesh,
    GenerationCancelledError,
    GenerationCompleted,
    GenerationRequest,
    GenerationResponse,
    GeneratorBackend,
    GeneratorCapabilities,
    GeneratorDocument,
    GeneratorSession,
    generated_mesh_id,
)
from blab.generators.host import ProviderContext, ProviderEvent, ProviderHost, ProviderHostError, SolveCommand, SolveJob
from blab.mesh_data import MeshData

__all__ = [
    "MeshData",
    "ProviderContext",
    "ProviderEvent",
    "ProviderHost",
    "ProviderHostError",
    "SolveCommand",
    "SolveJob",
    "PROVIDER_API_VERSION",
    "GeneratedGeometry",
    "GeneratedGeometryReference",
    "GeneratedMesh",
    "generated_mesh_id",
    "GenerationCancelledError",
    "GenerationCompleted",
    "GenerationRequest",
    "GenerationResponse",
    "GeneratorBackend",
    "GeneratorCapabilities",
    "GeneratorDocument",
    "GeneratorSession",
]
