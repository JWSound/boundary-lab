"""Headless CAD import: a STEP file in, a conformal FEM/BEM mesh assembly out.

Qt-free. Shared by the ``blab cad-import`` command and the CAD import geometry
provider, so both produce identical meshes for identical inputs.
"""

from .pipeline import (
    CadBody,
    CadClosedShell,
    CadFace,
    CadGeometry,
    CadImportResult,
    CadImportSettings,
    CadInterface,
    CadMesh,
    CadPreview,
    CadRegion,
    inspect_step,
    mesh_step,
    preview_step,
)

__all__ = [
    "CadBody",
    "CadClosedShell",
    "CadFace",
    "CadGeometry",
    "CadImportResult",
    "CadImportSettings",
    "CadInterface",
    "CadMesh",
    "CadPreview",
    "CadRegion",
    "inspect_step",
    "mesh_step",
    "preview_step",
]
