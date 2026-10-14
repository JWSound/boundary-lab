"""Headless CAD import: a STEP file in, a conformal FEM/BEM mesh assembly out.

Qt-free. Shared by the ``blab cad-import`` command and the CAD import geometry
provider, so both produce identical meshes for identical inputs.
"""

from .pipeline import (
    AXES,
    WALLS,
    CadBody,
    CadClosedShell,
    CadFace,
    CadFaceRef,
    CadGeometry,
    CadImportResult,
    CadImportSettings,
    CadInterface,
    CadMesh,
    CadPreview,
    CadRegion,
    CadScan,
    face_ref,
    inspect_step,
    match_face_ref,
    mesh_step,
    open_step,
    place_preview,
    preview_from_json,
    preview_to_json,
)

__all__ = [
    "AXES",
    "WALLS",
    "CadBody",
    "CadClosedShell",
    "CadFace",
    "CadFaceRef",
    "CadGeometry",
    "CadImportResult",
    "CadImportSettings",
    "CadInterface",
    "CadMesh",
    "CadPreview",
    "CadRegion",
    "CadScan",
    "face_ref",
    "inspect_step",
    "match_face_ref",
    "mesh_step",
    "open_step",
    "place_preview",
    "preview_from_json",
    "preview_to_json",
]
