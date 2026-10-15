"""Headless CAD import: a STEP file in, a conformal FEM/BEM mesh assembly out.

Qt-free. Shared by the ``blab cad-import`` command and the CAD import geometry
provider, so both produce identical meshes for identical inputs.
"""

from .design import CadFaceGroups, FaceGroupRow
from .pipeline import (
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
    CadSizeOverride,
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
    "WALLS",
    "CadBody",
    "CadClosedShell",
    "CadFace",
    "CadFaceGroups",
    "CadFaceRef",
    "FaceGroupRow",
    "CadGeometry",
    "CadImportResult",
    "CadImportSettings",
    "CadInterface",
    "CadMesh",
    "CadPreview",
    "CadRegion",
    "CadSizeOverride",
    "face_ref",
    "inspect_step",
    "match_face_ref",
    "mesh_step",
    "open_step",
    "place_preview",
    "preview_from_json",
    "preview_to_json",
]
