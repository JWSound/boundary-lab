"""Headless CAD import: a STEP file in, a conformal FEM/BEM mesh assembly out.

Qt-free. Shared by the ``blab cad-import`` command and the CAD import geometry
provider, so both produce identical meshes for identical inputs.
"""

from .design import WALLS, CadDesign, CadFaceGroups, FaceGroupRow, design_settings
from .face_refs import CadFace, CadFaceRef, face_ref, match_face_ref
from .pipeline import (
    CadBody,
    CadClosedShell,
    CadGeometry,
    CadImportResult,
    CadImportSettings,
    CadInterface,
    CadMesh,
    CadPreview,
    CadRegion,
    CadUnmeshableFace,
    inspect_step,
    mesh_step,
    open_step,
    place_preview,
    preview_from_json,
    preview_to_json,
    result_from_json,
    result_to_json,
)
from .sizing import BEM_LEVELS, FEM_LEVELS, LEVELS, MIXED, CadSizeOverride, CadSizes, CadSizing

__all__ = [
    "BEM_LEVELS",
    "WALLS",
    "CadBody",
    "CadClosedShell",
    "CadDesign",
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
    "CadSizes",
    "CadSizing",
    "CadUnmeshableFace",
    "FEM_LEVELS",
    "LEVELS",
    "MIXED",
    "design_settings",
    "face_ref",
    "inspect_step",
    "match_face_ref",
    "mesh_step",
    "open_step",
    "place_preview",
    "preview_from_json",
    "preview_to_json",
    "result_from_json",
    "result_to_json",
]
