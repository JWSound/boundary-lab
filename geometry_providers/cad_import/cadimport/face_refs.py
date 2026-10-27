"""Faces of a read CAD geometry, how a design stores them, and the placement both are seen through.

Qt-free. The pipeline, the face-group module and the sizing module share it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

ZERO = (0.0, 0.0, 0.0)  # no rotation, no translation
# Fingerprint tolerances for small CAD edits (see docs/specs/cad-import.md, "Re-import matching").
_CENTRE_TOLERANCE = 1e-2  # fraction of the bounding-box diagonal
_AREA_TOLERANCE = 5e-2  # fraction of the stored area
_NORMAL_COSINE = 0.999  # minimum cosine between stored and found normals


@dataclass(frozen=True)
class CadFaceRef:
    """A face as a face group stores it: STEP name and fingerprint in CAD axes (before the placement).

    It resolves to one face: the only face carrying ``step_name``, else the face with the same
    surface type, centre, area and normal (among those carrying ``step_name``, if several do).
    """

    step_name: str | None
    centre: tuple[float, float, float]
    area: float
    normal: tuple[float, float, float]
    surface_type: str


@dataclass(frozen=True)
class CadFace:
    """A face of the fragmented geometry; ``id`` is stable for the same file and placement."""

    id: int
    bodies: tuple[str, ...]
    step_name: str | None
    centre: tuple[float, float, float]
    area: float
    normal: tuple[float, float, float]
    surface_type: str


def face_ref(
    face: CadFace,
    *,
    rotation_deg: Sequence[float] = ZERO,
    translation_m: Sequence[float] = ZERO,
) -> CadFaceRef:
    """How a face group stores ``face`` of a geometry read with that placement: in CAD axes."""
    matrix, offset = placement(rotation_deg, translation_m)
    return CadFaceRef(
        face.step_name,
        tuple(float(v) for v in matrix.T @ (np.asarray(face.centre, dtype=float) - offset)),
        float(face.area),
        tuple(float(v) for v in matrix.T @ np.asarray(face.normal, dtype=float)),
        face.surface_type,
    )


def match_face_ref(
    ref: CadFaceRef,
    faces: Sequence[CadFace],
    bbox_m,
    *,
    rotation_deg: Sequence[float] = ZERO,
    translation_m: Sequence[float] = ZERO,
) -> int | None:
    """Id of the face among ``faces`` (read with that placement, bounded by ``bbox_m``) that ``ref`` stands for.

    The only face carrying its STEP name; else, among the faces with that name (or all faces
    when none has it), the one face whose fingerprint matches after the placement. Several
    matching faces are ambiguous and match none.
    """
    named = [f for f in faces if ref.step_name and f.step_name == ref.step_name]
    if len(named) == 1:
        return named[0].id
    matrix, offset = placement(rotation_deg, translation_m)
    centre, normal = moved(matrix, ref.centre, offset), moved(matrix, ref.normal)
    tol = _CENTRE_TOLERANCE * math.dist(*bbox_m)
    matches = [
        f
        for f in named or faces
        if f.surface_type == ref.surface_type
        and math.dist(f.centre, centre) <= tol
        and abs(f.area - ref.area) <= _AREA_TOLERANCE * ref.area
        and np.dot(f.normal, normal) >= _NORMAL_COSINE
    ]
    return matches[0].id if len(matches) == 1 else None


def placement(rotation_deg=ZERO, translation_m=ZERO) -> tuple[np.ndarray, np.ndarray]:
    """``(matrix, offset)`` placing CAD point p at ``matrix @ p + offset``: the rotations about the
    fixed X, Y and Z axes (in that order, about the origin), then the translation."""
    matrix = np.eye(3)
    for label, values in (("Rotation", rotation_deg), ("Translation", translation_m)):
        if not (len(values) == 3 and all(isinstance(v, int | float) and math.isfinite(v) for v in values)):
            raise ValueError(f"{label} must be three finite numbers.")
    for axis, angle in enumerate(math.radians(v) for v in rotation_deg):
        i, j = (axis + 1) % 3, (axis + 2) % 3
        turn = np.eye(3)
        turn[i, i] = turn[j, j] = math.cos(angle)
        turn[j, i], turn[i, j] = math.sin(angle), -math.sin(angle)
        matrix = turn @ matrix
    return matrix, np.array(translation_m, dtype=float)


def moved(matrix, vector, offset=ZERO) -> tuple[float, float, float]:
    return tuple(float(v) for v in matrix @ np.asarray(vector, dtype=float) + offset)
