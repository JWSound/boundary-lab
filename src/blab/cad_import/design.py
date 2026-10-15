"""A CAD import design on a read geometry: its face groups, walls and interface face groups.

Qt-free. The meshing pipeline and the editor resolve face groups through it alike.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from .pipeline import WALLS, ZERO, CadFaceRef, CadGeometry, match_face_ref


class FaceGroupRow(NamedTuple):
    """A row of the face-group list: ``kind`` is ``"walls"``, ``"interface"`` or ``"group"``."""

    name: str
    faces: frozenset[int]
    kind: str


class CadFaceGroups:
    """The face groups of ``geometry``: walls, the interface face groups, then the stored face groups and those
    named in the STEP file.

    ``face_groups`` maps a stored group's name to its faces: ``CadFaceRef`` (or mappings of its fields), or face
    ids. A stored face resolves once, when built (see :func:`match_face_ref`); ``rotation_deg`` and
    ``translation_m`` are the placement ``geometry`` was read with. Faces of a detected interface always stay in
    its interface face group, even when a stored group holds them. Faces with a STEP name in no stored group form
    a group of that name. Walls are the faces in no other group. ``interface_names`` renames interfaces by
    detected name.
    """

    def __init__(
        self,
        geometry: CadGeometry,
        face_groups: Mapping[str, Sequence[int | CadFaceRef | Mapping]],
        *,
        interface_names: Mapping[str, str] | None = None,
        rotation_deg: Sequence[float] = ZERO,
        translation_m: Sequence[float] = ZERO,
    ) -> None:
        self.geometry = geometry
        names = interface_names or {}
        self.interface_names = {i.name: names.get(i.name, i.name) for i in geometry.interfaces}
        self.interface_of = {f: self.interface_names[i.name] for i in geometry.interfaces for f in i.face_ids}
        placement = dict(rotation_deg=rotation_deg, translation_m=translation_m)
        self.found = {}  # stored group -> [(stored face, face id or None)]
        for name, refs in face_groups.items():
            self.found[name] = [(ref, _resolve(ref, geometry, placement)) for ref in refs]
        self._ids = {f.id for f in geometry.faces}
        reserved = {WALLS, *self.interface_names.values()}
        groups: dict[str, set[int]] = {}
        member: dict[int, str] = {}
        for name, found in self.found.items():
            faces = groups.setdefault(name, set())
            for _, face in found:
                if face in self._ids and face not in self.interface_of and face not in member:
                    faces.add(face)
                    member[face] = name
        for face in geometry.faces:
            name = face.step_name
            if name and name not in reserved and face.id not in member and face.id not in self.interface_of:
                groups.setdefault(name, set()).add(face.id)
                member[face.id] = name
        self.needs_review = tuple(name for name, found in self.found.items() if any(f is None for _, f in found))
        interfaces: dict[str, set[int]] = {}
        for face, name in self.interface_of.items():
            interfaces.setdefault(name, set()).add(face)
        walls = self._ids - member.keys() - self.interface_of.keys()
        self.rows = (
            [FaceGroupRow(WALLS, frozenset(walls), "walls")]
            + [FaceGroupRow(name, frozenset(faces), "interface") for name, faces in interfaces.items()]
            + [
                FaceGroupRow(name, frozenset(faces), "group")
                for name, faces in groups.items()
                if faces or name in self.needs_review
            ]
        )
        self._row_of = {face: row for row in self.rows for face in row.faces}

    @property
    def held_by_interfaces(self) -> frozenset[int]:
        """Faces a stored group holds that stay in their interface face group."""
        return frozenset(f for found in self.found.values() for _, f in found if f in self.interface_of)

    def row(self, name: str) -> FaceGroupRow | None:
        return next((row for row in self.rows if row.name == name), None)

    def group_of(self, face: int) -> FaceGroupRow:
        """The row holding ``face``."""
        return self._row_of[face]

    def check(self) -> None:
        """Reject what generation cannot mesh: reserved or empty names, stored faces not found, unknown face ids and
        faces in two stored groups."""
        reserved = {WALLS, *self.interface_names.values()}
        for name in self.found:
            if not name or name in reserved:
                raise ValueError(f"Face group name '{name}' is reserved.")
        # A group missing any stored face would silently lose it to walls: the user reviews it.
        if self.needs_review:
            raise ValueError(
                "Face groups need review (stored faces not found in the STEP file): "
                f"{', '.join(repr(n) for n in self.needs_review)}. Re-pick or delete them."
            )
        member = {}
        for name, found in self.found.items():
            for _, face in found:
                if face not in self._ids:
                    raise ValueError(f"Face group '{name}' refers to unknown face {face}.")
                if face in self.interface_of:
                    continue
                if member.setdefault(face, name) != name:
                    raise ValueError(f"Face {face} is in face groups '{member[face]}' and '{name}'.")

    def surface_groups(self, mesh_faces: Mapping[str, Sequence[int]]) -> dict[str, dict[str, list[int]]]:
        """Mesh id -> {surface group name -> faces}, sorted by name, of meshes bounded by ``mesh_faces``."""
        result = {}
        for mesh_id, faces in mesh_faces.items():
            groups: dict[str, list[int]] = {}
            for face in faces:
                groups.setdefault(self._row_of[face].name, []).append(face)
            result[mesh_id] = dict(sorted(groups.items()))
        return result


def _resolve(ref, geometry: CadGeometry, placement) -> int | None:
    if isinstance(ref, int):
        return ref  # a face id: unknown ones are rejected by the check
    ref = ref if isinstance(ref, CadFaceRef) else CadFaceRef(**ref)
    return match_face_ref(ref, geometry.faces, geometry.bbox_m, **placement)
