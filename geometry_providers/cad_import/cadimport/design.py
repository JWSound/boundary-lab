"""A CAD import design on a read geometry: its face groups, walls and interface face groups.

Qt-free. The meshing pipeline and the editor resolve face groups through it alike.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, NamedTuple

from blab.generators.configuration import exterior_sound_speed

from .face_refs import ZERO, CadFaceRef, face_ref, match_face_ref
from .settings import CadImportSettings
from .sizing import CadSizeOverride, CadSizes, CadSizing, legacy_sizes, migrated, read_face_sizes, size_override

if TYPE_CHECKING:
    from .pipeline import CadGeometry

WALLS = "walls"


class FaceGroupRow(NamedTuple):
    """A row of the face-group list: ``kind`` is ``"walls"``, ``"interface"`` or ``"group"``."""

    name: str
    faces: frozenset[int]
    kind: str


class FaceGroupEdit(NamedTuple):
    """An edit of the face groups: the design source fields it changes (none: nothing to store) and a note."""

    changes: dict
    note: str


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
        self._names = names = dict(interface_names or {})
        self.interface_names = {i.name: names.get(i.name, i.name).strip() for i in geometry.interfaces}
        self.interface_of = {f: self.interface_names[i.name] for i in geometry.interfaces for f in i.face_ids}
        self._placement = dict(rotation_deg=rotation_deg, translation_m=translation_m)
        self.found = {}  # stored group -> [(stored face, face id or None)]
        for name, refs in face_groups.items():
            self.found[name] = [(ref, _resolve(ref, geometry, self._placement)) for ref in refs]
        self._ids = {f.id for f in geometry.faces}
        # Names no interface can take besides walls: stored groups, and STEP names of faces outside interfaces.
        self._taken = {
            *self.found,
            *(f.step_name for f in geometry.faces if f.step_name and f.id not in self.interface_of),
        }
        self._reserved = reserved = {WALLS, *self.interface_names.values()}
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
            # A stored group is listed even without faces (all in interfaces, or not found), so it can be deleted.
            + [FaceGroupRow(name, frozenset(faces), "group") for name, faces in groups.items()]
        )
        self._row_of = {face: row for row in self.rows for face in row.faces}

    @property
    def held_by_interfaces(self) -> frozenset[int]:
        """Faces a stored group holds that stay in their interface face group."""
        return frozenset(f for found in self.found.values() for _, f in found if f in self.interface_of)

    def group_of(self, face: int) -> FaceGroupRow:
        """The row holding ``face``."""
        return self._row_of[face]

    def check(self) -> None:
        """Reject what generation cannot mesh: interface names that are not free, reserved or empty face group names,
        stored faces not found, unknown face ids and faces in two stored groups."""
        self._check_interface_names(self.interface_names)
        for name in self.found:
            if not name or name == WALLS:  # interface names: taken, checked above
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

    # -- edits: each returns the changed design source fields and a note, or raises ValueError ----------------

    def moved_to(self, name: str, faces) -> FaceGroupEdit:
        """Put ``faces`` into stored group ``name`` (new or existing), out of any other; ``walls`` takes them out of
        every group. Re-picking a group that needs review drops its missing faces. Interface faces are skipped."""
        name, faces = name.strip(), set(faces)
        if not name or not faces:
            raise ValueError("Select faces and enter a face group name first.")
        if name == WALLS:
            return self.moved_to_walls(faces)
        if name in self.interface_names.values():
            raise ValueError(f"Face group name '{name}' is reserved.")
        skipped = faces & self.interface_of.keys()
        faces -= skipped
        return FaceGroupEdit(self._regrouped(faces, name) if faces else {}, self.interface_note(skipped))

    def moved_to_walls(self, faces) -> FaceGroupEdit:
        """Take ``faces`` out of the stored groups. Faces with a STEP name stay in the group it forms."""
        faces = set(faces)
        if not faces:
            raise ValueError("Select faces first.")
        skipped = faces & self.interface_of.keys()
        named = sorted(
            f.id
            for f in self.geometry.faces
            if f.id in faces - skipped and f.step_name and f.step_name not in self._reserved
        )
        notes = [f"Faces {', '.join(map(str, named))} stay in the face groups their STEP names form."] if named else []
        notes.append(self.interface_note(skipped))
        return FaceGroupEdit(self._regrouped(faces), " ".join(n for n in notes if n))

    def deleted(self, name: str) -> FaceGroupEdit:
        """Delete stored group ``name``; its faces go back to walls (or their STEP-named group)."""
        if name not in self.found:
            raise ValueError(f"'{name}' is no picked face group.")
        kept = [
            {"name": n, "faces": [_stored(ref) for ref, _ in found]} for n, found in self.found.items() if n != name
        ]
        return FaceGroupEdit({"face_groups": kept}, "")

    def renamed_interface(self, name: str, new_name: str) -> FaceGroupEdit:
        """Rename interface face group ``name``; an empty ``new_name`` gives back the detected name."""
        detected = next(d for d, shown in self.interface_names.items() if shown == name)
        new_name = new_name.strip() or detected
        if new_name == name:
            return FaceGroupEdit({}, "")
        self._check_interface_names(self.interface_names | {detected: new_name})
        names = {d: s for d, s in self._names.items() if d != detected}  # kept for interfaces not detected now
        if new_name != detected:
            names[detected] = new_name
        return FaceGroupEdit({"interface_names": names}, "")

    def _check_interface_names(self, names: Mapping[str, str]) -> None:
        """Reject interface ``names`` (detected -> shown) that are empty, walls, two interfaces' or a face group's."""
        counts = Counter(names.values())
        for detected, name in names.items():
            if not name:
                raise ValueError(f"Interface '{detected}' needs a name.")
            if name == WALLS or counts[name] > 1 or name in self._taken:
                raise ValueError(f"Face group name '{name}' is taken.")

    def interface_note(self, faces) -> str:
        """Why ``faces``, interface faces, stayed where they are; empty without any."""
        if not faces:
            return ""
        return "; ".join(f"Face {f} stays in interface '{self.interface_of[f]}'" for f in sorted(faces)) + "."

    def _regrouped(self, faces: set[int], name: str | None = None) -> dict:
        """The stored groups with ``faces`` moved into group ``name`` (``None``: into none), refs taken afresh."""
        groups = []
        for group in [*self.found, *([name] if name is not None and name not in self.found else [])]:
            found = self.found.get(group, [])
            ids = {face for _, face in found if face is not None}
            unmatched = [_stored(ref) for ref, face in found if face is None]
            if group == name:
                ids, unmatched = ids | faces, []
            else:
                ids -= faces
            if ids or unmatched:
                groups.append({"name": group, "faces": unmatched + [self._ref(f) for f in sorted(ids)]})
        return {"face_groups": groups}

    def _ref(self, face_id: int) -> dict:
        face = next(f for f in self.geometry.faces if f.id == face_id)
        return asdict(face_ref(face, **self._placement))

    def surface_groups(self, mesh_faces: Mapping[str, Sequence[int]]) -> dict[str, dict[str, list[int]]]:
        """Mesh id -> {surface group name -> faces}, sorted by name, of meshes bounded by ``mesh_faces``."""
        result = {}
        for mesh_id, faces in mesh_faces.items():
            groups: dict[str, list[int]] = {}
            for face in faces:
                groups.setdefault(self._row_of[face].name, []).append(face)
            result[mesh_id] = dict(sorted(groups.items()))
        return result


# -- the design source as the editor edits it --------------------------------------------------------------------------

FEM_FINENESS = "elements_per_wavelength"  # design source keys of the mesh finenesses
BEM_FINENESS = "bem_elements_per_wavelength"


def source_placement(source: Mapping) -> dict:
    """The placement of design ``source`` as the pipeline functions take it (translation in metres)."""
    return dict(
        rotation_deg=tuple(source.get("rotation_deg", ZERO)),
        translation_m=_mm_to_m(source.get("translation_mm", ZERO)),
    )


def _mm_to_m(values: Sequence) -> tuple:
    """``values`` (millimetres) in metres; values that are no numbers are left for the check."""
    return tuple(v / 1000 if isinstance(v, int | float) else v for v in values)


class CadDesign:
    """A design ``source`` on the read ``geometry`` (``None`` before one is read): the one place the editor reads
    its face groups, sizes, fineness and placement, and the one that says what an edit changes.

    ``placement`` is the one ``geometry`` was read with (default: the source's). Edits return the design source
    fields they change; the editor stores them. ``groups`` (:class:`CadFaceGroups`) is ``None`` without a geometry.
    Stored sizes that cannot be read leave ``size_error`` set and ``sizes`` empty: shown Global, never written over.
    """

    def __init__(self, source: Mapping, geometry: CadGeometry | None = None, *, placement: Mapping | None = None):
        self.source = dict(source)
        self.geometry = geometry
        self.placement = dict(placement or source_placement(source))
        self.groups = self.sizes = None
        self.size_error = ""
        self._targets = None  # (sizing, face -> target fineness)
        if geometry is None:
            return
        self.groups = CadFaceGroups(
            geometry,
            {g["name"]: g["faces"] for g in self.source.get("face_groups", [])},
            interface_names=self.source.get("interface_names", {}),
            **self.placement,
        )
        try:
            self.sizes = CadSizes(
                geometry, self.source.get("face_sizes", []), self.source.get("volume_sizes", {}), **self.placement
            )
        except ValueError as exc:  # as Generate rejects it
            self.sizes = CadSizes(geometry, [], {}, **self.placement)
            self.size_error = str(exc)

    @property
    def rotation_deg(self) -> tuple:
        return tuple(self.source.get("rotation_deg", ZERO))

    @property
    def translation_mm(self) -> tuple:
        return tuple(self.source.get("translation_mm", ZERO))

    @property
    def stored_groups(self) -> set[str]:
        """The names of the face groups the source stores (not those only named in the STEP file)."""
        return {g["name"] for g in self.source.get("face_groups", [])}

    def number(self, key: str) -> float:
        """The numeric setting ``key`` (a fineness, curvature or growth rate); the BEM fineness
        (:data:`BEM_FINENESS`) is the FEM one (:data:`FEM_FINENESS`) when the source has none."""
        if key == BEM_FINENESS:
            return float(self.source.get(key, self.source[FEM_FINENESS]))
        return float(self.source[key])

    def sizing(self, fmax_hz: float | None, sound_speed_m_per_s: float) -> CadSizing:
        """The design's sizing at the project's maximum frequency (``None`` while unknown) and sound speed."""
        return CadSizing(
            fmax_hz, sound_speed_m_per_s, float(self.source[FEM_FINENESS]), self.source.get(BEM_FINENESS)
        )

    def targets(self, sizing: CadSizing, faces=(), volumes=()) -> list[CadSizeOverride]:
        """The mesh fineness of the target element size of each of ``faces`` then each air-body ``volumes``: what the
        size plan meshes them with unless an override wins."""
        if self._targets is None or self._targets[0] != sizing:  # per change, not per row
            self._targets = (sizing, sizing.face_targets(self.geometry))
        return [self._targets[1][f] for f in faces] + [sizing.fineness] * len(volumes)

    def set_sizes(self, size, faces=(), volumes=()) -> dict:
        """The size fields with ``size`` (an override or its stored fields; ``None``: Global) on each of ``faces``
        and air-body ``volumes``. Raises ``ValueError`` for a bad size, or for stored sizes that cannot be read."""
        if self.size_error:
            raise ValueError(self.size_error)
        return self.sizes.written(None if size is None else size_override(size), faces, volumes)

    def dropped(self) -> tuple[dict, str]:
        """The size fields without sizes of faces the geometry no longer has and of bodies that are no air bodies,
        and a note saying so; ``({}, "")`` when there is none (or the stored sizes cannot be read)."""
        return ({}, "") if self.size_error or self.sizes is None else self.sizes.dropped()

    def migrated(self, sizing: CadSizing) -> dict | None:
        """The whole source with older body and face-group sizes turned into face and volume sizes (see
        :func:`cadimport.sizing.migrated`), else ``None``; ``sizing`` needs the maximum frequency."""
        if self.groups is None or self.size_error or sizing.fmax_hz is None:
            return None
        return migrated(self.source, self.groups, self.sizes, sizing)


# -- the design source -> pipeline settings ----------------------------------------------------------------------------


def design_settings(source: Mapping, configuration: Mapping) -> CadImportSettings:
    """Pipeline settings of design ``source`` with the maximum frequency, sound speed and symmetry of the project
    ``configuration`` snapshot.

    The one reader of the source's keys: face groups, mesh finenesses, size overrides (also the older body and
    face-group sizes), placement (translation in millimetres there, metres in the settings) and closed shells.
    Faces and sizes resolve against the read geometry in :func:`mesh_step`, with :class:`CadFaceGroups` and
    :class:`CadSizes`. Rejects what generation cannot take (``ValueError``) before any meshing.
    """
    try:
        body_sizes, face_group_sizes = legacy_sizes(source)  # older designs, until the editor converts them
        return CadImportSettings(
            face_groups=_face_groups(source),
            fmax_hz=float(configuration["frequency"]["freq_max_hz"]),
            sound_speed_m_per_s=exterior_sound_speed(configuration),
            elements_per_wavelength=float(source["elements_per_wavelength"]),
            bem_elements_per_wavelength=_bem_elements_per_wavelength(source),
            body_sizes=body_sizes,
            face_group_sizes=face_group_sizes,
            face_sizes=tuple(read_face_sizes(source.get("face_sizes", []))),
            volume_sizes=dict(source.get("volume_sizes", {})),
            curvature_elements_per_circle=float(source["curvature_elements_per_circle"]),
            growth_rate=float(source["growth_rate"]),
            symmetry=configuration.get("symmetry_config", {}).get("mode", "off"),
            closed_shells=dict(source.get("closed_shells", {})),
            rotation_deg=tuple(source["rotation_deg"]),
            translation_m=_mm_to_m(source["translation_mm"]),
            interface_names=dict(source.get("interface_names", {})),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid CAD import source: {exc}") from exc


def _face_groups(source: Mapping) -> dict:
    """``{name: faces}`` from the source's list of ``{"name", "faces": [CadFaceRef fields]}``."""
    groups = source.get("face_groups", [])
    if not isinstance(groups, list):
        raise ValueError("Face groups must be a list.")
    result = {}
    for group in groups:
        if group["name"] in result:
            raise ValueError(f"Face group '{group['name']}' is defined twice.")
        result[group["name"]] = tuple(group["faces"])
    return result


def _bem_elements_per_wavelength(source: Mapping) -> float | None:
    """The BEM mesh fineness as a float; ``None`` (older designs): the FEM value."""
    value = source.get("bem_elements_per_wavelength")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("BEM elements per wavelength must be a positive number.") from exc


def _stored(ref):
    return asdict(ref) if isinstance(ref, CadFaceRef) else ref


def _resolve(ref, geometry: CadGeometry, placement) -> int | None:
    if isinstance(ref, int):
        return ref  # a face id: unknown ones are rejected by the check
    ref = ref if isinstance(ref, CadFaceRef) else CadFaceRef(**ref)
    return match_face_ref(ref, geometry.faces, geometry.bbox_m, **placement)
