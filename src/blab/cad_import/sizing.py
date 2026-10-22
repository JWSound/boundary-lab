"""Mesh sizes of a CAD import design: the target element size, size overrides, their text, and which one wins.

Qt-free. The pipeline, the provider backend and the editor all size through it. A size override is a level (elements
per wavelength at the maximum frequency) or an edge length; ``None`` stands for no override (the target element
size, "Global"). The editor stores overrides only on faces and air-body volumes (ADR 0001).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from blab.defaults import DEFAULT_SOUND_SPEED_M_PER_S

from .face_refs import ZERO, CadFaceRef, face_ref, match_face_ref

if TYPE_CHECKING:
    from .design import CadFaceGroups
    from .pipeline import CadGeometry, CadImportSettings

# Mesh size levels, fine to coarse: elements per wavelength at the maximum frequency (ADR 0002).
FEM_LEVELS = {"FEM Fine": 6, "FEM Medium": 4, "FEM Coarse": 2}
BEM_LEVELS = {"BEM Fine": 12, "BEM Medium": 5, "BEM Coarse": 0.3}
LEVELS = FEM_LEVELS | BEM_LEVELS
LEVEL_NAMES = {n: name for name, n in LEVELS.items()}
MIXED = "mixed"  # the shown size of several faces or volumes whose sizes differ


@dataclass(frozen=True)
class CadSizeOverride:
    """A size override: elements per wavelength at the maximum frequency (a mesh fineness), or an edge length."""

    elements_per_wavelength: float | None = None
    size_m: float | None = None

    def __post_init__(self) -> None:
        if (self.elements_per_wavelength is None) == (self.size_m is None):
            raise ValueError("Size override must be elements per wavelength or an edge length.")
        for label, value in (("elements per wavelength", self.elements_per_wavelength), ("edge length", self.size_m)):
            if value is not None and not (isinstance(value, int | float) and math.isfinite(value) and value > 0):
                raise ValueError(f"Size override {label} must be a positive number.")

    def stored(self) -> dict:
        """As a design source stores it: ``{"elements_per_wavelength": N}`` or ``{"size_m": x}``."""
        return {k: v for k, v in asdict(self).items() if v is not None}


def size_override(size: CadSizeOverride | Mapping) -> CadSizeOverride:
    """``size``, or the override its stored fields give (validated)."""
    return size if isinstance(size, CadSizeOverride) else CadSizeOverride(**size)


def read_face_sizes(stored: Sequence[Mapping]) -> list[tuple[CadFaceRef, CadSizeOverride]]:
    """``(face, override)`` of each design source ``face_sizes`` entry ``{"face": CadFaceRef fields, **override}``."""
    result = []
    for entry in stored:
        try:
            result.append((CadFaceRef(**entry["face"]), size_override({k: v for k, v in entry.items() if k != "face"})))
        except ValueError as exc:
            raise ValueError(f"Invalid face size: {exc}") from exc
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("Invalid face size: a face and its size override expected.") from exc
    return result


@dataclass(frozen=True)
class CadSizing:
    """Edge lengths from the sound speed and the maximum frequency (``None`` while unknown) at the FEM and BEM mesh
    finenesses (ADR 0002); without a BEM fineness the FEM one serves both."""

    fmax_hz: float | None
    sound_speed_m_per_s: float = DEFAULT_SOUND_SPEED_M_PER_S
    elements_per_wavelength: float = 6.0
    bem_elements_per_wavelength: float | None = None

    @property
    def fineness(self) -> CadSizeOverride:
        """The FEM mesh fineness."""
        return CadSizeOverride(elements_per_wavelength=self.elements_per_wavelength)

    @property
    def bem_fineness(self) -> CadSizeOverride:
        n = self.bem_elements_per_wavelength
        return CadSizeOverride(elements_per_wavelength=self.elements_per_wavelength if n is None else n)

    @property
    def target_size_m(self) -> float:
        """The FEM target element size: sound speed / maximum frequency / elements per wavelength."""
        return self.edge_m(self.fineness)

    @property
    def bem_target_size_m(self) -> float:
        return self.edge_m(self.bem_fineness)

    def face_targets(self, geometry: CadGeometry) -> dict[int, CadSizeOverride]:
        """Face -> the mesh fineness of its target element size: FEM on air-body faces, BEM on exterior surface
        faces, the finer of the two on a face on both (an interface)."""
        air = {f for body in geometry.bodies if body.kind == "air" for f in body.face_ids}
        exterior = {f for body in geometry.bodies if body.kind == "exterior" for f in body.face_ids}
        exterior |= {f for interface in geometry.interfaces for f in interface.face_ids}
        both = max(self.fineness, self.bem_fineness, key=lambda size: size.elements_per_wavelength)
        return {
            f.id: both if f.id in air and f.id in exterior else self.bem_fineness if f.id in exterior else self.fineness
            for f in geometry.faces
        }

    def edge_m(self, size: CadSizeOverride | None) -> float:
        """The edge length ``size`` asks for; ``None``: the FEM target element size."""
        size = size or self.fineness
        if size.size_m is not None:
            return size.size_m
        if self.fmax_hz is None:
            raise ValueError("The maximum frequency is not known yet.")
        return self.sound_speed_m_per_s / self.fmax_hz / size.elements_per_wavelength

    def text(self, size: CadSizeOverride | str | None, *, edge: bool = False) -> str:
        """The text of ``size`` (an override, ``None`` for Global, or :data:`MIXED`); ``edge`` adds a level's edge
        length once the maximum frequency is known."""
        if size == MIXED:
            return "Mixed"
        if size is None:
            return f"Global ({self.text(self.fineness, edge=edge)})"
        if size.size_m is not None:
            return f"{size.size_m * 1000:.3g} mm"
        n = size.elements_per_wavelength
        text = LEVEL_NAMES.get(n, f"{n:g} per wavelength")
        if edge and self.fmax_hz is not None:
            text += f" ({self.edge_m(size) * 1000:.3g} mm)"
        return text

    def global_text(self, targets: Sequence[CadSizeOverride]) -> str:
        """Global, naming the mesh fineness when all of a row's members ``targets`` (see :meth:`face_targets`) share
        it."""
        level = shared(list(targets))
        return "Global" if level == MIXED or level is None else f"Global ({self.text(level)})"

    def winning(
        self,
        face: CadSizeOverride | None,
        group: CadSizeOverride | None,
        bodies: Sequence[CadSizeOverride | None],
        target: CadSizeOverride | None = None,
    ) -> CadSizeOverride | None:
        """The override that sizes a face with its own ``face`` override, its face group's and its ``bodies``' (one
        per body, ``None`` for none): the most specific, else the finest body's; a body without one counts as the
        face's ``target`` (see :meth:`face_targets`; ``None``: the FEM one), so ``None`` when that is finer."""
        if face is not None or group is not None:
            return face or group
        sized = [size for size in bodies if size is not None]
        finest = min(sized, key=self.edge_m, default=None)
        if finest is not None and (len(sized) == len(bodies) or self.edge_m(finest) < self.edge_m(target)):
            return finest
        return None


def shared(sizes: Sequence[CadSizeOverride | None]) -> CadSizeOverride | str | None:
    """The size all of ``sizes`` are, else :data:`MIXED`; no sizes is ``None`` (Global)."""
    return sizes[0] if sizes and all(size == sizes[0] for size in sizes) else (MIXED if sizes else None)


class CadSizes:
    """The face and volume sizes of a design on a read geometry.

    ``face_sizes`` are stored entries ``{"face": CadFaceRef fields, **override}``, each resolved once, when built, on
    ``geometry`` read with that placement; ``volume_sizes`` map an air body to its override.
    """

    def __init__(
        self,
        geometry: CadGeometry,
        face_sizes: Sequence[Mapping],
        volume_sizes: Mapping[str, Mapping],
        *,
        rotation_deg: Sequence[float] = ZERO,
        translation_m: Sequence[float] = ZERO,
    ) -> None:
        self.geometry = geometry
        self._placement = dict(rotation_deg=rotation_deg, translation_m=translation_m)
        read = read_face_sizes(face_sizes)
        self.found = [
            (entry, match_face_ref(ref, geometry.faces, geometry.bbox_m, **self._placement))
            for entry, (ref, _) in zip(face_sizes, read, strict=True)
        ]
        self.faces = {face: size for (_, face), (_, size) in zip(self.found, read, strict=True) if face is not None}
        self.volumes = {name: size_override(size) for name, size in volume_sizes.items()}

    def shared(self, faces=(), volumes=()) -> CadSizeOverride | str | None:
        """The size ``faces`` and ``volumes`` share, else :data:`MIXED`; ``None``: all Global."""
        return shared([self.faces.get(f) for f in faces] + [self.volumes.get(v) for v in volumes])

    def written(self, size: CadSizeOverride | None, faces=(), volumes=()) -> dict:
        """The design source's size fields with ``size`` (``None``: Global) on each of ``faces`` and ``volumes``."""

        faces = set(faces)
        kept = [entry for entry, face in self.found if face not in faces]
        added = [] if size is None else [self.entry(face, size) for face in sorted(faces)]
        volume_sizes = {name: s.stored() for name, s in self.volumes.items() if name not in volumes}
        if size is not None:
            volume_sizes |= dict.fromkeys(volumes, size.stored())
        return {"face_sizes": kept + added, "volume_sizes": volume_sizes}

    def entry(self, face: int, size: CadSizeOverride) -> dict:
        """How ``face_sizes`` stores ``size`` on ``face``: with the face as a face group stores it."""
        found = next(f for f in self.geometry.faces if f.id == face)
        return {"face": asdict(face_ref(found, **self._placement))} | size.stored()

    def dropped(self) -> tuple[dict, str]:
        """The size fields without face sizes whose faces the geometry no longer has, nor volume sizes of bodies
        that are no air bodies, and a note saying so; ``({}, "")`` when there is none."""
        lost = sum(face is None for _, face in self.found)
        air = {body.name for body in self.geometry.bodies if body.kind == "air"}
        gone = sorted(set(self.volumes) - air)
        if not lost and not gone:
            return {}, ""
        notes = []
        if lost:
            notes.append(
                f"{lost} face size override{'' if lost == 1 else 's'} dropped: "
                "{} face{} no longer in the STEP file.".format(*(("its", " is") if lost == 1 else ("their", "s are")))
            )
        if gone:
            notes.append(f"Volume size of {', '.join(gone)} dropped: no air body.")
        changes = {
            "face_sizes": [entry for entry, face in self.found if face is not None],
            "volume_sizes": {name: size.stored() for name, size in self.volumes.items() if name in air},
        }
        return changes, " ".join(notes)


@dataclass(frozen=True)
class CadSizePlan:
    """The edge length (m) of each face and each air-body volume."""

    face_m: dict[int, float]
    volume_m: dict[str, float]


def size_plan(settings: CadImportSettings, geometry: CadGeometry, groups: Mapping[str, Mapping[str, Sequence[int]]]):
    """The sizes ``settings`` ask for on the read ``geometry`` and its surface ``groups`` (mesh id -> face group ->
    faces): on a face the most specific override wins (face, face group, finest body), else its target element size
    (:meth:`CadSizing.winning`); an air body's volume has its own, else its body's, else the FEM target element size.
    Raises ``ValueError`` for an override of an unknown body, face group, face or volume."""
    sizing = settings.sizing
    air = {body.name for body in geometry.bodies if body.kind == "air"}
    unknown = sorted(set(settings.body_sizes) - {b.name for b in geometry.bodies})
    if unknown:
        raise ValueError(f"Size override for unknown body '{unknown[0]}'.")
    group_of = {face: name for mesh_groups in groups.values() for name, faces in mesh_groups.items() for face in faces}
    unknown = sorted(set(settings.face_group_sizes) - set(group_of.values()))
    if unknown:
        raise ValueError(f"Size override for unknown face group '{unknown[0]}'.")
    known = {face.id for face in geometry.faces}
    own = {}
    for face, size in settings.face_sizes:
        if isinstance(face, CadFaceRef):
            face = match_face_ref(face, geometry.faces, geometry.bbox_m, **settings.placement)
            if face is None:
                continue  # gone after a re-import: the editor notes it, generation goes on
        elif face not in known:
            raise ValueError(f"Size override for unknown face {face}.")
        own[face] = size
    targets = sizing.face_targets(geometry)
    faces = {
        face.id: sizing.edge_m(
            sizing.winning(
                own.get(face.id),
                settings.face_group_sizes.get(group_of.get(face.id)),
                [settings.body_sizes.get(b) for b in face.bodies],
                targets[face.id],
            )
            or targets[face.id]
        )
        for face in geometry.faces
    }
    unknown = sorted(set(settings.volume_sizes) - air)
    if unknown:
        raise ValueError(f"Volume size for '{unknown[0]}', which is no air body.")
    volumes = {
        body.name: sizing.edge_m(settings.volume_sizes.get(body.name) or settings.body_sizes.get(body.name))
        for body in geometry.bodies
        if body.kind == "air"
    }
    return CadSizePlan(faces, volumes)


# -- older designs ---------------------------------------------------------------------------------------------------

LEGACY_SIZES = ("body_sizes", "face_group_sizes", "body_sizes_m", "face_group_sizes_m")  # design source keys


def legacy_sizes(source: Mapping) -> tuple[dict[str, CadSizeOverride], dict[str, CadSizeOverride]]:
    """``(body_sizes, face_group_sizes)`` of a design source. Designs saved before levels existed stored edge
    lengths (m) as ``body_sizes_m`` and ``face_group_sizes_m``; they are kept as edge lengths."""
    result = []
    for key in ("body_sizes", "face_group_sizes"):
        stored = {name: {"size_m": size} for name, size in source.get(f"{key}_m", {}).items()} | source.get(key, {})
        result.append({name: size_override(size) for name, size in stored.items()})
    return result[0], result[1]


def migrated(source: Mapping, groups: CadFaceGroups, sizes: CadSizes, sizing: CadSizing) -> dict | None:
    """``source`` with its body and face-group sizes turned into the face and volume sizes that mesh the same, or
    ``None`` when it has none (ADR 0001).

    Each face without its own size gets the one that won on it (face group, then the finest body), unless that is
    the face's own target element size; an air body's size becomes its volume size unless the volume has its own.
    ``groups`` (:class:`CadFaceGroups`) and ``sizes`` are the design's on the read geometry; ``sizing`` needs the
    maximum frequency, which compares levels with edge lengths.
    """
    if not any(source.get(key) for key in LEGACY_SIZES):
        return None
    bodies, group_sizes = legacy_sizes(source)
    targets = sizing.face_targets(sizes.geometry)
    added = []
    for face in sizes.geometry.faces:
        if face.id not in sizes.faces:
            group = group_sizes.get(groups.group_of(face.id).name)
            size = sizing.winning(None, group, [bodies.get(body) for body in face.bodies], targets[face.id])
            if size is not None:
                added.append(sizes.entry(face.id, size))
    air = {body.name for body in sizes.geometry.bodies if body.kind == "air"}
    volumes = {name: size.stored() for name, size in bodies.items() if name in air}
    return {k: v for k, v in source.items() if k not in LEGACY_SIZES} | {
        "face_sizes": [*source.get("face_sizes", []), *added],
        "volume_sizes": volumes | dict(source.get("volume_sizes", {})),
    }
