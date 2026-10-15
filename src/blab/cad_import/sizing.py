"""Mesh sizes of a CAD import design: the target element size, size overrides, their text, and which one wins.

Qt-free. The pipeline, the provider backend and the editor all size through it. A size override is a level (elements
per wavelength at the maximum frequency) or an edge length; ``None`` stands for no override (the target element
size, "Global"). The editor stores overrides only on faces and air-body volumes (ADR 0001).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

LEVELS = {"Coarse": 2, "Medium": 4, "Fine": 6}  # mesh fineness: elements per wavelength at the maximum frequency
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


@dataclass(frozen=True)
class CadSizing:
    """Edge lengths from the sound speed and the maximum frequency (``None`` while unknown) at a mesh fineness."""

    fmax_hz: float | None
    sound_speed_m_per_s: float = 343.0
    elements_per_wavelength: float = 6.0

    @property
    def fineness(self) -> CadSizeOverride:
        return CadSizeOverride(elements_per_wavelength=self.elements_per_wavelength)

    @property
    def target_size_m(self) -> float:
        """Sound speed / maximum frequency / elements per wavelength."""
        return self.edge_m(self.fineness)

    def edge_m(self, size: CadSizeOverride | None) -> float:
        """The edge length ``size`` asks for; ``None``: the target element size."""
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

    def winning(
        self, face: CadSizeOverride | None, group: CadSizeOverride | None, bodies: Sequence[CadSizeOverride | None]
    ) -> CadSizeOverride | None:
        """The override that sizes a face with its own ``face`` override, its face group's and its ``bodies``' (one
        per body, ``None`` for none): the most specific, else the finest body's; a body without one counts as the
        target element size, so ``None`` when that is finer."""
        if face is not None or group is not None:
            return face or group
        sized = [size for size in bodies if size is not None]
        finest = min(sized, key=self.edge_m, default=None)
        if finest is not None and (len(sized) == len(bodies) or self.edge_m(finest) < self.target_size_m):
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
        geometry,
        face_sizes: Sequence[Mapping],
        volume_sizes: Mapping[str, Mapping],
        *,
        rotation_deg=(0.0, 0.0, 0.0),
        translation_m=(0.0, 0.0, 0.0),
    ) -> None:
        from .pipeline import CadFaceRef, match_face_ref  # the pipeline sizes through this module

        self.geometry = geometry
        self._placement = dict(rotation_deg=rotation_deg, translation_m=translation_m)
        self.found = [
            (entry, match_face_ref(CadFaceRef(**entry["face"]), geometry.faces, geometry.bbox_m, **self._placement))
            for entry in face_sizes
        ]
        self.faces = {
            face: size_override({k: v for k, v in entry.items() if k != "face"})
            for entry, face in self.found
            if face is not None
        }
        self.volumes = {name: size_override(size) for name, size in volume_sizes.items()}

    def shared(self, faces=(), volumes=()) -> CadSizeOverride | str | None:
        """The size ``faces`` and ``volumes`` share, else :data:`MIXED`; ``None``: all Global."""
        return shared([self.faces.get(f) for f in faces] + [self.volumes.get(v) for v in volumes])

    def written(self, size: CadSizeOverride | None, faces=(), volumes=()) -> dict:
        """The design source's size fields with ``size`` (``None``: Global) on each of ``faces`` and ``volumes``."""
        from .pipeline import face_ref

        faces = set(faces)
        kept = [entry for entry, face in self.found if face not in faces]
        added = []
        if size is not None:
            by_id = {f.id: f for f in self.geometry.faces}
            for face in sorted(faces):
                added.append({"face": asdict(face_ref(by_id[face], **self._placement))} | size.stored())
        volume_sizes = {name: s.stored() for name, s in self.volumes.items() if name not in volumes}
        if size is not None:
            volume_sizes |= dict.fromkeys(volumes, size.stored())
        return {"face_sizes": kept + added, "volume_sizes": volume_sizes}

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
