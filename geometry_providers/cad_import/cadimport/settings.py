"""Qt-free inputs shared by the CAD import design and meshing pipeline."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from blab.defaults import DEFAULT_SOUND_SPEED_M_PER_S

from .face_refs import ZERO, CadFaceRef
from .face_refs import placement as _placement
from .sizing import CadSizeOverride, CadSizing, size_override

SYMMETRY_LABELS = {"off": "Off", "x": "X", "xy": "XY"}


@dataclass(frozen=True)
class CadImportSettings:
    """Meshing inputs. Sizes in metres.

    The placement rotates by ``rotation_deg`` about the fixed X, Y and Z axes (in that order,
    about the origin) and translates by ``translation_m``.

    ``face_groups`` maps a group name to its faces: ``CadFaceRef`` (or mappings of its
    fields), or face ids (``CadFace.id``). Faces with a STEP name in no face group form a
    group of that name. ``symmetry`` is the project symmetry to check the geometry
    against; ``None`` accepts what is detected. ``closed_shells`` maps a closed shell name
    to ``"air"`` or ``"exterior"``; one without a choice gets its guessed kind (see
    :class:`CadClosedShell`). ``interface_names`` renames interfaces by
    their detected name; names of interfaces not detected are ignored.

    ``elements_per_wavelength`` is the FEM mesh fineness, sizing air-body volumes and faces;
    ``bem_elements_per_wavelength`` the BEM one, sizing exterior surface faces (``None``: the FEM
    one). A face on both takes the finer target element size.

    Size overrides (:class:`CadSizeOverride` or mappings of its fields) replace the face's target element
    size per body, per face group or, in ``face_sizes`` as ``(face, override)`` pairs, per face (a
    ``CadFaceRef`` that no longer matches a face is ignored). On a face the most specific one wins:
    face, face group, the finest of its bodies, then the face's target element size. ``volume_sizes`` sizes
    only the interior of an air body, never its faces; it wins over the body's override there.
    """

    fmax_hz: float
    sound_speed_m_per_s: float = DEFAULT_SOUND_SPEED_M_PER_S
    elements_per_wavelength: float = 6.0
    bem_elements_per_wavelength: float | None = None
    face_groups: Mapping[str, tuple[int | CadFaceRef, ...]] = field(default_factory=dict)
    body_sizes: Mapping[str, CadSizeOverride] = field(default_factory=dict)
    face_group_sizes: Mapping[str, CadSizeOverride] = field(default_factory=dict)
    face_sizes: tuple[tuple[int | CadFaceRef, CadSizeOverride], ...] = ()
    volume_sizes: Mapping[str, CadSizeOverride] = field(default_factory=dict)
    curvature_elements_per_circle: float = 12.0
    growth_rate: float = 1.3
    symmetry: str | None = None
    closed_shells: Mapping[str, str] = field(default_factory=dict)
    rotation_deg: tuple[float, float, float] = ZERO
    translation_m: tuple[float, float, float] = ZERO
    interface_names: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bem = self.bem_elements_per_wavelength
        for label, value in (
            ("Maximum frequency", self.fmax_hz),
            ("Sound speed", self.sound_speed_m_per_s),
            ("Elements per wavelength", self.elements_per_wavelength),
            ("BEM elements per wavelength", self.elements_per_wavelength if bem is None else bem),
            ("Curvature elements per circle", self.curvature_elements_per_circle),
        ):
            if not (isinstance(value, int | float) and math.isfinite(value) and value > 0):
                raise ValueError(f"{label} must be a positive number.")
        if not (math.isfinite(self.growth_rate) and self.growth_rate > 1):
            raise ValueError("Growth rate must be greater than 1.")
        object.__setattr__(self, "rotation_deg", tuple(self.rotation_deg))
        object.__setattr__(self, "translation_m", tuple(self.translation_m))
        _placement(**self.placement)
        groups = {
            name: tuple(CadFaceRef(**face) if isinstance(face, Mapping) else face for face in faces)
            for name, faces in self.face_groups.items()
        }
        object.__setattr__(self, "face_groups", groups)
        for name in ("body_sizes", "face_group_sizes", "volume_sizes"):
            object.__setattr__(self, name, {k: size_override(v) for k, v in getattr(self, name).items()})
        faces = tuple(
            (CadFaceRef(**face) if isinstance(face, Mapping) else face, size_override(size))
            for face, size in self.face_sizes
        )
        object.__setattr__(self, "face_sizes", faces)
        if self.symmetry not in (None, *SYMMETRY_LABELS):
            raise ValueError(f"Unknown symmetry '{self.symmetry}'.")

    @property
    def sizing(self) -> CadSizing:
        return CadSizing(
            self.fmax_hz, self.sound_speed_m_per_s, self.elements_per_wavelength, self.bem_elements_per_wavelength
        )

    @property
    def target_size_m(self) -> float:
        """The FEM target element size: sound speed / maximum frequency / elements per wavelength."""
        return self.sizing.target_size_m

    @property
    def placement(self) -> dict:
        """The placement as keyword arguments of :func:`face_ref`, :func:`match_face_ref` and the readers."""
        return dict(rotation_deg=self.rotation_deg, translation_m=self.translation_m)
