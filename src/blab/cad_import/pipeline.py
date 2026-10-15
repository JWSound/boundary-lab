"""STEP -> classified, fragmented OpenCASCADE model -> one conformal mesh -> assembly files.

Solids are air bodies (bounded regions, one tetrahedral mesh per touching group);
surface bodies are exterior surface bodies (together one BEM surface mesh). Closed
shells of free faces are air bodies or exterior surface, as chosen or guessed. An air
face that fills an opening of the exterior, or coincides with a whole exterior
face, is an interface: the same face group name appears in both meshes.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import meshio
import numpy as np

WALLS = "walls"
EXTERIOR_MESH_ID = "exterior"
SYMMETRY_LABELS = {"off": "Off", "x": "X", "xy": "XY"}
ZERO = (0.0, 0.0, 0.0)  # no rotation, no translation
_GMSH_LOCK = threading.RLock()
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


def _size_override(size) -> CadSizeOverride:
    return size if isinstance(size, CadSizeOverride) else CadSizeOverride(**size)


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

    Size overrides (:class:`CadSizeOverride` or mappings of its fields) replace the target element
    size per body, per face group or, in ``face_sizes`` as ``(face, override)`` pairs, per face (a
    ``CadFaceRef`` that no longer matches a face is ignored). On a face the most specific one wins:
    face, face group, the finest of its bodies, then the target element size. ``volume_sizes`` sizes
    only the interior of an air body, never its faces; it wins over the body's override there.
    """

    fmax_hz: float
    sound_speed_m_per_s: float = 343.0
    elements_per_wavelength: float = 6.0
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
        for label, value in (
            ("Maximum frequency", self.fmax_hz),
            ("Sound speed", self.sound_speed_m_per_s),
            ("Elements per wavelength", self.elements_per_wavelength),
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
            object.__setattr__(self, name, {k: _size_override(v) for k, v in getattr(self, name).items()})
        faces = tuple(
            (CadFaceRef(**face) if isinstance(face, Mapping) else face, _size_override(size))
            for face, size in self.face_sizes
        )
        object.__setattr__(self, "face_sizes", faces)
        if self.symmetry not in (None, *SYMMETRY_LABELS):
            raise ValueError(f"Unknown symmetry '{self.symmetry}'.")

    @property
    def target_size_m(self) -> float:
        """Sound speed / maximum frequency / elements per wavelength."""
        return self.sound_speed_m_per_s / self.fmax_hz / self.elements_per_wavelength

    def edge_m(self, size: CadSizeOverride) -> float:
        """The edge length a size override asks for."""
        if size.size_m is not None:
            return size.size_m
        return self.sound_speed_m_per_s / self.fmax_hz / size.elements_per_wavelength

    @property
    def placement(self) -> dict:
        """The placement as keyword arguments of :func:`face_ref`, :func:`match_face_ref` and the readers."""
        return dict(rotation_deg=self.rotation_deg, translation_m=self.translation_m)


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


@dataclass(frozen=True)
class CadBody:
    name: str
    kind: str  # "air" | "exterior"
    face_ids: tuple[int, ...]


@dataclass(frozen=True)
class CadRegion:
    mesh_id: str
    kind: str  # "bounded_air" | "unbounded_air"
    bodies: tuple[str, ...]


@dataclass(frozen=True)
class CadInterface:
    """An interface between ``mesh_id`` and the exterior; ``name`` is the face group on both sides."""

    name: str
    mesh_id: str
    face_ids: tuple[int, ...]


@dataclass(frozen=True)
class CadClosedShell:
    """Closed connected free faces: an air body or an exterior surface body, named after the shell.

    Without a choice it is guessed air, unless the file has no other exterior surface: then the
    shell with the largest bounding box is exterior and the others air.
    """

    name: str
    role: str  # "air" | "exterior"
    face_ids: tuple[int, ...]


@dataclass(frozen=True)
class CadGeometry:
    bodies: tuple[CadBody, ...]
    faces: tuple[CadFace, ...]
    regions: tuple[CadRegion, ...]
    interfaces: tuple[CadInterface, ...]
    bbox_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    symmetry: str  # suggested project symmetry: "off" | "x" | "xy"
    warnings: tuple[str, ...]
    closed_shells: tuple[CadClosedShell, ...] = ()


@dataclass(frozen=True)
class CadMesh:
    id: str
    purpose: str  # "fem_volume" | "bem_surface"
    path: Path
    volume_groups: tuple[str, ...]
    surface_groups: tuple[str, ...]


@dataclass(frozen=True)
class CadImportResult:
    geometry: CadGeometry
    meshes: tuple[CadMesh, ...]
    target_size_m: float


def inspect_step(
    step_path: str | Path,
    *,
    rotation_deg: Sequence[float] = ZERO,
    translation_m: Sequence[float] = ZERO,
    closed_shells: Mapping[str, str] | None = None,
) -> CadGeometry:
    """Read, place, classify and fragment a STEP file without meshing it."""
    placement = _placement(rotation_deg, translation_m)
    with _gmsh_session() as gmsh:
        return _analyse(gmsh, Path(step_path), placement, closed_shells or {}).geometry


def face_ref(
    face: CadFace,
    *,
    rotation_deg: Sequence[float] = ZERO,
    translation_m: Sequence[float] = ZERO,
) -> CadFaceRef:
    """How a face group stores ``face`` of a geometry read with that placement: in CAD axes."""
    matrix, offset = _placement(rotation_deg, translation_m)
    return CadFaceRef(
        face.step_name,
        tuple(float(v) for v in matrix.T @ (np.asarray(face.centre, dtype=float) - offset)),
        float(face.area),
        tuple(float(v) for v in matrix.T @ np.asarray(face.normal, dtype=float)),
        face.surface_type,
    )


@dataclass(frozen=True)
class _Openings:
    """What finding the openings into the air bodies needs; a placement moves only ``edge_points``.

    ``direct`` are the exterior faces lying on air faces: openings whatever the placement.
    ``cover_warnings`` name the exterior faces that cover air faces instead.
    """

    face_edges: Mapping[int, tuple[int, ...]]  # boundary curves of the outer air and exterior faces
    edges: tuple[int, ...]
    edge_points: np.ndarray  # nine sampled points (m) per edge of ``edges``
    air_body: Mapping[int, str]  # outer air face -> the air body it bounds
    exterior: tuple[int, ...]
    direct: tuple[int, ...]
    cover_warnings: tuple[str, ...]


@dataclass(frozen=True)
class CadPreview:
    """A coarse display tessellation: ``triangles`` index ``points`` (m); ``face_ids`` per triangle.

    ``openings`` lets :func:`place_preview` redo the placement-dependent interfaces, symmetry
    suggestion and warnings. ``rotation_deg`` and ``translation_m`` are the placement it shows.
    """

    geometry: CadGeometry
    points: np.ndarray
    triangles: np.ndarray
    face_ids: np.ndarray
    openings: _Openings
    rotation_deg: tuple[float, float, float] = ZERO
    translation_m: tuple[float, float, float] = ZERO


def open_step(
    step_path: str | Path,
    *,
    rotation_deg: Sequence[float] = ZERO,
    translation_m: Sequence[float] = ZERO,
    closed_shells: Mapping[str, str] | None = None,
) -> CadPreview:
    """Read a STEP file for display with that placement."""
    placement = _placement(rotation_deg, translation_m)
    with _gmsh_session() as gmsh:
        model = _analyse(gmsh, Path(step_path), placement, closed_shells or {})
        geometry = model.geometry
        return CadPreview(
            geometry,
            *_tessellate(gmsh, [face.id for face in geometry.faces], geometry.bbox_m),
            openings=model.openings,
            **_shown(rotation_deg, translation_m),
        )


def place_preview(
    preview: CadPreview,
    *,
    rotation_deg: Sequence[float] = ZERO,
    translation_m: Sequence[float] = ZERO,
) -> CadPreview:
    """``preview`` moved to that placement without reading the STEP file again.

    The result reports what reading the file with that placement does, face ids aside.
    """
    new_matrix, new_offset = _placement(rotation_deg, translation_m)
    old_matrix, old_offset = _placement(preview.rotation_deg, preview.translation_m)
    matrix = new_matrix @ old_matrix.T
    move = (matrix, new_offset - matrix @ old_offset)
    points = _place(move, preview.points)
    geometry = preview.geometry
    openings = replace(preview.openings, edge_points=_place(move, preview.openings.edge_points))
    faces = tuple(
        replace(face, centre=_moved(matrix, face.centre, move[1]), normal=_moved(matrix, face.normal))
        for face in geometry.faces
    )
    found = _find_openings(
        openings,
        {face.id: face for face in faces},
        [b.name for b in geometry.bodies if b.kind == "air"],
        {b: r.mesh_id for r in geometry.regions for b in r.bodies},
        _plane_tolerance(*geometry.bbox_m),
    )
    geometry = replace(
        geometry,
        faces=faces,
        interfaces=found.interfaces,
        # From the faces' points: a turned box around them would grow at angles off the axes.
        bbox_m=(tuple(points.min(axis=0).tolist()), tuple(points.max(axis=0).tolist())),
        symmetry=found.symmetry,
        warnings=openings.cover_warnings + found.warnings,
    )
    return replace(preview, geometry=geometry, points=points, openings=openings, **_shown(rotation_deg, translation_m))


def preview_to_json(preview: CadPreview) -> dict:
    """``preview`` as JSON data; :func:`preview_from_json` reads it back."""
    data = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(preview).items() if k != "openings"}
    openings = preview.openings
    data["openings"] = dict(
        face_edges=[[f, list(e)] for f, e in openings.face_edges.items()],
        edges=list(openings.edges),
        edge_points=openings.edge_points.tolist(),
        air_body=[[f, b] for f, b in openings.air_body.items()],
        exterior=list(openings.exterior),
        direct=list(openings.direct),
        cover_warnings=list(openings.cover_warnings),
    )
    return data


def result_to_json(result: CadImportResult) -> dict:
    """``result`` as JSON data; :func:`result_from_json` reads it back."""
    data = asdict(result)
    for mesh in data["meshes"]:
        mesh["path"] = str(mesh["path"])
    return data


def result_from_json(data: Mapping) -> CadImportResult:
    """The meshing result :func:`result_to_json` wrote as ``data``."""
    meshes = tuple(
        CadMesh(m["id"], m["purpose"], Path(m["path"]), tuple(m["volume_groups"]), tuple(m["surface_groups"]))
        for m in data["meshes"]
    )
    return CadImportResult(_geometry_from_json(data["geometry"]), meshes, data["target_size_m"])


def preview_from_json(data: Mapping) -> CadPreview:
    """The preview :func:`preview_to_json` wrote as ``data``."""
    geometry = _geometry_from_json(data["geometry"])
    o = data["openings"]
    openings = _Openings(
        {f: tuple(e) for f, e in o["face_edges"]},
        tuple(o["edges"]),
        np.asarray(o["edge_points"], dtype=float).reshape(-1, 9, 3),
        dict(o["air_body"]),
        tuple(o["exterior"]),
        tuple(o["direct"]),
        tuple(o["cover_warnings"]),
    )
    return CadPreview(
        geometry,
        np.asarray(data["points"], dtype=float).reshape(-1, 3),
        np.asarray(data["triangles"], dtype=np.int64).reshape(-1, 3),
        np.asarray(data["face_ids"], dtype=np.int64),
        openings,
        tuple(data["rotation_deg"]),
        tuple(data["translation_m"]),
    )


def _geometry_from_json(g: Mapping) -> CadGeometry:
    return CadGeometry(
        tuple(CadBody(b["name"], b["kind"], tuple(b["face_ids"])) for b in g["bodies"]),
        tuple(
            CadFace(
                f["id"],
                tuple(f["bodies"]),
                f["step_name"],
                tuple(f["centre"]),
                f["area"],
                tuple(f["normal"]),
                f["surface_type"],
            )
            for f in g["faces"]
        ),
        tuple(CadRegion(r["mesh_id"], r["kind"], tuple(r["bodies"])) for r in g["regions"]),
        tuple(CadInterface(i["name"], i["mesh_id"], tuple(i["face_ids"])) for i in g["interfaces"]),
        (tuple(g["bbox_m"][0]), tuple(g["bbox_m"][1])),
        g["symmetry"],
        tuple(g["warnings"]),
        tuple(CadClosedShell(s["name"], s["role"], tuple(s["face_ids"])) for s in g["closed_shells"]),
    )


def _tessellate(gmsh, faces, bbox_m) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coarse display triangles of ``faces``: points, triangles, face id per triangle."""
    gmsh.option.setNumber("Mesh.MeshSizeMax", math.dist(*bbox_m) / 30)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    gmsh.model.mesh.generate(2)
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    index = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
    index[node_tags.astype(np.int64)] = np.arange(len(node_tags))
    triangles, face_ids = [], []
    for face in faces:
        types, _, nodes = gmsh.model.mesh.getElements(2, face)
        for kind, data in zip(types, nodes, strict=True):
            if kind == 2:
                block = index[np.asarray(data, dtype=np.int64)].reshape(-1, 3)
                triangles.append(block)
                face_ids.append(np.full(len(block), face))
    return (
        np.asarray(coords).reshape(-1, 3),
        np.concatenate(triangles) if triangles else np.zeros((0, 3), dtype=np.int64),
        np.concatenate(face_ids) if face_ids else np.zeros(0, dtype=np.int64),
    )


def _shown(rotation_deg, translation_m) -> dict:
    """The placement fields of a preview showing that placement."""
    return dict(rotation_deg=tuple(rotation_deg), translation_m=tuple(translation_m))


def _moved(matrix, vector, offset=ZERO) -> tuple[float, float, float]:
    return tuple(float(v) for v in matrix @ np.asarray(vector, dtype=float) + offset)


def _place(placement, points) -> np.ndarray:
    """``points`` (any shape ending in 3) moved by ``(matrix, offset)``."""
    matrix, offset = placement
    return points @ matrix.T + offset


def mesh_step(step_path: str | Path, settings: CadImportSettings, output_dir: str | Path) -> CadImportResult:
    """Mesh a STEP file into ``output_dir`` as ``<mesh id>.msh`` files (metres)."""
    output_dir = Path(output_dir)
    with _gmsh_session() as gmsh:
        model = _analyse(gmsh, Path(step_path), _placement(**settings.placement), settings.closed_shells)
        model.rename_interfaces(settings.interface_names)
        geometry = model.geometry
        if settings.symmetry is not None and settings.symmetry != geometry.symmetry:
            raise ValueError(
                f"Project symmetry {SYMMETRY_LABELS[settings.symmetry]} does not match the geometry, "
                f"which suggests symmetry {SYMMETRY_LABELS[geometry.symmetry]}."
            )
        from .design import CadFaceGroups  # it builds on this module

        face_groups = CadFaceGroups(geometry, settings.face_groups, **settings.placement)
        face_groups.check()
        groups = face_groups.surface_groups(model.mesh_faces)
        _apply_sizes(gmsh, model, settings, groups)
        gmsh.model.mesh.generate(3)
        output_dir.mkdir(parents=True, exist_ok=True)
        meshes = []
        for region in geometry.regions:
            path = output_dir / f"{region.mesh_id}.msh"
            if region.kind == "bounded_air":
                _write_volume(gmsh, path, groups[region.mesh_id], {b: model.volumes[b] for b in region.bodies})
                meshes.append(CadMesh(region.mesh_id, "fem_volume", path, region.bodies, tuple(groups[region.mesh_id])))
            else:
                _write_surface(gmsh, path, groups[region.mesh_id])
                meshes.append(CadMesh(region.mesh_id, "bem_surface", path, (), tuple(groups[region.mesh_id])))
        return CadImportResult(geometry, tuple(meshes), settings.target_size_m)


# --- analysis -----------------------------------------------------------------------------------


@dataclass
class _Model:
    geometry: CadGeometry
    volumes: dict[str, list[int]]  # air body -> volume tags
    mesh_faces: dict[str, list[int]]  # mesh id -> faces on its boundary
    interface_of: dict[int, str]  # face -> interface name
    face_bodies: dict[int, tuple[str, ...]]
    openings: _Openings

    def rename_interfaces(self, names: Mapping[str, str]) -> None:
        """Rename the interfaces by detected name; names of interfaces not detected are ignored."""
        renamed = {i.name: names.get(i.name, i.name).strip() for i in self.geometry.interfaces}
        counts = Counter(renamed.values())
        for old, new in renamed.items():
            if not new or new == WALLS or counts[new] > 1:
                raise ValueError(f"Interface '{old}' cannot be named '{new}'.")
        self.geometry = replace(
            self.geometry, interfaces=tuple(replace(i, name=renamed[i.name]) for i in self.geometry.interfaces)
        )
        self.interface_of = {face: renamed[name] for face, name in self.interface_of.items()}


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
    matrix, offset = _placement(rotation_deg, translation_m)
    centre, normal = _moved(matrix, ref.centre, offset), _moved(matrix, ref.normal)
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


@dataclass
class _Step:
    """A STEP file as imported: solids, free faces and the closed shells among them, with body names."""

    solids: list[int]
    solid_names: list[str]
    sheets: list[int]
    shells: list[list[int]]
    shell_names: list[str]


def _read(gmsh, path: Path) -> _Step:
    occ = gmsh.model.occ
    if not path.is_file():
        raise ValueError(f"STEP file not found: {path}")
    gmsh.option.setString("Geometry.OCCTargetUnit", "M")
    gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
    try:
        occ.importShapes(str(path), highestDimOnly=False)
    except Exception as exc:
        raise ValueError(f"Could not read STEP file {path}: {exc}") from exc
    occ.synchronize()
    solids = [tag for _, tag in gmsh.model.getEntities(3)]
    sheets = [tag for _, tag in gmsh.model.getEntities(2) if len(gmsh.model.getAdjacencies(2, tag)[0]) == 0]
    if not solids and not sheets:
        raise ValueError(f"STEP file {path} contains no solids or surfaces.")
    # Closed shells: free faces that close up among themselves (SolidWorks writes solids of
    # parts mixing solid and surface bodies this way). The user decides what each one is.
    sheet_edges = {f: _edges(gmsh, f) for f in sheets}
    shells = [
        faces
        for faces in _components(sheets, sheet_edges)
        if set(Counter(e for f in faces for e in sheet_edges[f]).values()) == {2}
    ]
    names = _unique(
        [_label(gmsh, 3, t) or f"Air body {i + 1}" for i, t in enumerate(solids)]
        + [_label(gmsh, 2, faces[0]) or f"Closed shell {k}" for k, faces in enumerate(shells, start=1)]
    )
    return _Step(solids, names[: len(solids)], sheets, shells, names[len(solids) :])


def _analyse(gmsh, path: Path, placement: tuple[np.ndarray, np.ndarray], closed_shells: Mapping[str, str]) -> _Model:
    """Classify and fragment ``path`` placed by ``(matrix, offset)``."""
    occ = gmsh.model.occ
    for name, role in closed_shells.items():
        if role not in ("air", "exterior"):
            raise ValueError(f"Closed shell '{name}' must be 'air' or 'exterior', not '{role}'.")
    step = _read(gmsh, path)
    solids, sheets, shells, shell_names = list(step.solids), list(step.sheets), step.shells, step.shell_names
    solid_names = list(step.solid_names)
    unknown = sorted(set(closed_shells) - set(shell_names))
    if unknown:
        raise ValueError(f"Unknown closed shell '{unknown[0]}'.")
    roles = _guessed_roles(gmsh, step) | dict(closed_shells)
    shell_of = {}
    for name, faces in zip(shell_names, shells, strict=True):
        if roles[name] == "air":
            solids.append(occ.addVolume([occ.addSurfaceLoop(faces)]))
            solid_names.append(name)
            sheets = [f for f in sheets if f not in faces]
        else:
            shell_of.update(dict.fromkeys(faces, name))
    sheet_names = [shell_of.get(t) or _label(gmsh, 2, t) for t in sheets]
    inputs = [(3, t) for t in solids] + [(2, t) for t in sheets]
    # Rigid motions keep exact surface types (a general affine transform makes BSplines).
    matrix, offset = placement
    angle = math.acos(np.clip((np.trace(matrix) - 1) / 2, -1, 1))
    if angle > 1e-9:
        if math.pi - angle > 1e-9:
            axis = [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
        else:
            axis = (matrix + np.eye(3))[:, int(np.argmax(np.diag(matrix)))]
        occ.rotate(inputs, 0, 0, 0, *axis, angle)
    if offset.any():
        occ.translate(inputs, *offset)
    _, children = occ.fragment(inputs, [])
    occ.synchronize()

    volumes: dict[str, list[int]] = {}
    body_of_volume = {}
    for name, kids in zip(solid_names, children, strict=False):
        volumes[name] = [t for d, t in kids if d == 3]
        for volume in volumes[name]:
            if volume in body_of_volume:
                raise ValueError(f"Air bodies '{body_of_volume[volume]}' and '{name}' overlap.")
            body_of_volume[volume] = name
    sheet_children = [[t for d, t in kids if d == 2] for kids in children[len(solids) :]]
    exterior = sorted({f for kids in sheet_children for f in kids})

    edges = {f: _edges(gmsh, f) for _, f in gmsh.model.getEntities(2)}
    exterior_bodies = _exterior_bodies(sheet_names, sheet_children, edges)
    face_volumes = {f: [v for v in gmsh.model.getAdjacencies(2, f)[0] if v in body_of_volume] for f in edges}
    outer_air = sorted(f for f, vols in face_volumes.items() if len(vols) == 1)

    bbox = gmsh.model.getBoundingBox(-1, -1)
    tol = _plane_tolerance(bbox[:3], bbox[3:])

    # An exterior face lying on an air face opens into it, unless it is part of a larger exterior face.
    split = {f for kids in sheet_children if len(kids) > 1 for f in kids}
    exterior_name = {f: name for name, faces in exterior_bodies.items() for f in faces}
    cover_warnings, direct = [], []
    for f in sorted(set(exterior).intersection(outer_air)):
        if f in split:
            cover_warnings.append(
                f"Exterior surface '{exterior_name[f]}' covers a face of air body "
                f"'{body_of_volume[face_volumes[f][0]]}' (face {f}) instead of opening into it; "
                "no interface was created there."
            )
        else:
            direct.append(f)

    # Regions: touching air bodies form one mesh.
    parent = {name: name for name in volumes}

    def root(name):
        while parent[name] != name:
            name = parent[name]
        return name

    for vols in face_volumes.values():
        if len(vols) == 2:
            parent[root(body_of_volume[vols[0]])] = root(body_of_volume[vols[1]])
    clusters = defaultdict(list)
    for name in volumes:
        clusters[root(name)].append(name)
    regions = []
    mesh_faces = {}
    used_ids = {EXTERIOR_MESH_ID}
    for bodies in clusters.values():
        mesh_id = _unique_id(_slug(bodies[0]), used_ids)
        regions.append(CadRegion(mesh_id, "bounded_air", tuple(bodies)))
        mesh_faces[mesh_id] = [f for f in outer_air if body_of_volume[face_volumes[f][0]] in bodies]
    if exterior:
        regions.append(CadRegion(EXTERIOR_MESH_ID, "unbounded_air", tuple(exterior_bodies)))
    mesh_of_body = {b: r.mesh_id for r in regions for b in r.bodies}

    step_names = _match_step_names(gmsh, path, placement, sorted(edges), tol)
    face_bodies = {}
    faces = []
    for f in sorted(set(outer_air) | set(exterior)):
        owners = tuple(body_of_volume[v] for v in face_volumes[f])
        face_bodies[f] = owners + ((exterior_name[f],) if f in exterior_name else ())
        faces.append(_face(gmsh, f, face_bodies[f], step_names.get(f)))
    curves = sorted({e for f in face_bodies for e in edges[f]})
    openings = _Openings(
        {f: tuple(sorted(edges[f])) for f in face_bodies},
        tuple(curves),
        _sample_edges(gmsh, curves),
        {f: body_of_volume[face_volumes[f][0]] for f in outer_air},
        tuple(exterior),
        tuple(direct),
        tuple(cover_warnings),
    )
    found = _find_openings(openings, {face.id: face for face in faces}, list(volumes), mesh_of_body, tol)
    if exterior:
        mesh_faces[EXTERIOR_MESH_ID] = found.shell

    bodies = [CadBody(n, "air", tuple(f for f in face_bodies if n in face_bodies[f])) for n in volumes]
    bodies += [CadBody(n, "exterior", tuple(fs)) for n, fs in exterior_bodies.items()]
    body_faces = {b.name: b.face_ids for b in bodies}
    closed = []
    for name, shell_faces in zip(shell_names, shells, strict=True):
        if roles[name] == "air":
            closed.append(CadClosedShell(name, "air", body_faces[name]))
        else:
            kids = {f for s, k in zip(sheets, sheet_children, strict=True) if s in shell_faces for f in k}
            closed.append(CadClosedShell(name, "exterior", tuple(sorted(kids))))
    geometry = CadGeometry(
        tuple(bodies),
        tuple(faces),
        tuple(regions),
        found.interfaces,
        (tuple(bbox[:3]), tuple(bbox[3:])),
        found.symmetry,
        openings.cover_warnings + found.warnings,
        tuple(closed),
    )
    return _Model(geometry, volumes, mesh_faces, found.interface_of, face_bodies, openings)


def _guessed_roles(gmsh, step: _Step) -> dict[str, str]:
    """Closed shell name -> its guessed kind: air, unless the file has no other exterior surface;
    then the shell with the largest bounding box is exterior."""
    roles = dict.fromkeys(step.shell_names, "air")
    in_shells = {f for faces in step.shells for f in faces}
    if step.shells and set(step.sheets) <= in_shells:

        def extent(faces):
            boxes = np.array([gmsh.model.getBoundingBox(2, f) for f in faces])
            return math.dist(boxes[:, :3].min(axis=0), boxes[:, 3:].max(axis=0))

        largest = max(range(len(step.shells)), key=lambda k: extent(step.shells[k]))
        roles[step.shell_names[largest]] = "exterior"
    return roles


@dataclass(frozen=True)
class _Found:
    interfaces: tuple[CadInterface, ...]
    interface_of: dict[int, str]  # face -> interface name
    shell: list[int]  # the exterior mesh's faces: the exterior and the faces filling its openings
    symmetry: str
    warnings: tuple[str, ...]


def _find_openings(openings: _Openings, faces: Mapping[int, CadFace], air_bodies, mesh_of_body, tol) -> _Found:
    """The interfaces, the exterior mesh's faces, the suggested symmetry and the edge warnings.

    An interface is a connected group of an air body's faces opening into the exterior: faces whose
    edges close holes in the exterior surface (or lie on X=0 or Y=0), and exterior faces on air faces.
    """
    face_edges = {f: set(e) for f, e in openings.face_edges.items()}
    points = dict(zip(openings.edges, openings.edge_points, strict=True))
    exterior = set(openings.exterior)
    counts = Counter(e for f in exterior for e in face_edges[f])
    free = {e for e, n in counts.items() if n == 1}

    def on_plane(edge):
        return any(np.all(np.abs(points[edge][:, a]) <= tol) for a in (0, 1))

    fillers = {
        f
        for f in openings.air_body
        if f not in exterior and face_edges[f] & free and not _on_symmetry_plane(faces[f], tol)
    }
    while True:
        shared = Counter(e for f in fillers for e in face_edges[f])
        keep = {f for f in fillers if all(e in free or shared[e] > 1 or on_plane(e) for e in face_edges[f])}
        if keep == fillers:
            break
        fillers = keep

    shell = sorted(exterior | fillers)
    counts = Counter(e for f in shell for e in face_edges[f])
    open_edges = np.array([points[e] for e, n in sorted(counts.items()) if n == 1]).reshape(-1, 9, 3)
    overlap_edges = np.array([points[e] for e, n in sorted(counts.items()) if n > 2]).reshape(-1, 9, 3)
    symmetry, warnings = _edge_findings(open_edges, overlap_edges, tol)

    interfaces, interface_of = [], {}
    by_body = defaultdict(list)
    for f in sorted(fillers | set(openings.direct)):
        by_body[openings.air_body[f]].append(f)
    for body in air_bodies:
        for k, group in enumerate(_components(by_body[body], face_edges), start=1):
            name = f"Interface {k} {body}"
            interfaces.append(CadInterface(name, mesh_of_body[body], tuple(group)))
            interface_of.update(dict.fromkeys(group, name))
    return _Found(tuple(interfaces), interface_of, shell, symmetry, warnings)


def _on_symmetry_plane(face: CadFace, tol) -> bool:
    return face.surface_type == "Plane" and any(
        abs(face.centre[a]) <= tol and abs(abs(face.normal[a]) - 1) < 1e-9 for a in (0, 1)
    )


def _sample_edges(gmsh, edges) -> np.ndarray:
    """Nine points along each of ``edges``: (edge, point, xyz)."""
    samples = []
    for edge in edges:
        low, high = gmsh.model.getParametrizationBounds(1, edge)
        samples.append(gmsh.model.getValue(1, edge, np.linspace(low[0], high[0], 9).tolist()))
    return np.reshape(samples, (-1, 9, 3))


def _plane_tolerance(low, high) -> float:
    return max(1e-9, 1e-6 * math.dist(low, high))


def _edge_findings(open_edges, overlap_edges, tol) -> tuple[str, tuple[str, ...]]:
    """Suggested symmetry and warnings from the exterior's open and overlapping edges (sampled points, m)."""
    on_x = [bool(np.all(np.abs(e[:, 0]) <= tol)) for e in open_edges]
    on_y = [not x and bool(np.all(np.abs(e[:, 1]) <= tol)) for e, x in zip(open_edges, on_x, strict=True)]
    symmetry = ("xy" if any(on_y) else "x") if any(on_x) else "off"
    # Y=0 alone is no supported symmetry, so those edges are gaps too.
    gaps = [e for e, x, y in zip(open_edges, on_x, on_y, strict=True) if not (x or (y and any(on_x)))]
    warnings = []
    if gaps:
        warnings.append(f"Exterior surface is not closed: {len(gaps)} open edge(s) near {_describe(gaps)}.")
    if len(overlap_edges):
        warnings.append(
            f"Exterior surface bodies overlap: {len(overlap_edges)} edge(s) are shared by more than two faces "
            f"near {_describe(overlap_edges)}."
        )
    return symmetry, tuple(warnings)


def _describe(edges, limit=5) -> str:
    text = ", ".join("({:.1f}, {:.1f}, {:.1f}) mm".format(*(e[4] * 1000)) for e in edges[:limit])
    return text + (", ..." if len(edges) > limit else "")


def _exterior_bodies(names, children, edges) -> dict[str, list[int]]:
    """Group exterior faces by STEP body name; unnamed faces by connectivity."""
    named = defaultdict(set)
    unnamed = set()
    for name, kids in zip(names, children, strict=True):
        (named[name] if name else unnamed).update(kids)
    unnamed -= set().union(*named.values()) if named else set()
    bodies = {name: sorted(faces) for name, faces in named.items()}
    for k, faces in enumerate(_components(sorted(unnamed), edges), start=1):
        bodies[_unique_id(f"Exterior surface {k}", set(bodies), sep=" ")] = faces
    return bodies


def _edges(gmsh, face) -> set[int]:
    """Boundary curves of a face, without seams (which bound the face on both sides)."""
    tags = [abs(t) for _, t in gmsh.model.getBoundary([(2, face)], combined=False, oriented=True)]
    return {t for t in tags if tags.count(t) == 1}


def _components(faces, edges) -> list[list[int]]:
    """Edge-connected components of ``faces``, each sorted, ordered by smallest face."""
    by_edge = defaultdict(list)
    for face in faces:
        for edge in edges[face]:
            by_edge[edge].append(face)
    seen = set()
    result = []
    for seed in sorted(faces):
        if seed in seen:
            continue
        seen.add(seed)
        component, stack = [seed], [seed]
        while stack:
            for edge in edges[stack.pop()]:
                for other in by_edge[edge]:
                    if other not in seen:
                        seen.add(other)
                        component.append(other)
                        stack.append(other)
        result.append(sorted(component))
    return result


def _face(gmsh, face, bodies, step_name) -> CadFace:
    centre = tuple(gmsh.model.occ.getCenterOfMass(2, face))
    return CadFace(
        face,
        bodies,
        step_name,
        centre,
        gmsh.model.occ.getMass(2, face),
        _normal(gmsh, face),
        gmsh.model.getType(2, face),
    )


def _normal(gmsh, face) -> tuple[float, float, float]:
    """Unit normal at the face point closest to its centre of mass."""
    _, param = gmsh.model.getClosestPoint(2, face, gmsh.model.occ.getCenterOfMass(2, face))
    return tuple(float(v) for v in gmsh.model.getNormal(face, param))


def _label(gmsh, dim, tag) -> str:
    """Body name from an OCC import label ``Shapes/<assembly>/<n>/<product>``."""
    name = gmsh.model.getEntityName(dim, tag)
    return name.rsplit("/", 1)[-1].strip() if name.startswith("Shapes/") else ""


def _unique(names) -> list[str]:
    used = set()
    return [_unique_id(name, used, sep=" ") for name in names]


def _unique_id(name, used, sep="_") -> str:
    candidate, k = name, 1
    while candidate in used:
        k += 1
        candidate = f"{name}{sep}{k}"
    used.add(candidate)
    return candidate


def _slug(name) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug if slug[:1].isalpha() else f"air_{slug}".rstrip("_")


def _placement(rotation_deg=ZERO, translation_m=ZERO) -> tuple[np.ndarray, np.ndarray]:
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


# --- STEP face names ------------------------------------------------------------------------------

_SI_PREFIX = {"$": 1.0, ".MILLI.": 1e-3, ".CENTI.": 1e-2, ".DECI.": 0.1, ".KILO.": 1e3, ".MICRO.": 1e-6}
_CONVERSION_UNITS = {"INCH": 0.0254, "FOOT": 0.3048}


def _match_step_names(gmsh, path, placement, faces, tol) -> dict[int, str]:
    """Attach ADVANCED_FACE names to the faces whose vertices they share.

    Gmsh does not import face names, so they are read from the STEP text. Faces
    split by fragmentation no longer match and stay unnamed.
    """
    named = _step_face_vertices(path)
    if not named:
        return {}
    vertices = {}
    for f in faces:
        points = gmsh.model.getBoundary([(2, f)], combined=False, oriented=False, recursive=True)
        vertices[f] = np.array([gmsh.model.getValue(0, p, []) for _, p in set(points)]).reshape(-1, 3)
    result = {}
    for name, points in named:
        points = _place(placement, points)
        for f, own in vertices.items():
            if len(own) == len(points) and all(np.min(np.linalg.norm(own - p, axis=1)) <= tol for p in points):
                result[f] = name
    return result


def _step_face_vertices(path) -> list[tuple[str, np.ndarray]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not re.search(r"ADVANCED_FACE\s*\(\s*'[^']", text):
        return []
    records = dict(re.findall(r"#(\d+)\s*=\s*([A-Z_0-9]+\s*\(.*?\))\s*;", text, flags=re.S))
    conversion = re.search(r"CONVERSION_BASED_UNIT\s*\(\s*'(\w+)'", text)
    if conversion and conversion.group(1).upper() in _CONVERSION_UNITS:
        scale = _CONVERSION_UNITS[conversion.group(1).upper()]
    else:
        prefix = re.search(r"SI_UNIT\s*\(\s*(\$|\.\w+\.)\s*,\s*\.METRE\.\s*\)", text)
        scale = _SI_PREFIX.get(prefix.group(1), 1.0) if prefix else 1.0
    walk = ("FACE_", "EDGE_LOOP", "ORIENTED_EDGE", "EDGE_CURVE", "VERTEX_LOOP")

    def vertex_points(key, seen):
        body = records.get(key, "")
        if body.startswith("VERTEX_POINT"):
            return {re.findall(r"#(\d+)", body)[0]}
        if not body.startswith(walk):
            return set()
        found = set()
        for ref in re.findall(r"#(\d+)", body):
            if ref not in seen:
                seen.add(ref)
                found |= vertex_points(ref, seen)
        return found

    result = []
    for key, body in records.items():
        match = re.match(r"ADVANCED_FACE\s*\(\s*'((?:[^']|'')+)'\s*,\s*\(([^)]*)\)", body)
        if not match or match.group(1) == "NONE":  # SolidWorks' placeholder for unnamed faces
            continue
        refs = set().union(*(vertex_points(b, set()) for b in re.findall(r"#(\d+)", match.group(2))))
        coords = [re.search(r"\(([^()]*)\)\s*\)$", records[r]).group(1).split(",") for r in refs]
        points = np.array([[float(v) for v in c] for c in coords]).reshape(-1, 3) * scale
        if len(points):
            result.append((match.group(1).replace("''", "'"), points))
    return result


# --- meshing --------------------------------------------------------------------------------------


def _apply_sizes(gmsh, model: _Model, settings: CadImportSettings, groups) -> None:
    """Target size everywhere, overrides per body / face group / face (the most specific wins) and air-body
    interiors, graded by the growth rate."""
    target = settings.target_size_m
    geometry = model.geometry
    unknown = sorted(set(settings.body_sizes) - {b.name for b in geometry.bodies})
    if unknown:
        raise ValueError(f"Size override for unknown body '{unknown[0]}'.")
    unknown = sorted(set(settings.face_group_sizes) - {name for g in groups.values() for name in g})
    if unknown:
        raise ValueError(f"Size override for unknown face group '{unknown[0]}'.")
    body_size = {b.name: target for b in geometry.bodies} | {
        name: settings.edge_m(size) for name, size in settings.body_sizes.items()
    }
    face_size = {f: min(body_size[b] for b in bodies) for f, bodies in model.face_bodies.items()}
    for mesh_groups in groups.values():
        for name, faces in mesh_groups.items():
            if name in settings.face_group_sizes:
                face_size.update(dict.fromkeys(faces, settings.edge_m(settings.face_group_sizes[name])))
    for face, size in settings.face_sizes:
        if isinstance(face, CadFaceRef):
            face = match_face_ref(face, geometry.faces, geometry.bbox_m, **settings.placement)
            if face is None:
                continue  # gone after a re-import: the editor notes it, generation goes on
        elif face not in face_size:
            raise ValueError(f"Size override for unknown face {face}.")
        face_size[face] = settings.edge_m(size)
    unknown = sorted(set(settings.volume_sizes) - set(model.volumes))
    if unknown:
        raise ValueError(f"Volume size for '{unknown[0]}', which is no air body.")
    volume_size = {b: body_size[b] for b in model.volumes} | {
        name: settings.edge_m(size) for name, size in settings.volume_sizes.items()
    }
    sizes = [*face_size.values(), *volume_size.values()] or [target]
    smallest, largest = min(sizes), max(sizes)
    option = gmsh.option.setNumber
    # Curvature may refine below the target size; only a finer override sets a floor.
    option("Mesh.MeshSizeMin", smallest if smallest < target else 0)
    option("Mesh.MeshSizeMax", largest)
    option("Mesh.MeshSizeFromCurvature", settings.curvature_elements_per_circle)
    if smallest == largest:
        return
    coarsest_face = defaultdict(float)
    for f, bodies in model.face_bodies.items():
        for body in bodies:
            coarsest_face[body] = max(coarsest_face[body], face_size[f])
    if any(size > coarsest_face[body] for body, size in volume_size.items()):
        # A volume coarser than all its faces never gets its size from them: sizes then extend from the faces
        # into surfaces only, and interiors take theirs from the fields below. Otherwise meshes stay as before.
        option("Mesh.MeshSizeExtendFromBoundary", -2)
    fields = gmsh.model.mesh.field
    region_of = {b: r.mesh_id for r in model.geometry.regions for b in r.bodies}
    mesh_volumes = defaultdict(list)
    for body, volumes in model.volumes.items():
        mesh_volumes[region_of[body]].extend(volumes)
    ramps = []
    # A face refines only the meshes it belongs to, not air bodies that are merely nearby.
    classes = defaultdict(list)
    for f, size in face_size.items():
        if size < largest:
            classes[size, tuple(m for m, faces in model.mesh_faces.items() if f in faces)].append(f)
    for (size, meshes), faces in sorted(classes.items()):
        distance = fields.add("Distance")
        fields.setNumbers(distance, "SurfacesList", faces)
        ramp = fields.add("Threshold")
        fields.setNumber(ramp, "InField", distance)
        fields.setNumber(ramp, "SizeMin", size)
        fields.setNumber(ramp, "SizeMax", largest)
        fields.setNumber(ramp, "DistMin", 0)
        fields.setNumber(ramp, "DistMax", (largest - size) / (settings.growth_rate - 1))
        restrict = fields.add("Restrict")
        fields.setNumber(restrict, "InField", ramp)
        fields.setNumbers(restrict, "SurfacesList", sorted({f for m in meshes for f in model.mesh_faces[m]}))
        fields.setNumbers(restrict, "VolumesList", [v for m in meshes for v in mesh_volumes[m]])
        ramps.append(restrict)
    for body, size in volume_size.items():
        if size < largest:
            cap = fields.add("Constant")
            fields.setNumbers(cap, "VolumesList", model.volumes[body])
            fields.setNumber(cap, "VIn", size)
            fields.setNumber(cap, "VOut", largest)
            fields.setNumber(cap, "IncludeBoundary", 0)
            ramps.append(cap)
    combined = fields.add("Min")
    fields.setNumbers(combined, "FieldsList", ramps)
    fields.setAsBackgroundMesh(combined)


def _write_volume(gmsh, path, surface_groups, volume_groups) -> None:
    """Native Gmsh 4.1 ASCII with entity classification, as BEAT's volume importer requires."""
    for dim, tag in gmsh.model.getPhysicalGroups():
        # Names outlive their groups in Gmsh and would label the reused tags.
        gmsh.model.removePhysicalName(gmsh.model.getPhysicalName(dim, tag))
    gmsh.model.removePhysicalGroups()
    tag = 1
    for name, faces in surface_groups.items():
        gmsh.model.addPhysicalGroup(2, faces, tag, name)
        tag += 1
    for name, volumes in volume_groups.items():
        gmsh.model.addPhysicalGroup(3, volumes, tag, name)
        tag += 1
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.write(str(path))


def _write_surface(gmsh, path, surface_groups) -> None:
    """Gmsh 2.2 ASCII triangles with physical and geometrical tags, oriented towards exterior air."""
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    index = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
    index[node_tags.astype(np.int64)] = np.arange(len(node_tags))
    points = np.asarray(coords).reshape(-1, 3)
    triangles, physical, geometrical, names = [], [], [], {}
    for tag, (name, faces) in enumerate(surface_groups.items(), start=1):
        names[name] = np.array([tag, 2])
        for face in faces:
            types, _, nodes = gmsh.model.mesh.getElements(2, face)
            for kind, data in zip(types, nodes, strict=True):
                if kind != 2:
                    raise ValueError(f"Unexpected Gmsh element type {kind} on face {face}.")
                block = index[np.asarray(data, dtype=np.int64)].reshape(-1, 3)
                triangles.append(block)
                physical.append(np.full(len(block), tag))
                geometrical.append(np.full(len(block), face))
    triangles = _orient_outward(points, np.concatenate(triangles))
    used = np.unique(triangles)
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    mesh = meshio.Mesh(
        points[used],
        [("triangle", remap[triangles])],
        cell_data={"gmsh:physical": [np.concatenate(physical)], "gmsh:geometrical": [np.concatenate(geometrical)]},
        field_data=names,
    )
    meshio.write(path, mesh, file_format="gmsh22", binary=False)


def _orient_outward(points, triangles) -> np.ndarray:
    """Orient each edge-connected patch consistently, with positive enclosed volume.

    The signed volume is taken about the origin, so patches left open on the X=0
    and Y=0 symmetry planes are still judged correctly.
    """
    triangles = triangles.copy()
    by_edge = defaultdict(list)
    for i, (a, b, c) in enumerate(triangles):
        for p, q in ((a, b), (b, c), (c, a)):
            by_edge[(min(p, q), max(p, q))].append((i, p < q))
    neighbours = [[] for _ in triangles]
    for adjacent in by_edge.values():
        if len(adjacent) == 2:
            (i, forward_i), (j, forward_j) = adjacent
            neighbours[i].append((j, forward_i == forward_j))
            neighbours[j].append((i, forward_i == forward_j))
    flip = np.full(len(triangles), -1)
    for seed in range(len(triangles)):
        if flip[seed] >= 0:
            continue
        flip[seed] = 0
        component, stack = [seed], [seed]
        while stack:
            i = stack.pop()
            for j, same_direction in neighbours[i]:
                if flip[j] < 0:
                    flip[j] = flip[i] ^ same_direction
                    component.append(j)
                    stack.append(j)
        component = np.array(component)
        flips = component[flip[component] == 1]
        triangles[flips] = triangles[flips][:, [0, 2, 1]]
        a, b, c = points[triangles[component]].transpose(1, 0, 2)
        if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
            triangles[component] = triangles[component][:, [0, 2, 1]]
    return triangles


@contextmanager
def _gmsh_session():
    import gmsh

    with _GMSH_LOCK:
        initialized_here = not gmsh.isInitialized()
        if initialized_here:
            gmsh.initialize(readConfigFiles=False, interruptible=False)
        gmsh.model.add("cad_import")
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.NumThreads", 1)
            yield gmsh
        finally:
            gmsh.model.remove()
            if initialized_here:
                gmsh.finalize()
