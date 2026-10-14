"""STEP -> classified, fragmented OpenCASCADE model -> one conformal mesh -> assembly files.

Solids are air bodies (bounded regions, one tetrahedral mesh per touching group);
surface bodies are exterior surface bodies (together one BEM surface mesh). Closed
shells of free faces are exterior unless chosen as air, then they become solids. An air
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
from dataclasses import dataclass, field
from pathlib import Path

import meshio
import numpy as np

WALLS = "walls"
EXTERIOR_MESH_ID = "exterior"
SYMMETRY_LABELS = {"off": "Off", "x": "X", "xy": "XY"}
AXES = {
    "+X": (1, 0, 0),
    "-X": (-1, 0, 0),
    "+Y": (0, 1, 0),
    "-Y": (0, -1, 0),
    "+Z": (0, 0, 1),
    "-Z": (0, 0, -1),
}
_GMSH_LOCK = threading.RLock()
# Fingerprint tolerances for small CAD edits: centre within 1 % of the bounding box
# diagonal, area within 5 %, normals within about 2.6 degrees.
CENTRE_TOLERANCE, AREA_TOLERANCE, NORMAL_COSINE = 1e-2, 5e-2, 0.999


@dataclass(frozen=True)
class CadFaceRef:
    """A face as a face group stores it: STEP name and fingerprint in CAD axes (before axis mapping).

    It resolves to one face: the only face carrying ``step_name``, else the face with the same
    surface type, centre, area and normal (among those carrying ``step_name``, if several do).
    """

    step_name: str | None
    centre: tuple[float, float, float]
    area: float
    normal: tuple[float, float, float]
    surface_type: str


@dataclass(frozen=True)
class CadImportSettings:
    """Meshing inputs. Sizes in metres; ``forward``/``up`` name the CAD axes mapped to +Z/+Y.

    ``face_groups`` maps a group name to its faces: ``CadFaceRef`` (or mappings of its
    fields), or face ids (``CadFace.id``). Faces with a STEP name in no face group form a
    group of that name. ``symmetry`` is the project symmetry to check the geometry
    against; ``None`` accepts what is detected. ``closed_shells`` maps a closed shell name
    to ``"air"`` or ``"exterior"`` (the default).
    """

    fmax_hz: float
    sound_speed_m_per_s: float = 343.0
    elements_per_wavelength: float = 5.0
    forward: str = "+Z"
    up: str = "+Y"
    face_groups: Mapping[str, tuple[int | CadFaceRef, ...]] = field(default_factory=dict)
    body_sizes_m: Mapping[str, float] = field(default_factory=dict)
    face_group_sizes_m: Mapping[str, float] = field(default_factory=dict)
    curvature_elements_per_circle: float = 12.0
    growth_rate: float = 1.3
    symmetry: str | None = None
    closed_shells: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (
            ("Maximum frequency", self.fmax_hz),
            ("Sound speed", self.sound_speed_m_per_s),
            ("Elements per wavelength", self.elements_per_wavelength),
            ("Curvature elements per circle", self.curvature_elements_per_circle),
            *((f"Size override for '{k}'", v) for k, v in {**self.body_sizes_m, **self.face_group_sizes_m}.items()),
        ):
            if not (isinstance(value, int | float) and math.isfinite(value) and value > 0):
                raise ValueError(f"{label} must be a positive number.")
        if not (math.isfinite(self.growth_rate) and self.growth_rate > 1):
            raise ValueError("Growth rate must be greater than 1.")
        _axis_matrix(self.forward, self.up)
        groups = {
            name: tuple(CadFaceRef(**face) if isinstance(face, Mapping) else face for face in faces)
            for name, faces in self.face_groups.items()
        }
        object.__setattr__(self, "face_groups", groups)
        if self.symmetry not in (None, *SYMMETRY_LABELS):
            raise ValueError(f"Unknown symmetry '{self.symmetry}'.")

    @property
    def target_size_m(self) -> float:
        """Sound speed / maximum frequency / elements per wavelength."""
        return self.sound_speed_m_per_s / self.fmax_hz / self.elements_per_wavelength


@dataclass(frozen=True)
class CadFace:
    """A face of the fragmented geometry; ``id`` is stable for the same file and axes."""

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
    """An opening between ``mesh_id`` and the exterior; ``name`` is the face group on both sides."""

    name: str
    mesh_id: str
    face_ids: tuple[int, ...]


@dataclass(frozen=True)
class CadClosedShell:
    """Closed connected free faces: an air volume or exterior surface, as the user chooses."""

    name: str
    role: str  # "air" | "exterior"


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
    step_path: str | Path, *, forward: str = "+Z", up: str = "+Y", closed_shells: Mapping[str, str] | None = None
) -> CadGeometry:
    """Read, orient, classify and fragment a STEP file without meshing it."""
    with _gmsh_session() as gmsh:
        return _analyse(gmsh, Path(step_path), _axis_matrix(forward, up), closed_shells or {}).geometry


def face_ref(face: CadFace, *, forward: str = "+Z", up: str = "+Y") -> CadFaceRef:
    """How a face group stores ``face`` of a geometry read with axes ``forward``/``up``."""
    back = _axis_matrix(forward, up).T
    return CadFaceRef(
        face.step_name,
        tuple(float(v) for v in back @ np.asarray(face.centre, dtype=float)),
        float(face.area),
        tuple(float(v) for v in back @ np.asarray(face.normal, dtype=float)),
        face.surface_type,
    )


@dataclass(frozen=True)
class CadPreview:
    """A coarse display tessellation: ``triangles`` index ``points`` (m); ``face_ids`` per triangle."""

    geometry: CadGeometry
    points: np.ndarray
    triangles: np.ndarray
    face_ids: np.ndarray


def preview_step(
    step_path: str | Path, *, forward: str = "+Z", up: str = "+Y", closed_shells: Mapping[str, str] | None = None
) -> CadPreview:
    """Inspect a STEP file and tessellate its faces coarsely for display."""
    with _gmsh_session() as gmsh:
        geometry = _analyse(gmsh, Path(step_path), _axis_matrix(forward, up), closed_shells or {}).geometry
        low, high = geometry.bbox_m
        gmsh.option.setNumber("Mesh.MeshSizeMax", math.dist(low, high) / 30)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.model.mesh.generate(2)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        index = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
        index[node_tags.astype(np.int64)] = np.arange(len(node_tags))
        triangles, face_ids = [], []
        for face in geometry.faces:
            types, _, nodes = gmsh.model.mesh.getElements(2, face.id)
            for kind, data in zip(types, nodes, strict=True):
                if kind == 2:
                    block = index[np.asarray(data, dtype=np.int64)].reshape(-1, 3)
                    triangles.append(block)
                    face_ids.append(np.full(len(block), face.id))
        return CadPreview(
            geometry,
            np.asarray(coords).reshape(-1, 3),
            np.concatenate(triangles) if triangles else np.zeros((0, 3), dtype=np.int64),
            np.concatenate(face_ids) if face_ids else np.zeros(0, dtype=np.int64),
        )


def mesh_step(step_path: str | Path, settings: CadImportSettings, output_dir: str | Path) -> CadImportResult:
    """Mesh a STEP file into ``output_dir`` as ``<mesh id>.msh`` files (metres)."""
    output_dir = Path(output_dir)
    with _gmsh_session() as gmsh:
        model = _analyse(gmsh, Path(step_path), _axis_matrix(settings.forward, settings.up), settings.closed_shells)
        geometry = model.geometry
        if settings.symmetry is not None and settings.symmetry != geometry.symmetry:
            raise ValueError(
                f"Project symmetry {SYMMETRY_LABELS[settings.symmetry]} does not match the geometry, "
                f"which suggests symmetry {SYMMETRY_LABELS[geometry.symmetry]}."
            )
        groups = model.surface_groups(settings.face_groups, forward=settings.forward, up=settings.up)
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

    def surface_groups(
        self, face_groups: Mapping[str, tuple[int | CadFaceRef, ...]], *, forward: str, up: str
    ) -> dict[str, dict[str, list[int]]]:
        """Mesh id -> {surface group name -> faces}, sorted by name; ``forward``/``up`` as read."""
        member = {}
        reserved = {WALLS, *self.interface_of.values()}
        geometry = self.geometry
        found = {}
        for name, refs in face_groups.items():
            if not name or name in reserved:
                raise ValueError(f"Face group name '{name}' is reserved.")
            found[name] = [
                match_face_ref(ref, geometry.faces, geometry.bbox_m, forward=forward, up=up)
                if isinstance(ref, CadFaceRef)
                else ref
                for ref in refs
            ]
        # A group missing any stored face would silently lose it to walls: the user reviews it.
        review = [name for name, faces in found.items() if None in faces]
        if review:
            raise ValueError(
                "Face groups need review (stored faces not found in the STEP file): "
                f"{', '.join(repr(n) for n in review)}. Re-pick or delete them."
            )
        for name, faces in found.items():
            for face in faces:
                if face not in self.face_bodies:
                    raise ValueError(f"Face group '{name}' refers to unknown face {face}.")
                if face in self.interface_of:
                    raise ValueError(
                        f"Face group '{name}' contains face {face} of interface '{self.interface_of[face]}'."
                    )
                if face in member and member[face] != name:
                    raise ValueError(f"Face {face} is in face groups '{member[face]}' and '{name}'.")
                member[face] = name
        for face in self.geometry.faces:
            name = face.step_name
            if name and name not in reserved and face.id not in member and face.id not in self.interface_of:
                member[face.id] = name
        result = {}
        for mesh_id, faces in self.mesh_faces.items():
            groups = defaultdict(list)
            for face in faces:
                groups[self.interface_of.get(face) or member.get(face) or WALLS].append(face)
            result[mesh_id] = dict(sorted(groups.items()))
        return result


def match_face_ref(
    ref: CadFaceRef, faces: Sequence[CadFace], bbox_m, *, forward: str = "+Z", up: str = "+Y"
) -> int | None:
    """Id of the face among ``faces`` (read with ``forward``/``up``, bounded by ``bbox_m``) that ``ref`` stands for.

    The only face carrying its STEP name; else, among the faces with that name (or all faces
    when none has it), the closest whose fingerprint matches after axis mapping.
    """
    named = [f for f in faces if ref.step_name and f.step_name == ref.step_name]
    if len(named) == 1:
        return named[0].id
    matrix = _axis_matrix(forward, up)
    centre, normal = matrix @ np.asarray(ref.centre, dtype=float), matrix @ np.asarray(ref.normal, dtype=float)
    tol = CENTRE_TOLERANCE * math.dist(*bbox_m)
    matches = [
        f
        for f in named or faces
        if f.surface_type == ref.surface_type
        and math.dist(f.centre, centre) <= tol
        and abs(f.area - ref.area) <= AREA_TOLERANCE * ref.area
        and abs(np.dot(f.normal, normal)) >= NORMAL_COSINE
    ]
    return min(matches, key=lambda f: math.dist(f.centre, centre)).id if matches else None


def _analyse(gmsh, path: Path, matrix: np.ndarray, closed_shells: Mapping[str, str]) -> _Model:
    occ = gmsh.model.occ
    for name, role in closed_shells.items():
        if role not in ("air", "exterior"):
            raise ValueError(f"Closed shell '{name}' must be 'air' or 'exterior', not '{role}'.")
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
    solid_names, shell_names = names[: len(solids)], names[len(solids) :]
    unknown = sorted(set(closed_shells) - set(shell_names))
    if unknown:
        raise ValueError(f"Unknown closed shell '{unknown[0]}'.")
    roles = {name: closed_shells.get(name, "exterior") for name in shell_names}
    for name, faces in zip(shell_names, shells, strict=True):
        if roles[name] == "air":
            solids.append(occ.addVolume([occ.addSurfaceLoop(faces)]))
            solid_names.append(name)
            sheets = [f for f in sheets if f not in faces]
    sheet_names = [_label(gmsh, 2, t) for t in sheets]
    inputs = [(3, t) for t in solids] + [(2, t) for t in sheets]
    # A rigid rotation keeps exact surface types (a general affine transform makes BSplines).
    angle = math.acos(np.clip((np.trace(matrix) - 1) / 2, -1, 1))
    if angle > 1e-9:
        if math.pi - angle > 1e-9:
            axis = [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
        else:
            axis = (matrix + np.eye(3))[:, int(np.argmax(np.diag(matrix)))]
        occ.rotate(inputs, 0, 0, 0, *axis, angle)
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
    tol = max(1e-9, 1e-6 * math.dist(bbox[:3], bbox[3:]))
    plane = _PlaneTest(gmsh, tol)

    # Openings: air faces whose edges close holes in the exterior surface.
    exterior_set = set(exterior)
    counts = Counter(e for f in exterior for e in edges[f])
    free = {e for e, n in counts.items() if n == 1}
    fillers = {f for f in outer_air if f not in exterior_set and edges[f] & free and not plane.face(f)}
    while True:
        shared = Counter(e for f in fillers for e in edges[f])
        keep = {f for f in fillers if all(e in free or shared[e] > 1 or plane.edge(e) for e in edges[f])}
        if keep == fillers:
            break
        fillers = keep

    warnings = []
    split = {f for kids in sheet_children if len(kids) > 1 for f in kids}
    exterior_name = {f: name for name, faces in exterior_bodies.items() for f in faces}
    openings = set(fillers)
    for f in sorted(exterior_set.intersection(outer_air)):
        if f in split:
            warnings.append(
                f"Exterior surface '{exterior_name[f]}' covers a face of air body "
                f"'{body_of_volume[face_volumes[f][0]]}' (face {f}) instead of opening into it; "
                "no interface was created there."
            )
        else:
            openings.add(f)

    shell = sorted(exterior_set | fillers)
    counts = Counter(e for f in shell for e in edges[f])
    open_edges = sorted(e for e, n in counts.items() if n == 1)
    on_x = [e for e in open_edges if plane.edge(e, axis=0)]
    on_y = [e for e in open_edges if e not in on_x and plane.edge(e, axis=1)]
    symmetry = ("xy" if on_y else "x") if on_x else "off"
    # Y=0 alone is no supported symmetry, so those edges are gaps too.
    on_planes = set(on_x) | (set(on_y) if on_x else set())
    gaps = [e for e in open_edges if e not in on_planes]
    if gaps:
        warnings.append(f"Exterior surface is not closed: {len(gaps)} open edge(s) near {plane.describe(gaps)}.")
    overlaps = sorted(e for e, n in counts.items() if n > 2)
    if overlaps:
        warnings.append(
            f"Exterior surface bodies overlap: {len(overlaps)} edge(s) are shared by more than two faces "
            f"near {plane.describe(overlaps)}."
        )

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
    if shell:
        regions.append(CadRegion(EXTERIOR_MESH_ID, "unbounded_air", tuple(exterior_bodies)))
        mesh_faces[EXTERIOR_MESH_ID] = shell
    mesh_of_body = {b: r.mesh_id for r in regions for b in r.bodies}

    # Interfaces: connected opening faces of one air body.
    interfaces = []
    interface_of = {}
    by_body = defaultdict(list)
    for f in sorted(openings):
        by_body[body_of_volume[face_volumes[f][0]]].append(f)
    for body in volumes:
        for k, faces in enumerate(_components(by_body[body], edges), start=1):
            name = f"{body} opening" + (f" {k}" if k > 1 else "")
            interfaces.append(CadInterface(name, mesh_of_body[body], tuple(faces)))
            interface_of.update(dict.fromkeys(faces, name))

    step_names = _match_step_names(gmsh, path, matrix, sorted(edges), tol)
    face_bodies = {}
    faces = []
    for f in sorted(set(outer_air) | set(shell)):
        owners = tuple(body_of_volume[v] for v in face_volumes[f])
        face_bodies[f] = owners + ((exterior_name[f],) if f in exterior_name else ())
        faces.append(_face(gmsh, f, face_bodies[f], step_names.get(f)))
    bodies = [CadBody(n, "air", tuple(f for f in face_bodies if n in face_bodies[f])) for n in volumes]
    bodies += [CadBody(n, "exterior", tuple(fs)) for n, fs in exterior_bodies.items()]
    geometry = CadGeometry(
        tuple(bodies),
        tuple(faces),
        tuple(regions),
        tuple(interfaces),
        (tuple(bbox[:3]), tuple(bbox[3:])),
        symmetry,
        tuple(warnings),
        tuple(CadClosedShell(name, role) for name, role in roles.items()),
    )
    return _Model(geometry, volumes, mesh_faces, interface_of, face_bodies)


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


class _PlaneTest:
    """Symmetry-plane tests (X=0, Y=0) on edges and faces; edges are sampled at nine points."""

    def __init__(self, gmsh, tol):
        self.gmsh, self.tol, self._points = gmsh, tol, {}

    def points(self, edge):
        if edge not in self._points:
            low, high = self.gmsh.model.getParametrizationBounds(1, edge)
            params = np.linspace(low[0], high[0], 9).tolist()
            self._points[edge] = np.reshape(self.gmsh.model.getValue(1, edge, params), (-1, 3))
        return self._points[edge]

    def edge(self, edge, axis=None):
        axes = (0, 1) if axis is None else (axis,)
        return any(np.all(np.abs(self.points(edge)[:, a]) <= self.tol) for a in axes)

    def face(self, face):
        if self.gmsh.model.getType(2, face) != "Plane":
            return False
        centre = self.gmsh.model.occ.getCenterOfMass(2, face)
        normal = _normal(self.gmsh, face)
        return any(abs(centre[a]) <= self.tol and abs(abs(normal[a]) - 1) < 1e-9 for a in (0, 1))

    def describe(self, edges, limit=5):
        mids = [self.points(e)[4] * 1000 for e in edges[:limit]]
        text = ", ".join("({:.1f}, {:.1f}, {:.1f}) mm".format(*m) for m in mids)
        return text + (", ..." if len(edges) > limit else "")


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


def _axis_matrix(forward, up) -> np.ndarray:
    """Rows: the CAD directions that become Boundary Lab +X, +Y and +Z."""
    if forward not in AXES or up not in AXES:
        raise ValueError(f"Axes must be one of {', '.join(AXES)}.")
    f, u = np.array(AXES[forward], dtype=float), np.array(AXES[up], dtype=float)
    if f @ u != 0:
        raise ValueError(f"Forward axis {forward} and up axis {up} must be perpendicular.")
    return np.array([np.cross(u, f), u, f])


# --- STEP face names ------------------------------------------------------------------------------

_SI_PREFIX = {"$": 1.0, ".MILLI.": 1e-3, ".CENTI.": 1e-2, ".DECI.": 0.1, ".KILO.": 1e3, ".MICRO.": 1e-6}
_CONVERSION_UNITS = {"INCH": 0.0254, "FOOT": 0.3048}


def _match_step_names(gmsh, path, matrix, faces, tol) -> dict[int, str]:
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
        points = points @ matrix.T
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
    """Target size everywhere, overrides per body / face group, graded by the growth rate."""
    target = settings.target_size_m
    unknown = sorted(set(settings.body_sizes_m) - {b.name for b in model.geometry.bodies})
    if unknown:
        raise ValueError(f"Size override for unknown body '{unknown[0]}'.")
    unknown = sorted(set(settings.face_group_sizes_m) - {name for g in groups.values() for name in g})
    if unknown:
        raise ValueError(f"Size override for unknown face group '{unknown[0]}'.")
    body_size = {b.name: settings.body_sizes_m.get(b.name, target) for b in model.geometry.bodies}
    face_size = {f: min(body_size[b] for b in bodies) for f, bodies in model.face_bodies.items()}
    for mesh_groups in groups.values():
        for name, faces in mesh_groups.items():
            if name in settings.face_group_sizes_m:
                face_size.update(dict.fromkeys(faces, settings.face_group_sizes_m[name]))
    volume_size = {b: body_size[b] for b in model.volumes}
    sizes = [*face_size.values(), *volume_size.values()] or [target]
    smallest, largest = min(sizes), max(sizes)
    option = gmsh.option.setNumber
    # Curvature may refine below the target size; only a finer override sets a floor.
    option("Mesh.MeshSizeMin", smallest if smallest < target else 0)
    option("Mesh.MeshSizeMax", largest)
    option("Mesh.MeshSizeFromCurvature", settings.curvature_elements_per_circle)
    if smallest == largest:
        return
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
