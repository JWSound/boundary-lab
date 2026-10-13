"""STEP fixtures for CAD import tests, built in code with Gmsh's OpenCASCADE kernel.

Coordinates are millimetres; Gmsh writes them as a millimetre STEP file. Gmsh does
not export names, so product (body) and face names are written into the STEP text
afterwards. Bodies are exported as one product per solid, then one per free face.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import gmsh

HALF_SPACES = {"off": None, "x": (0, -1000, -1000), "xy": (0, 0, -1000)}


def _cut_to_symmetry(entities, symmetry):
    corner = HALF_SPACES[symmetry]
    if corner is None:
        return entities
    occ = gmsh.model.occ
    result = []
    for entity in entities:
        box = occ.addBox(*corner, 2000, 2000, 2000)
        kept, _ = occ.intersect([entity], [(3, box)])
        result.extend(kept)
    return result


def _box_faces(x0, y0, z0, dx, dy, dz):
    occ = gmsh.model.occ
    box = occ.addBox(x0, y0, z0, dx, dy, dz)
    occ.synchronize()
    faces = gmsh.model.getBoundary([(3, box)], oriented=False)
    occ.remove([(3, box)])
    return faces


def _cabinet(half_x, half_y, depth, holes=()):
    """Closed box surface with its front face at z=0, cut by circular holes (x, y, r)."""
    occ = gmsh.model.occ
    faces = _box_faces(-half_x, -half_y, -depth, 2 * half_x, 2 * half_y, depth)
    occ.synchronize()
    front = [f for f in faces if abs(occ.getCenterOfMass(*f)[2]) < 1e-9]
    rest = [f for f in faces if f not in front]
    if holes:
        front, _ = occ.cut(front, [(3, occ.addCylinder(x, y, -1, 0, 0, 2, r)) for x, y, r in holes])
    return rest + front


def write_step(
    path: Path,
    build: Callable[[], tuple[Sequence[tuple[int, int]], Sequence[str]]],
    *,
    face_names: Callable[[tuple[float, float, float]], str | None] = lambda centre: None,
) -> Path:
    """Run ``build`` in a fresh Gmsh model and write its entities as named STEP.

    ``build`` returns the top-level entities and one product name per entity
    (solids first, in tag order, then free faces in tag order). ``face_names``
    maps a face's vertex centroid (mm) to a STEP face name.
    """
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        # OpenCASCADE keeps its STEP unit process-wide; an earlier import in metres would
        # otherwise make this export scale millimetre coordinates by 1000.
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        entities, names = build()
        gmsh.model.occ.synchronize()
        order = sorted(e for e in entities if e[0] == 3) + sorted(e for e in entities if e[0] == 2)
        assert len(order) == len(names), (order, names)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    text = path.read_text(encoding="utf-8")
    # The first product is the assembly root; one part product per entity follows.
    product_names = iter([None, *names])

    def rename(match):
        name = next(product_names)
        return match.group(0) if name is None else f"PRODUCT('{name}','{name}'"

    text = re.sub(r"PRODUCT\('[^']*',\s*'[^']*'", rename, text)
    path.write_text(_name_faces(text, face_names), encoding="utf-8")
    return path


def strip_names(source: Path, target: Path) -> Path:
    """Copy a STEP file with every product and face name removed."""
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"PRODUCT\('[^']*',\s*'[^']*'", "PRODUCT('',''", text)
    text = re.sub(r"ADVANCED_FACE\('[^']*'", "ADVANCED_FACE(''", text)
    target.write_text(text, encoding="utf-8")
    return target


def _name_faces(text, face_names):
    records = dict(re.findall(r"#(\d+)\s*=\s*([A-Z_0-9]+\(.*?\));", text, flags=re.S))
    points = {}
    for key, body in records.items():
        if body.startswith("CARTESIAN_POINT"):
            points[key] = tuple(float(v) for v in re.search(r"\(([^()]*)\)\s*\)$", body).group(1).split(","))

    def vertices(key, seen):
        body = records[key]
        if body.startswith("VERTEX_POINT"):
            return {re.findall(r"#(\d+)", body)[0]}
        if not body.startswith(("FACE_", "EDGE_LOOP", "ORIENTED_EDGE", "EDGE_CURVE", "VERTEX_LOOP")):
            return set()
        found = set()
        for ref in re.findall(r"#(\d+)", body):
            if ref not in seen:
                seen.add(ref)
                found |= vertices(ref, seen)
        return found

    def rename(match):
        key = match.group(1)
        bounds = re.findall(r"#(\d+)", records[key].split("),")[0])
        refs = set().union(*(vertices(b, set()) for b in bounds))
        coords = [points[r] for r in refs]
        centre = tuple(sum(c[i] for c in coords) / len(coords) for i in range(3))
        name = face_names(centre)
        return match.group(0) if name is None else f"#{key} = ADVANCED_FACE('{name}'"

    return re.sub(r"#(\d+)\s*=\s*ADVANCED_FACE\(''", rename, text)


# (forward, up) CAD axes -> rotation (axis x, y, z, angle) taking model +Z/+Y onto them.
CAD_ROTATIONS = {("+Z", "+Y"): None, ("-Y", "+Z"): (1, 0, 0, math.pi / 2)}


def vented_box(
    path: Path,
    *,
    symmetry: str = "off",
    face_names=None,
    covered: bool = False,
    open_back: bool = False,
    cad_axes: tuple[str, str] = ("+Z", "+Y"),
) -> Path:
    """Chamber (100x100x180 mm) joined by a port (r=10 mm) to a hole in a cabinet surface.

    The cabinet front lies at z=0 and the port mouth fills its hole. ``covered``
    leaves the cabinet front without the hole, covering the port mouth;
    ``open_back`` leaves out the cabinet's back face (z=-210 mm).
    ``cad_axes`` (forward, up) names the CAD axes that the model's +Z and +Y are
    written along.
    """

    def build():
        occ = gmsh.model.occ
        chamber = _cut_to_symmetry([(3, occ.addBox(-50, -50, -200, 100, 100, 180))], symmetry)
        port = _cut_to_symmetry([(3, occ.addCylinder(0, 0, -20, 0, 0, 20, 10))], symmetry)
        cabinet = _cut_to_symmetry(_cabinet(60, 60, 210, () if covered else [(0, 0, 10)]), symmetry)
        if open_back:
            occ.synchronize()
            back = [f for f in cabinet if abs(occ.getCenterOfMass(*f)[2] + 210) < 1e-9]
            occ.remove(back, recursive=True)
            cabinet = [f for f in cabinet if f not in back]
        entities = chamber + port + cabinet
        rotation = CAD_ROTATIONS[cad_axes]
        if rotation:
            occ.rotate(entities, 0, 0, 0, *rotation)
        return entities, ["Chamber", "Port"] + ["Cabinet"] * len(cabinet)

    def default_names(centre):
        return "Driver" if abs(centre[2] + 200) < 1e-6 else None

    return write_step(path, build, face_names=face_names or default_names)


def waveguide(path: Path, *, surface_model: bool = False) -> Path:
    """Conical throat (r 15 -> 40 mm over 40 mm) opening into a 200 mm baffle box at z=0.

    ``surface_model`` writes it like SolidWorks writes parts mixing solid and surface
    bodies: no solid, the throat only as the closed shell of a surface model, no names.
    """

    def build():
        occ = gmsh.model.occ
        throat = [(3, occ.addCone(0, 0, -40, 0, 0, 40, 15, 40))]
        baffle = _cabinet(100, 100, 150, [(0, 0, 40)])
        return throat + baffle, ["Throat"] + ["Baffle"] * len(baffle)

    write_step(path, build, face_names=lambda c: "Driver" if abs(c[2] + 40) < 1e-6 else None)
    if surface_model:
        text = path.read_text(encoding="utf-8")
        text = text.replace("ADVANCED_BREP_SHAPE_REPRESENTATION(", "MANIFOLD_SURFACE_SHAPE_REPRESENTATION(")
        text = re.sub(r"MANIFOLD_SOLID_BREP\('[^']*',\s*(#\d+)\)", r"SHELL_BASED_SURFACE_MODEL('NONE',(\1))", text)
        path.write_text(text, encoding="utf-8")
        strip_names(path, path)
    return path


def two_chambers(path: Path) -> Path:
    """Two separate ported chambers side by side behind one cabinet with two holes."""

    def build():
        occ = gmsh.model.occ
        chambers = [(3, occ.addBox(x - 40, -40, -150, 80, 80, 130)) for x in (-60, 60)]
        ports = [(3, occ.addCylinder(x, 0, -20, 0, 0, 20, 10)) for x in (-60, 60)]
        cabinet = _cabinet(120, 60, 160, [(-60, 0, 10), (60, 0, 10)])
        return chambers + ports + cabinet, ["Left", "Right", "Left port", "Right port"] + ["Cabinet"] * len(cabinet)

    return write_step(path, build)
