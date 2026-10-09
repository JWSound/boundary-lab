"""Isolated Gmsh worker. One air volume, two exports, one shared mouth mesh."""

import json
import math
import runpy
import sys
from pathlib import Path

import gmsh
import meshio
import numpy as np

# Executed by absolute filename so packaged plugins need no sys.path installation.
if __package__:
    from .parameters import port_profile, validate
else:
    # Embedded Windows Python uses an isolated ._pth and does not add this
    # script's directory to sys.path. Load our sibling by its explicit path.
    _parameters = runpy.run_path(str(Path(__file__).with_name("parameters.py")))
    port_profile, validate = _parameters["port_profile"], _parameters["validate"]


def build(source, output):
    p = validate(source)
    output = Path(output)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.NumThreads", 2)
        gmsh.model.add("vented_enclosure")
        occ = gmsh.model.occ
        w, h, d, t = (p[k] for k in ("width_m", "height_m", "depth_m", "wall_m"))
        dy, py = p["driver_y_m"], p["port_y_m"]
        dr = p["driver_diameter_m"] / 2
        profile = port_profile(p)
        pr = profile["mouth"][1]
        outer = pr + p["port_wall_m"]
        length = p["port_length_m"]

        def circle(radius, y, z):
            return occ.addWire([occ.addCircle(0, y, z, radius)])

        def face(width, height, z, holes):
            corners = [
                (-width / 2, -height / 2),
                (width / 2, -height / 2),
                (width / 2, height / 2),
                (-width / 2, height / 2),
            ]
            points = [occ.addPoint(x, y, z) for x, y in corners]
            wire = occ.addWire([occ.addLine(points[i], points[(i + 1) % 4]) for i in range(4)])
            return occ.addPlaneSurface([wire, *holes])

        def box_sides(width, height, depth, front):
            volume = occ.addBox(-width / 2, -height / 2, front - depth, width, height, depth)
            occ.synchronize()
            faces = [tag for _, tag in gmsh.model.getBoundary([(3, volume)], oriented=False)]
            front_tag = max(faces, key=lambda tag: occ.getCenterOfMass(2, tag)[2])
            occ.remove([(3, volume)], recursive=False)
            occ.remove([(2, front_tag)], recursive=False)
            return [tag for tag in faces if tag != front_tag]

        def lathe(start, end, y, centre=None):
            # Coordinates are axial z, radial r. Revolve exact lines/circle arcs.
            a, b = (occ.addPoint(r, y, z) for z, r in (start, end))
            if centre is None:
                curve = occ.addLine(a, b)
            else:
                z, r = centre
                curve = occ.addCircleArc(a, occ.addPoint(r, y, z), b)
            return [tag for dim, tag in occ.revolve([(1, curve)], 0, y, 0, 0, 0, 1, 2 * math.pi) if dim == 2]

        bem_wall = box_sides(w, h, d, 0)
        bem_wall.append(face(w, h, 0, [circle(dr, dy, 0), circle(pr, py, 0)]))
        fem_wall = box_sides(w - 2 * t, h - 2 * t, d - 2 * t, -t)
        fem_wall.append(face(w - 2 * t, h - 2 * t, -t, [circle(dr, dy, -t), circle(outer, py, -t)]))
        fem_wall += lathe((-t, dr), (0, dr), dy)
        fem_wall += lathe((-t, outer), (-length, outer), py)
        fem_wall.append(occ.addPlaneSurface([circle(outer, py, -length), circle(pr, py, -length)]))

        driver = lathe((0, dr), (0, dr - p["surround_width_m"]), dy)
        cap = p["cap_diameter_m"] / 2
        depth = p["cone_depth_m"]
        driver += lathe((0, dr - p["surround_width_m"]), (-depth, cap), dy)
        cap_height = cap / 2
        centre_z = -depth + (cap_height**2 - cap**2) / (2 * cap_height)
        driver += lathe((-depth, cap), (-depth + cap_height, 0), dy, (centre_z, 0))

        throat, join = profile["throat"], profile["join"]
        for lower in (False, True):

            def mirrored(point):
                return (-length - point[0], point[1]) if lower else point

            centre = profile["flare_centre"]
            fem_wall += lathe(mirrored(throat), mirrored(join), py, mirrored(centre) if centre else None)
            if p["port_roundover_m"] > 0:
                fem_wall += lathe(mirrored(join), mirrored(profile["mouth"]), py, mirrored(profile["lip_centre"]))
        mouth = occ.addPlaneSurface([circle(pr, py, 0)])
        # Merge coincident edges before volume creation: driver and mouth surfaces
        # belong to both exports and are meshed only once.
        occ.removeAllDuplicates()
        shell = occ.addSurfaceLoop([*fem_wall, *driver, mouth], sewing=True)
        volume = occ.addVolume([shell])
        occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [volume], 10, "air")
        groups = {
            "inner_wall": (1, fem_wall),
            "outer_wall": (2, bem_wall),
            "driver": (3, driver),
            "port_mouth": (4, [mouth]),
        }
        for name, (tag, surfaces) in groups.items():
            gmsh.model.addPhysicalGroup(2, surfaces, tag, name)
        gmsh.option.setNumber("Mesh.MeshSizeMax", p["mesh_size_m"])
        gmsh.option.setNumber("Mesh.MeshSizeMin", p["detail_size_m"])
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), p["mesh_size_m"])
        detail_surfaces = [*driver, mouth, *fem_wall[6:]]
        detail_points = gmsh.model.getBoundary([(2, s) for s in detail_surfaces], recursive=True, oriented=False)
        gmsh.model.mesh.setSize(detail_points, p["detail_size_m"])
        gmsh.model.mesh.generate(3)

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        points = np.asarray(coordinates).reshape(-1, 3)
        index = {int(tag): i for i, tag in enumerate(node_tags)}

        def elements(dim, entity, expected):
            types, _, nodes = gmsh.model.mesh.getElements(dim, entity)
            result = []
            for kind, data in zip(types, nodes, strict=True):
                if kind != expected:
                    raise ValueError(f"Unexpected Gmsh element type {kind}.")
                result.extend(index[int(tag)] for tag in data)
            return np.asarray(result, dtype=int).reshape(-1, 3 if dim == 2 else 4)

        triangles = {
            name: np.concatenate([elements(2, s, 2) for s in surfaces]) for name, (_, surfaces) in groups.items()
        }
        tets = elements(3, volume, 4)

        def export(name, surface_names, include_volume=False):
            tri = np.concatenate([triangles[key] for key in surface_names])
            tags = np.concatenate([np.full(len(triangles[key]), groups[key][0]) for key in surface_names])
            if not include_volume:
                tri = orient_closed(points, tri)
            cells = [("triangle", tri)]
            physical = [tags]
            names = {key: [groups[key][0], 2] for key in surface_names}
            if include_volume:
                cells.append(("tetra", tets))
                physical.append(np.full(len(tets), 10))
                names["air"] = [10, 3]
            # Compact away unused nodes in each resource.
            used = np.unique(np.concatenate([data.ravel() for _, data in cells]))
            remap = np.full(len(points), -1, dtype=int)
            remap[used] = np.arange(len(used))
            if include_volume:
                # BEAT's volume importer requires native Gmsh 4.1 ASCII with
                # entity classification. Exclude the unrelated exterior walls.
                gmsh.model.removePhysicalGroups([(2, groups["outer_wall"][0])])
                gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
                gmsh.option.setNumber("Mesh.Binary", 0)
                gmsh.option.setNumber("Mesh.SaveAll", 0)
                gmsh.write(str(output / f"{name}.msh"))
            else:
                meshio.write(
                    output / f"{name}.msh",
                    meshio.Mesh(
                        points[used],
                        [(kind, remap[data]) for kind, data in cells],
                        cell_data={"gmsh:physical": physical, "gmsh:geometrical": physical},
                        field_data=names,
                    ),
                    file_format="gmsh22",
                    binary=False,
                )
            return {"nodes": len(used), "triangles": len(tri), "tetrahedra": len(tets) if include_volume else 0}

        stats = {
            "interior": export("interior", ("inner_wall", "driver", "port_mouth"), True),
            "exterior": export("exterior", ("outer_wall", "driver", "port_mouth")),
        }
        stats["air_volume_m3"] = occ.getMass(3, volume)
        stats["gmsh_version"] = gmsh.__version__
        (output / "mesh_info.json").write_text(json.dumps(stats), encoding="utf-8")
        gmsh.write(str(output / "geometry.brep"))
        return stats
    finally:
        gmsh.finalize()


def orient_closed(points, triangles):
    """Orient a single connected watertight BEM shell towards exterior air."""
    triangles = triangles.copy()
    edges = {}
    for i, tri in enumerate(triangles):
        for a, b in zip(tri, np.roll(tri, -1), strict=True):
            edges.setdefault(tuple(sorted((a, b))), []).append((i, a < b))
    if any(len(adjacent) != 2 for adjacent in edges.values()):
        raise ValueError("Generated exterior shell is not watertight.")
    neighbors = [[] for _ in triangles]
    for (a, sign_a), (b, sign_b) in edges.values():
        neighbors[a].append((b, sign_a == sign_b))
        neighbors[b].append((a, sign_a == sign_b))
    flips = {0: False}
    stack = [0]
    while stack:
        a = stack.pop()
        for b, opposite in neighbors[a]:
            flip = flips[a] ^ opposite
            if b in flips:
                if flips[b] != flip:
                    raise ValueError("Generated exterior shell is not orientable.")
            else:
                flips[b] = flip
                stack.append(b)
    if len(flips) != len(triangles):
        raise ValueError("Generated exterior shell is disconnected.")
    ids = [i for i, flip in flips.items() if flip]
    triangles[ids] = triangles[ids][:, [0, 2, 1]]
    a, b, c = points[triangles].transpose(1, 0, 2)
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
        triangles = triangles[:, [0, 2, 1]]
    return triangles


if __name__ == "__main__":
    build(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")), Path(sys.argv[1]).parent)
