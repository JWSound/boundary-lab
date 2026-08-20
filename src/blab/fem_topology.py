"""Topology helpers for physical-volume FEM boundary ownership."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import meshio
import numpy as np

_TETRAHEDRON_TYPES = {"tetra", "tetra4", "tetra10"}
_TRIANGLE_TYPES = {"triangle", "triangle3", "triangle6"}


def selected_volume_surface_tags(
    mesh: meshio.Mesh,
    selected_volume_tags: Iterable[int],
) -> frozenset[int]:
    """Return tagged triangle groups on the boundary of selected tetrahedra.

    The operation mirrors the Julia FEM volume restriction: faces shared by
    two selected tetrahedra are internal, while faces occurring once are
    retained only when an explicit physical triangle exists on that face.
    """

    selected_tags = {int(tag) for tag in selected_volume_tags}
    if not selected_tags:
        raise ValueError("Select at least one physical FEM volume tag.")
    physical_blocks = mesh.cell_data.get("gmsh:physical")
    if physical_blocks is None or len(physical_blocks) != len(mesh.cells):
        raise ValueError("FEM mesh elements do not contain aligned gmsh:physical tags.")

    face_counts: Counter[tuple[int, int, int]] = Counter()
    selected_tetrahedron_count = 0
    for block, raw_tags in zip(mesh.cells, physical_blocks, strict=True):
        if block.type not in _TETRAHEDRON_TYPES:
            continue
        tetrahedra = np.asarray(block.data, dtype=np.int64)
        tags = np.asarray(raw_tags, dtype=np.int64)
        if tetrahedra.ndim != 2 or tetrahedra.shape[1] < 4 or tags.shape != (tetrahedra.shape[0],):
            raise ValueError("FEM mesh physical-volume tags do not align with tetrahedra.")
        for tetrahedron in tetrahedra[np.isin(tags, tuple(selected_tags)), :4]:
            selected_tetrahedron_count += 1
            a, b, c, d = map(int, tetrahedron)
            face_counts.update(
                (
                    tuple(sorted((a, b, c))),
                    tuple(sorted((a, b, d))),
                    tuple(sorted((a, c, d))),
                    tuple(sorted((b, c, d))),
                )
            )
    if selected_tetrahedron_count == 0:
        requested = ", ".join(map(str, sorted(selected_tags)))
        raise ValueError(f"Selected FEM volume tags contain no tetrahedra: {requested}.")

    exterior_faces = {face for face, count in face_counts.items() if count == 1}
    surface_tags: set[int] = set()
    for block, raw_tags in zip(mesh.cells, physical_blocks, strict=True):
        if block.type not in _TRIANGLE_TYPES:
            continue
        triangles = np.asarray(block.data, dtype=np.int64)
        tags = np.asarray(raw_tags, dtype=np.int64)
        if triangles.ndim != 2 or triangles.shape[1] < 3 or tags.shape != (triangles.shape[0],):
            raise ValueError("FEM mesh physical-surface tags do not align with triangles.")
        for triangle, tag in zip(triangles[:, :3], tags, strict=True):
            if tuple(sorted(map(int, triangle))) in exterior_faces:
                surface_tags.add(int(tag))
    return frozenset(surface_tags)


__all__ = ["selected_volume_surface_tags"]
