"""Immutable, file-independent mesh snapshots and their packed wire format."""

from __future__ import annotations

import base64
import hashlib
import json
from types import MappingProxyType

import meshio
import numpy as np

CELL_WIDTHS = {"triangle": 3, "triangle6": 6, "tetra": 4, "tetra10": 10}


def _array(values, dtype, shape_tail):
    array = np.asarray(values)
    if array.ndim != 1 + len(shape_tail) or array.shape[1:] != shape_tail:
        raise ValueError(f"Mesh array must have shape (N, {shape_tail}).")
    if np.dtype(dtype).kind == "i" and array.dtype.kind not in "iu":
        raise ValueError("Connectivity and physical tags must be integer arrays.")
    if array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError("Mesh arrays must contain finite numeric values.")
    # Bytes-backed arrays cannot be made writable, even by a retained alias.
    return np.frombuffer(array.astype(dtype).tobytes(order="C"), dtype=dtype).reshape(array.shape)


def _packed(array):
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


class MeshData:
    """Coordinates, ordered cell blocks and Gmsh physical groups; zero-based indices.

    Units/transforms live on the mesh resource. ``from_meshio`` accepts only the
    acoustic element types supported here and never drops unsupported cells.
    """

    __slots__ = ("_points", "_cells", "_physical_tags", "_physical_names", "_digest")

    def __init__(self, *, points, cells, physical_tags, physical_names):
        points = _array(points, "<f8", (3,))
        if not len(points):
            raise ValueError("A mesh requires vertices.")
        if len(cells) != len(physical_tags) or not cells:
            raise ValueError("Each cell block requires physical tags.")
        blocks, tags = [], []
        for (kind, indices), block_tags in zip(cells, physical_tags, strict=True):
            if not isinstance(kind, str) or kind not in CELL_WIDTHS:
                raise ValueError(f"Unsupported in-memory element type: {kind}")
            indices = _array(indices, "<i8", (CELL_WIDTHS[kind],))
            block_tags = _array(block_tags, "<i8", ())
            if not len(indices) or len(indices) != len(block_tags):
                raise ValueError("Cell blocks must be nonempty and match their tag counts.")
            if np.any(indices < 0) or np.any(indices >= len(points)) or np.any(block_tags <= 0):
                raise ValueError("Invalid vertex index or nonpositive physical tag.")
            if any(len(set(row)) != len(row) for row in indices):
                raise ValueError("An element cannot reference a vertex more than once.")
            blocks.append((kind, indices))
            tags.append(block_tags)
        names = {}
        for name, value in physical_names.items():
            if not isinstance(name, str) or not name or len(value) != 2:
                raise ValueError("Physical groups require a name, positive tag and dimension.")
            tag, dimension = value
            if not isinstance(tag, (int, np.integer)) or not isinstance(dimension, (int, np.integer)):
                raise ValueError("Physical group tags and dimensions must be integers.")
            if (
                isinstance(tag, (bool, np.bool_))
                or isinstance(dimension, (bool, np.bool_))
                or tag <= 0
                or dimension not in (2, 3)
            ):
                raise ValueError("Invalid physical group tag or dimension.")
            names[name] = (int(tag), int(dimension))
        for (kind, _), block_tags in zip(blocks, tags, strict=True):
            dimension = 3 if kind.startswith("tetra") else 2
            named = {tag for tag, dim in names.values() if dim == dimension}
            if not set(map(int, block_tags)) <= named:
                raise ValueError("Every physical tag must have a name in its element dimension.")
        object.__setattr__(self, "_points", points)
        object.__setattr__(self, "_cells", tuple(blocks))
        object.__setattr__(self, "_physical_tags", tuple(tags))
        object.__setattr__(self, "_physical_names", MappingProxyType(names))
        object.__setattr__(
            self, "_digest", hashlib.sha256(json.dumps(self.to_payload(), sort_keys=True).encode()).hexdigest()
        )

    def __setattr__(self, name, value):
        raise AttributeError("MeshData is immutable.")

    def __deepcopy__(self, memo):
        return self

    @property
    def points(self):
        return self._points.view()

    @property
    def cells(self):
        return tuple((kind, values.view()) for kind, values in self._cells)

    @property
    def physical_tags(self):
        return tuple(values.view() for values in self._physical_tags)

    @property
    def physical_names(self):
        return self._physical_names

    @property
    def digest(self):
        return self._digest

    def to_meshio(self, *, copy=True):
        take = (lambda value: value.copy()) if copy else (lambda value: value)
        return meshio.Mesh(
            take(self.points),
            [(kind, take(values)) for kind, values in self.cells],
            # CAD entity IDs are not part of the acoustic contract. The legacy
            # conformer carries these labels through; synthesize one per group.
            cell_data={
                "gmsh:physical": [take(tags) for tags in self.physical_tags],
                "gmsh:geometrical": [take(tags) for tags in self.physical_tags],
            },
            field_data={name: np.array(value) for name, value in self.physical_names.items()},
        )

    @classmethod
    def from_meshio(cls, mesh):
        return cls(
            points=mesh.points,
            cells=[(block.type, block.data) for block in mesh.cells],
            physical_tags=mesh.cell_data.get("gmsh:physical", ()),
            physical_names=mesh.field_data,
        )

    def to_payload(self):
        return {
            "schema_version": 1,
            "points": _packed(self.points),
            "cells": [
                {"type": kind, "connectivity": _packed(indices), "physical_tags": _packed(tags)}
                for (kind, indices), tags in zip(self.cells, self.physical_tags, strict=True)
            ],
            "physical_names": {name: list(value) for name, value in self.physical_names.items()},
        }

    @classmethod
    def from_payload(cls, payload):
        from beat_engine.beat_contract.mesh import validate_mesh_data

        validate_mesh_data(payload)

        def unpack(value):
            return np.frombuffer(base64.b64decode(value["data"], validate=True), dtype=value["dtype"]).reshape(
                value["shape"]
            )

        return cls(
            points=unpack(payload["points"]),
            cells=[(block["type"], unpack(block["connectivity"])) for block in payload["cells"]],
            physical_tags=[unpack(block["physical_tags"]) for block in payload["cells"]],
            physical_names=payload["physical_names"],
        )


def read_resource_mesh(resource):
    """Read a file-backed resource or detach mutable working arrays from a snapshot."""
    data = getattr(resource, "mesh_data", None)
    if data is not None:
        return data.to_meshio()
    from blab.mesh_cache import read_mesh

    return read_mesh(resource.file)


def surface_names(resource):
    mesh = read_resource_mesh(resource)
    return {name: int(value[0]) for name, value in mesh.field_data.items() if int(value[1]) == 2}
