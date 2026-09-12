"""Bounded, thread-safe mesh reads shared by preparation and presentation.

Cached meshes never escape: callers receive independent copies because mesh
transforms and interface conformance mutate arrays and physical groups.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import meshio
import numpy as np

from blab.fem_topology import selected_volume_surface_tags


@dataclass(frozen=True)
class MeshInventory:
    surfaces: tuple[tuple[str, int], ...]
    volumes: tuple[tuple[str, int], ...]
    has_tetrahedra: bool
    surfaces_by_volume: tuple[tuple[str, tuple[str, ...]], ...]


class MeshCache:
    def __init__(self, *, max_bytes: int = 128 * 1024 * 1024, max_entries: int = 32):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = RLock()

    @staticmethod
    def fingerprint(path: str | Path) -> tuple:
        path = Path(path).resolve()
        stat = path.stat()
        return (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)

    def invalidate(self, path: str | Path | None = None) -> None:
        with self._lock:
            name = None if path is None else str(Path(path).resolve())
            for key in list(self._entries):
                if name is None or key[0] == name:
                    self._bytes -= self._entries.pop(key)[1]

    def _entry(self, path):
        key = self.fingerprint(path)
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        self.invalidate(path)
        mesh = meshio.read(path)
        if self.fingerprint(path) != key:
            raise OSError(f"Mesh changed while being read: {path}. Retry after the file has finished updating.")
        arrays = [mesh.points, *(block.data for block in mesh.cells), *mesh.point_data.values()]
        arrays.extend(array for blocks in mesh.cell_data.values() for array in blocks)
        # Include headroom for cell blocks, physical groups and Python objects.
        size = sum(np.asarray(array).nbytes for array in arrays) + 1024 * (1 + len(mesh.cells))
        entry = [mesh, size, None]
        if size <= self.max_bytes and self.max_entries > 0:
            self._entries[key] = entry
            self._bytes += size
            while self._bytes > self.max_bytes or len(self._entries) > self.max_entries:
                self._bytes -= self._entries.popitem(last=False)[1][1]
        return entry

    def read(self, path: str | Path) -> meshio.Mesh:
        with self._lock:
            return deepcopy(self._entry(path)[0])

    def inventory(self, path: str | Path) -> MeshInventory:
        with self._lock:
            entry = self._entry(path)
            if entry[2] is None:
                mesh = entry[0]
                surfaces = tuple(
                    sorted((str(name), int(raw[0])) for name, raw in mesh.field_data.items() if int(raw[1]) == 2)
                )
                volumes = tuple(
                    sorted((str(name), int(raw[0])) for name, raw in mesh.field_data.items() if int(raw[1]) == 3)
                )
                has_tetrahedra = any(block.type in {"tetra", "tetra4"} and len(block.data) for block in mesh.cells)
                names = {tag: name for name, tag in surfaces}
                ownership = (
                    tuple(
                        (
                            name,
                            tuple(
                                sorted(
                                    names[tag]
                                    for tag in selected_volume_surface_tags(mesh, (volume_tag,))
                                    if tag in names
                                )
                            ),
                        )
                        for name, volume_tag in volumes
                    )
                    if has_tetrahedra
                    else ()
                )
                entry[2] = MeshInventory(surfaces, volumes, has_tetrahedra, ownership)
            return entry[2]


mesh_cache = MeshCache()
read_mesh = mesh_cache.read
