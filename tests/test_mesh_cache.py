from dataclasses import replace
from pathlib import Path
from shutil import copyfile

import numpy as np
import pytest

import blab.mesh_cache as cache_module
import blab.mesh_inventory as inventory_module
from blab.mesh_cache import MeshCache
from blab.mesh_inventory import InventoryEntry, inspect_system_mesh_variants


def test_cached_meshes_are_isolated_and_replacement_invalidates(tmp_path, monkeypatch):
    path = tmp_path / "mesh.msh"
    copyfile(Path(__file__).parent / "fixtures" / "exterior.msh", path)
    cache = MeshCache()
    read = cache_module.meshio.read
    calls = []

    def counted(path):
        calls.append(path)
        return read(path)

    monkeypatch.setattr(cache_module.meshio, "read", counted)
    first = cache.read(path)
    expected = first.points.copy()
    first.points[:] = 123
    first.field_data.clear()
    second = cache.read(path)
    np.testing.assert_array_equal(second.points, expected)
    assert second.field_data
    assert len(calls) == 1
    replacement = tmp_path / "new.msh"
    copyfile(Path(__file__).parent / "fixtures" / "exterior_conforming.msh", replacement)
    replacement.replace(path)
    cache.read(path)
    assert len(calls) == 2
    cache.invalidate(path)
    cache.read(path)
    assert len(calls) == 3
    path.unlink()
    with pytest.raises(FileNotFoundError):
        cache.read(path)


def test_cache_is_bounded_and_changed_during_read_is_not_cached(tmp_path, monkeypatch):
    fixture = Path(__file__).parent / "fixtures" / "exterior.msh"
    a, b = tmp_path / "a.msh", tmp_path / "b.msh"
    copyfile(fixture, a)
    copyfile(fixture, b)
    cache = MeshCache(max_entries=1)
    cache.read(a)
    cache.read(b)
    assert len(cache._entries) == 1
    assert next(iter(cache._entries))[0] == str(b.resolve())
    cache.invalidate()
    read = cache_module.meshio.read

    def unstable(path):
        mesh = read(path)
        with Path(path).open("a") as stream:
            stream.write("\n")
        return mesh

    monkeypatch.setattr(cache_module.meshio, "read", unstable)
    with pytest.raises(OSError, match="changed while being read"):
        cache.read(a)
    assert not cache._entries


def test_symmetry_variants_share_imported_inventory_and_keep_transforms(monkeypatch):
    cache = MeshCache()
    monkeypatch.setattr(inventory_module, "mesh_cache", cache)
    fixture = Path(__file__).parent / "fixtures"
    imported = InventoryEntry("volume", str(fixture / "femvolume.msh"))
    generated = InventoryEntry("waveguide", str(fixture / "exterior.msh"), locked=True)
    reduced = replace(generated, source_file=str(fixture / "exterior_conforming.msh"))
    calls = []
    inspect = inventory_module.inspect_system_meshes

    def counted(entries):
        calls.append(entries)
        return inspect(entries)

    monkeypatch.setattr(inventory_module, "inspect_system_meshes", counted)
    canonical, symmetry = inspect_system_mesh_variants((generated, imported), (reduced, imported))
    assert calls == [(generated, imported), (reduced,)]
    assert canonical[1] is symmetry[1]
    assert canonical[1].has_tetrahedra
    ownership = cache.inventory(imported.source_file)
    assert cache.inventory(imported.source_file) is ownership
    transformed = inspect((replace(imported, scale_factor=0.002, translation_mm=(10, 0, 0)),))[0]
    assert transformed.scale_to_m == 0.002
    assert transformed.translation_m == (0.01, 0.0, 0.0)
    assert transformed.surface_groups_by_volume == canonical[1].surface_groups_by_volume
