"""Physical-group inventory shared by headless and desktop preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from blab.mesh_cache import mesh_cache


class MeshEntry(Protocol):
    name: str
    source_file: str
    cleaned_file: str | None
    scale_factor: float
    translation_mm: tuple[float, float, float]
    enabled: bool
    locked: bool


@dataclass(frozen=True)
class InventoryEntry:
    name: str
    source_file: str
    cleaned_file: str | None = None
    scale_factor: float = 0.001
    translation_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    enabled: bool = True
    locked: bool = False


@dataclass(frozen=True)
class AvailableSystemMesh:
    """An enabled application mesh available to the physical-system editor."""

    name: str
    source_file: str
    file: str
    scale_to_m: float
    translation_m: tuple[float, float, float]
    surface_groups: tuple[str, ...]
    volume_groups: tuple[str, ...]
    has_tetrahedra: bool
    surface_groups_by_volume: tuple[tuple[str, tuple[str, ...]], ...] = ()
    locked: bool = False

    def surface_groups_for_volume(self, volume_group: str | None) -> tuple[str, ...]:
        if volume_group is None:
            return self.surface_groups
        return dict(self.surface_groups_by_volume).get(volume_group, ())


def inspect_system_meshes(meshes: tuple[MeshEntry, ...]) -> tuple[AvailableSystemMesh, ...]:
    """Read physical-group inventory without modifying imported mesh files."""

    inspected = []
    for entry in meshes:
        if not entry.enabled:
            continue
        source_path = Path(entry.source_file)
        effective_path = (
            Path(entry.cleaned_file)
            if entry.cleaned_file is not None and Path(entry.cleaned_file).is_file()
            else source_path
        )
        inventory = mesh_cache.inventory(effective_path)
        inspected.append(
            AvailableSystemMesh(
                name=entry.name,
                source_file=str(source_path),
                file=str(effective_path),
                scale_to_m=float(entry.scale_factor),
                translation_m=tuple(float(value) / 1000.0 for value in entry.translation_mm),
                surface_groups=tuple(name for name, _tag in inventory.surfaces),
                volume_groups=tuple(name for name, _tag in inventory.volumes),
                has_tetrahedra=inventory.has_tetrahedra,
                surface_groups_by_volume=inventory.surfaces_by_volume,
                locked=bool(entry.locked),
            )
        )
    return tuple(inspected)


def inspect_system_mesh_variants(
    mesh_entries: tuple[MeshEntry, ...],
    symmetry_mesh_entries: tuple[MeshEntry, ...],
) -> tuple[tuple[AvailableSystemMesh, ...], tuple[AvailableSystemMesh, ...]]:
    """Inspect canonical and symmetry meshes without rereading identical inputs."""

    meshes = inspect_system_meshes(mesh_entries)
    if symmetry_mesh_entries == mesh_entries:
        return meshes, meshes
    by_entry = dict(zip((entry for entry in mesh_entries if entry.enabled), meshes, strict=True))
    missing = tuple(dict.fromkeys(entry for entry in symmetry_mesh_entries if entry.enabled and entry not in by_entry))
    by_entry.update(zip(missing, inspect_system_meshes(missing), strict=True))
    symmetry_meshes = tuple(by_entry[entry] for entry in symmetry_mesh_entries if entry.enabled)
    return meshes, symmetry_meshes
