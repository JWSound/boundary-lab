"""Pure helpers used by the main window mixins."""

from __future__ import annotations

from dataclasses import replace

from blab.preview_hierarchy import physical_system_preview_metadata as _physical_system_preview_metadata
from blab.ui.dialogs import (
    MeshDialogEntry,
)

__all__ = ["_physical_system_preview_metadata", "_mesh_entries_with_file_overrides"]


def _mesh_entries_with_file_overrides(
    meshes: tuple[MeshDialogEntry, ...],
    overrides_by_name: dict[str, str],
) -> tuple[MeshDialogEntry, ...]:
    return tuple(replace(mesh, cleaned_file=overrides_by_name.get(mesh.name, mesh.cleaned_file)) for mesh in meshes)
