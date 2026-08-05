"""Geometry preparation service shared by preview and solve workflows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import meshio
import numpy as np

from blab.ath import read_surface_physical_names
from blab.config import MeshConfig, RadiatorConfig
from blab.mesh_clean import AREA_TOL, MERGE_TOL, clean_mesh_file, stitch_meshes
from blab.ui.project_state import DEFAULT_MESH_SCALE_FACTOR, ImportedMeshState

STITCHED_MESH_NAME = "stitched"
STITCH_FAILURE_MESSAGE = (
    "Error - unable to stitch separate mesh entities. "
    "Refer to help documentation for more info on multi-mesh workflows."
)


@dataclass(frozen=True)
class PreparedMeshAssembly:
    imported_meshes: tuple[ImportedMeshState, ...]
    source_mesh_configs: tuple[MeshConfig, ...]
    mesh_configs: tuple[MeshConfig, ...]
    radiators: tuple[RadiatorConfig, ...]
    surface_tags_by_mesh: dict[str, dict[str, int]]

    @property
    def surface_tags(self) -> dict[str, tuple[str, int]]:
        return {
            f"{mesh_name}:{surface_name}": (mesh_name, tag)
            for mesh_name, names in self.surface_tags_by_mesh.items()
            for surface_name, tag in names.items()
        }


class MeshAssemblyService:
    """Materialize cleaned/stitched meshes without depending on Qt widgets."""

    def __init__(self, output_root: str | Path):
        self.output_root = Path(output_root)

    def clean_imported_meshes(
        self,
        meshes: tuple[ImportedMeshState, ...],
    ) -> tuple[ImportedMeshState, ...]:
        cleaned_meshes = []
        for mesh in meshes:
            if not mesh.enabled:
                cleaned_meshes.append(mesh)
                continue
            source_path = Path(mesh.source_file)
            if source_path.suffix.lower() != ".msh":
                raise ValueError(f"Only .msh mesh files can be imported: {source_path}")
            if not source_path.exists():
                raise FileNotFoundError(f"Imported mesh not found: {source_path}")

            if self.is_volume_mesh(source_path):
                # The legacy cleaner is intentionally a surface-mesh operation.
                # Preserve FEM volume connectivity and all embedded physical groups.
                cleaned_meshes.append(replace(mesh, cleaned_file=None))
                continue

            cleaned_path = Path(mesh.cleaned_file) if mesh.cleaned_file else self.cleaned_imported_mesh_path(mesh)
            if not cleaned_path.exists() or source_path.stat().st_mtime > cleaned_path.stat().st_mtime:
                cleaned_path.parent.mkdir(parents=True, exist_ok=True)
                clean_mesh_file(
                    str(source_path),
                    str(cleaned_path),
                    merge_tol=MERGE_TOL,
                    area_tol=AREA_TOL,
                    mirror_x=False,
                    binary=False,
                )
            cleaned_meshes.append(replace(mesh, cleaned_file=str(cleaned_path)))
        return tuple(cleaned_meshes)

    @staticmethod
    def is_volume_mesh(path: str | Path) -> bool:
        mesh = meshio.read(Path(path))
        return any(block.type in {"tetra", "tetra4"} and len(block.data) for block in mesh.cells)

    def prepare(
        self,
        *,
        generated_mesh_configs: tuple[MeshConfig, ...],
        imported_meshes: tuple[ImportedMeshState, ...],
        radiators: tuple[RadiatorConfig, ...],
        stitch_imported_meshes: bool,
        stitch_tolerance_mm: float,
        symmetry: str,
    ) -> PreparedMeshAssembly:
        cleaned_imported = self.clean_imported_meshes(imported_meshes)
        imported_configs = tuple(self._imported_mesh_config(mesh) for mesh in cleaned_imported if mesh.enabled)
        candidates = (*generated_mesh_configs, *imported_configs)
        mesh_configs, resolved_radiators = self.prepare_mesh_configs(
            tuple(candidates),
            radiators,
            stitch_meshes_enabled=stitch_imported_meshes,
            stitch_tolerance_mm=stitch_tolerance_mm,
            symmetry=symmetry,
        )
        surface_tags_by_mesh = {mesh.name: read_surface_physical_names(Path(mesh.file)) for mesh in mesh_configs}
        return PreparedMeshAssembly(
            imported_meshes=cleaned_imported,
            source_mesh_configs=tuple(candidates),
            mesh_configs=mesh_configs,
            radiators=resolved_radiators,
            surface_tags_by_mesh=surface_tags_by_mesh,
        )

    def prepare_mesh_configs(
        self,
        mesh_configs: tuple[MeshConfig, ...],
        radiators: tuple[RadiatorConfig, ...],
        *,
        stitch_meshes_enabled: bool,
        stitch_tolerance_mm: float,
        symmetry: str,
    ) -> tuple[tuple[MeshConfig, ...], tuple[RadiatorConfig, ...]]:
        """Apply optional region assembly to already materialized mesh resources."""

        candidates = tuple(mesh_configs)
        if stitch_meshes_enabled and len(candidates) > 1:
            stitched = self._stitched_mesh_config(candidates, stitch_tolerance_mm, symmetry)
            resolved_meshes = (stitched,)
            resolved_radiators = self.radiators_for_stitched_mesh(candidates, radiators)
        else:
            resolved_meshes = candidates
            resolved_radiators = radiators
        return resolved_meshes, resolved_radiators

    def cleaned_imported_mesh_path(self, mesh: ImportedMeshState) -> Path:
        source_path = Path(mesh.source_file)
        source_hash = hashlib.sha1(str(source_path.resolve()).encode("utf-8")).hexdigest()[:10]
        safe_name = "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in mesh.name).strip("_")
        return self.output_root / f"{safe_name or 'mesh'}_{source_hash}_clean.msh"

    def radiators_for_stitched_mesh(
        self,
        source_mesh_configs: tuple[MeshConfig, ...],
        radiators: tuple[RadiatorConfig, ...],
    ) -> tuple[RadiatorConfig, ...]:
        stitched_map = self.stitched_radiator_map(source_mesh_configs)
        resolved = []
        for radiator in radiators:
            stitched_surface = stitched_map.get((radiator.mesh, radiator.tag))
            if stitched_surface is None:
                resolved.append(replace(radiator, mesh=STITCHED_MESH_NAME))
                continue
            stitched_name, stitched_tag = stitched_surface
            resolved.append(
                replace(
                    radiator,
                    name=stitched_name,
                    mesh=STITCHED_MESH_NAME,
                    tag=stitched_tag,
                )
            )
        return tuple(resolved)

    def _imported_mesh_config(self, mesh: ImportedMeshState) -> MeshConfig:
        mesh_file = mesh.cleaned_file if mesh.cleaned_file and Path(mesh.cleaned_file).exists() else mesh.source_file
        return MeshConfig(
            name=mesh.name,
            file=mesh_file,
            scale_factor=float(mesh.scale_factor),
            translation_m=tuple(value / 1000.0 for value in mesh.translation_mm),
        )

    def _stitched_mesh_config(
        self,
        mesh_configs: tuple[MeshConfig, ...],
        stitch_tolerance_mm: float,
        symmetry: str,
    ) -> MeshConfig:
        stitched_path = self.stitched_mesh_path(mesh_configs, stitch_tolerance_mm, symmetry)
        if not stitched_path.exists():
            stitched_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                stitched_mesh, _result = stitch_meshes(
                    tuple(self.mesh_for_stitching(mesh) for mesh in mesh_configs),
                    stitch_tol=float(stitch_tolerance_mm),
                    area_tol=AREA_TOL,
                    ignored_boundary_axes=self.ignored_boundary_axes(symmetry),
                )
                meshio.write(stitched_path, stitched_mesh, file_format="gmsh22", binary=False)
            except Exception as exc:
                raise RuntimeError(STITCH_FAILURE_MESSAGE) from exc
        return MeshConfig(name=STITCHED_MESH_NAME, file=str(stitched_path), scale_factor=DEFAULT_MESH_SCALE_FACTOR)

    def stitched_mesh_path(
        self,
        mesh_configs: tuple[MeshConfig, ...],
        stitch_tolerance_mm: float,
        symmetry: str,
    ) -> Path:
        payload = {
            "symmetry": symmetry,
            "ignored_boundary_axes": self.ignored_boundary_axes(symmetry),
            "tol_mm": round(float(stitch_tolerance_mm), 6),
            "meshes": [
                {
                    "name": mesh.name,
                    "file": str(Path(mesh.file).resolve()),
                    "mtime_ns": Path(mesh.file).stat().st_mtime_ns,
                    "translation_m": mesh.translation_m,
                    "scale_factor": mesh.scale_factor,
                }
                for mesh in mesh_configs
            ],
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        return self.output_root / f"stitched_{digest}.msh"

    @staticmethod
    def mesh_for_stitching(mesh_config: MeshConfig) -> meshio.Mesh:
        mesh = meshio.read(mesh_config.file)
        scale_factor = (
            DEFAULT_MESH_SCALE_FACTOR if mesh_config.scale_factor is None else float(mesh_config.scale_factor)
        )
        points_m = np.asarray(mesh.points, dtype=float) * scale_factor + np.asarray(
            mesh_config.translation_m, dtype=float
        )
        return meshio.Mesh(
            points=points_m / DEFAULT_MESH_SCALE_FACTOR,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            field_data=mesh.field_data,
        )

    @staticmethod
    def ignored_boundary_axes(symmetry: str) -> tuple[str, ...]:
        if symmetry == "x":
            return ("x",)
        if symmetry == "xy":
            return ("x", "y")
        return ()

    def stitched_radiator_map(
        self,
        mesh_configs: tuple[MeshConfig, ...],
    ) -> dict[tuple[str | None, int], tuple[str, int]]:
        mapping: dict[tuple[str | None, int], tuple[str, int]] = {}
        used_surface_names: set[str] = set()
        used_surface_tags: set[int] = set()
        next_surface_tag = 1
        for mesh_index, mesh_config in enumerate(mesh_configs):
            names_by_tag = {tag: name for name, tag in read_surface_physical_names(Path(mesh_config.file)).items()}
            for old_tag in self.used_surface_tags(mesh_config):
                surface_name = names_by_tag.get(old_tag, f"mesh{mesh_index + 1}_surface_{old_tag}")
                stitched_name = self.unique_surface_name(surface_name, used_surface_names, mesh_index)
                used_surface_names.add(stitched_name)
                if old_tag not in used_surface_tags:
                    new_tag = old_tag
                else:
                    while next_surface_tag in used_surface_tags:
                        next_surface_tag += 1
                    new_tag = next_surface_tag
                used_surface_tags.add(new_tag)
                mapping[(mesh_config.name, old_tag)] = (f"{STITCHED_MESH_NAME}:{stitched_name}", new_tag)
        return mapping

    @staticmethod
    def used_surface_tags(mesh_config: MeshConfig) -> tuple[int, ...]:
        mesh = meshio.read(mesh_config.file)
        triangle_tags = mesh.cell_data_dict.get("gmsh:physical", {}).get("triangle")
        if triangle_tags is None:
            return ()
        return tuple(sorted(int(tag) for tag in np.unique(triangle_tags)))

    @staticmethod
    def unique_surface_name(surface_name: str, used: set[str], mesh_index: int) -> str:
        if surface_name not in used:
            return surface_name
        suffix = 2
        candidate = f"{surface_name}_mesh{mesh_index + 1}"
        while candidate in used:
            candidate = f"{surface_name}_mesh{mesh_index + 1}_{suffix}"
            suffix += 1
        return candidate
