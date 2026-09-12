"""Geometry preparation service shared by preview and solve workflows."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import meshio
import numpy as np

from blab.ath import read_surface_physical_names
from blab.config import MeshConfig, RadiatorConfig
from blab.exterior_preparation import prepare_exterior_system
from blab.mesh_clean import AREA_TOL, MERGE_TOL, clean_mesh_file
from blab.physical_model import PhysicalSystem
from blab.ui.project_state import ImportedMeshState

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
    source_surface_tags_by_mesh: dict[str, dict[str, int]]
    solver_surface_by_source: dict[tuple[str, int | None], tuple[str, int | None]]

    physical_system: PhysicalSystem | None = None

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
        physical_system: PhysicalSystem | None = None,
    ) -> PreparedMeshAssembly:
        cleaned_imported = self.clean_imported_meshes(imported_meshes)
        imported_configs = tuple(self._imported_mesh_config(mesh) for mesh in cleaned_imported if mesh.enabled)
        candidates = (*generated_mesh_configs, *imported_configs)
        mesh_configs = tuple(candidates)
        resolved_radiators = radiators
        source_surface_tags_by_mesh = {mesh.name: read_surface_physical_names(Path(mesh.file)) for mesh in candidates}
        solver_surface_by_source = self.solver_surface_map(tuple(candidates), stitched=False)
        prepared_system = physical_system
        if stitch_imported_meshes and physical_system is not None:
            configs_by_name = {mesh.name: mesh for mesh in candidates}
            synced = replace(
                physical_system,
                meshes=tuple(
                    replace(
                        resource,
                        file=configs_by_name[resource.name].file,
                        scale_to_m=configs_by_name[resource.name].scale_factor,
                        translation_m=configs_by_name[resource.name].translation_m,
                    )
                    if resource.name in configs_by_name
                    else resource
                    for resource in physical_system.meshes
                ),
            )
            try:
                prepared_system = prepare_exterior_system(
                    synced,
                    stitch_tolerance_mm=stitch_tolerance_mm,
                    symmetry_mode=symmetry,
                    output_root=self.output_root,
                )
            except ValueError as exc:
                raise RuntimeError(STITCH_FAILURE_MESSAGE) from exc
            source_by_id = {mesh.id: mesh for mesh in synced.meshes}
            prepared_by_region = {region.id: region for region in prepared_system.regions}
            prepared_by_id = {mesh.id: mesh for mesh in prepared_system.meshes}
            for entry in prepared_system.metadata.get("exterior_preparation", []):
                target = prepared_by_id[prepared_by_region[entry["region_id"]].mesh_ids[0]]
                tags = read_surface_physical_names(Path(target.file))
                for mapping in entry["surface_map"]:
                    source = source_by_id[mapping["mesh_id"]]
                    source_tag = source_surface_tags_by_mesh[source.name][mapping["source_name"]]
                    solver_surface_by_source[(source.name, source_tag)] = (target.name, tags[mapping["assembled_name"]])
            source_names = {resource.name for resource in synced.meshes}
            mesh_configs = tuple(
                MeshConfig(
                    name=resource.name,
                    file=resource.file,
                    scale_factor=resource.scale_to_m,
                    translation_m=resource.translation_m,
                )
                for resource in prepared_system.meshes
            ) + tuple(mesh for mesh in candidates if mesh.name not in source_names)
            resolved_radiators = tuple(
                replace(radiator, mesh=target[0], tag=target[1])
                if (target := solver_surface_by_source.get((radiator.mesh, radiator.tag)))
                else radiator
                for radiator in radiators
            )
        surface_tags_by_mesh = {mesh.name: read_surface_physical_names(Path(mesh.file)) for mesh in mesh_configs}
        return PreparedMeshAssembly(
            physical_system=prepared_system,
            imported_meshes=cleaned_imported,
            source_mesh_configs=tuple(candidates),
            mesh_configs=mesh_configs,
            radiators=resolved_radiators,
            surface_tags_by_mesh=surface_tags_by_mesh,
            source_surface_tags_by_mesh=source_surface_tags_by_mesh,
            solver_surface_by_source=solver_surface_by_source,
        )

    def cleaned_imported_mesh_path(self, mesh: ImportedMeshState) -> Path:
        source_path = Path(mesh.source_file)
        source_hash = hashlib.sha1(str(source_path.resolve()).encode("utf-8")).hexdigest()[:10]
        safe_name = "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in mesh.name).strip("_")
        return self.output_root / f"{safe_name or 'mesh'}_{source_hash}_clean.msh"

    def _imported_mesh_config(self, mesh: ImportedMeshState) -> MeshConfig:
        mesh_file = mesh.cleaned_file if mesh.cleaned_file and Path(mesh.cleaned_file).exists() else mesh.source_file
        return MeshConfig(
            name=mesh.name,
            file=mesh_file,
            scale_factor=float(mesh.scale_factor),
            translation_m=tuple(value / 1000.0 for value in mesh.translation_mm),
        )

    def stitched_radiator_map(
        self,
        mesh_configs: tuple[MeshConfig, ...],
    ) -> dict[tuple[str | None, int], tuple[str, int]]:
        """Reconstruct old stitched identities only to migrate saved radiator assignments."""
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

    def solver_surface_map(
        self,
        mesh_configs: tuple[MeshConfig, ...],
        *,
        stitched: bool,
    ) -> dict[tuple[str, int | None], tuple[str, int | None]]:
        """Map source physical surfaces to the actor identity in solver geometry."""

        if stitched:
            return {
                source: (STITCHED_MESH_NAME, stitched_surface[1])
                for source, stitched_surface in self.stitched_radiator_map(mesh_configs).items()
            }
        mapping: dict[tuple[str, int | None], tuple[str, int | None]] = {}
        for mesh_config in mesh_configs:
            tags = self.used_surface_tags(mesh_config)
            if tags:
                mapping.update(((mesh_config.name, int(tag)), (mesh_config.name, int(tag))) for tag in tags)
            else:
                mapping[(mesh_config.name, None)] = (mesh_config.name, None)
        return mapping

    @staticmethod
    def used_surface_tags(mesh_config: MeshConfig) -> tuple[int, ...]:
        mesh = meshio.read(mesh_config.file)
        physical = mesh.cell_data_dict.get("gmsh:physical", {})
        triangle_tags = physical.get("triangle")
        if triangle_tags is None:
            triangle_tags = physical.get("triangle3")
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
