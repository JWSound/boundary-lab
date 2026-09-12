"""Qt-free snapshot preparation for desktop mesh previews and inventories."""

from __future__ import annotations

from dataclasses import dataclass

from blab.config import MeshConfig, RadiatorConfig
from blab.generators.base import GeneratedGeometry, GeneratorDocument
from blab.generators.postprocess import ensure_reduced_geometry
from blab.mesh_cache import read_mesh
from blab.mesh_inventory import InventoryEntry, inspect_system_mesh_variants
from blab.mesh_topology import analyze_exterior_mesh_topology
from blab.physical_model import PhysicalSystem
from blab.preview_hierarchy import build_preview_hierarchy, physical_system_preview_metadata
from blab.ui.mesh_assembly import STITCH_FAILURE_MESSAGE, MeshAssemblyService
from blab.ui.project_state import ImportedMeshState, generator_mesh_name


@dataclass
class MeshPreparationSnapshot:
    documents: tuple[GeneratorDocument, ...]
    generated: dict[str, GeneratedGeometry]
    imported: tuple[ImportedMeshState, ...]
    radiators: tuple[RadiatorConfig, ...]
    system: PhysicalSystem | None
    symmetry: str
    stitch: bool
    tolerance_mm: float

    def entries(self, symmetry: str) -> tuple[InventoryEntry, ...]:
        entries = []
        for document in self.documents:
            result = self.generated.get(document.id)
            if result is None:
                continue
            if symmetry != "off" and document.mesh_enabled:
                result = ensure_reduced_geometry(result)
                self.generated[document.id] = result
            entries.append(
                InventoryEntry(
                    name=generator_mesh_name(document),
                    source_file=str(result.solver_mesh_path_for_symmetry(symmetry)),
                    scale_factor=float(document.mesh_scale_factor),
                    translation_mm=document.mesh_translation_mm,
                    enabled=document.mesh_enabled,
                    locked=True,
                )
            )
        entries.extend(
            InventoryEntry(
                name=mesh.name,
                source_file=mesh.source_file,
                cleaned_file=mesh.cleaned_file,
                scale_factor=mesh.scale_factor,
                translation_mm=mesh.translation_mm,
                enabled=mesh.enabled,
            )
            for mesh in self.imported
        )
        return tuple(entries)


def prepare_system_inventory(snapshot: MeshPreparationSnapshot):
    canonical = snapshot.entries("off")
    symmetry = snapshot.entries(snapshot.symmetry)
    return inspect_system_mesh_variants(canonical, symmetry)


def prepare_preview(snapshot: MeshPreparationSnapshot, output_root):
    service = MeshAssemblyService(output_root)
    generated = tuple(
        MeshConfig(
            name=entry.name,
            file=entry.source_file,
            scale_factor=entry.scale_factor,
            translation_m=tuple(value / 1000 for value in entry.translation_mm),
        )
        for entry in snapshot.entries(snapshot.symmetry)
        if entry.locked and entry.enabled
    )
    options = dict(
        generated_mesh_configs=generated,
        imported_meshes=snapshot.imported,
        radiators=snapshot.radiators,
        physical_system=snapshot.system,
        stitch_imported_meshes=snapshot.stitch,
        stitch_tolerance_mm=snapshot.tolerance_mm,
        symmetry=snapshot.symmetry,
    )
    warning = None
    try:
        assembly = service.prepare(**options)
    except RuntimeError as exc:
        if str(exc) != STITCH_FAILURE_MESSAGE or not snapshot.stitch:
            raise
        assembly = service.prepare(**(options | {"stitch_imported_meshes": False}))
        warning = "Mesh preview showing unstitched meshes; stitching failed"
    interfaces, components, regions, interior = physical_system_preview_metadata(
        assembly.physical_system or snapshot.system,
        assembly.surface_tags_by_mesh,
    )
    exterior = tuple(mesh for mesh in assembly.mesh_configs if not interior or regions.get(mesh.name) == "exterior")
    try:
        topology = analyze_exterior_mesh_topology(exterior, symmetry=snapshot.symmetry) if exterior else None
    except (OSError, ValueError):
        topology = None
    hierarchy = build_preview_hierarchy(
        snapshot.system,
        source_mesh_configs=assembly.source_mesh_configs,
        source_surface_tags_by_mesh=assembly.source_surface_tags_by_mesh,
        solver_surface_by_source=assembly.solver_surface_by_source,
    )
    view_options = dict(
        driven_surfaces={(r.mesh, r.tag) for r in assembly.radiators} | components,
        surface_tags_by_mesh=assembly.surface_tags_by_mesh,
        interface_surfaces=interfaces,
        mesh_regions=regions,
        symmetry=snapshot.symmetry,
        topology_report=topology,
        hierarchy=hierarchy,
        loaded_meshes={mesh.name: read_mesh(mesh.file) for mesh in assembly.mesh_configs},
    )
    return assembly, snapshot.generated, view_options, warning
