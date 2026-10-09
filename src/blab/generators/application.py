"""Stage provider configuration against generated artifacts without GUI state."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace

from blab.config import RadiatorConfig
from blab.generators.base import GeneratedGeometry, GenerationCompleted, generated_mesh_id
from blab.generators.configuration import _check_references, apply_configuration_patch, project_revision
from blab.generators.resources import generated_mesh_entries, generated_mesh_name
from blab.mesh_data import read_resource_mesh
from blab.mesh_inventory import InventoryEntry, inspect_system_meshes
from blab.physical_compiler import PhysicalSystemCompiler
from blab.physical_model import MeshPurpose, MeshResource, PhysicalSystem, infer_physical_solve_kind
from blab.project.migration import seed_exterior_system
from blab.project.model import ProjectDocument, generator_mesh_name, replace_generator_document


def stage_generation(
    project: ProjectDocument,
    generated: Mapping[str, GeneratedGeometry],
    completed: GenerationCompleted,
    *,
    imported_radiators: tuple[RadiatorConfig, ...] = (),
) -> ProjectDocument:
    """Return a detached candidate with the new artifact and merged configuration.

    Callers commit both the returned project and the generated artifact together.
    Solver completeness is checked by normal solve preparation; this boundary
    checks assignments, ownership and surviving mesh-group references.
    """
    request, geometry = completed.request, completed.result
    if request.project_revision is not None and request.project_revision != project_revision(project):
        raise ValueError("Generation discarded because project inputs changed.")
    document = next((item for item in project.generator_documents if item.id == request.document_id), None)
    if document is None or document.provider_id != request.provider_id or document.source != request.source:
        raise ValueError("Generation discarded because the design changed or was removed.")
    if geometry.meshes:
        return _stage_assembly(project, generated, completed, document, imported_radiators)
    if document.artifact is not None and document.artifact.meshes:
        raise ValueError("An assembly design cannot be replaced by a legacy single-mesh response.")
    candidate = deepcopy(project)
    candidate.generator_documents = replace_generator_document(
        candidate.generator_documents,
        document.id,
        artifact=geometry.to_reference(),
    )
    all_geometry = dict(generated) | {document.id: geometry}
    if candidate.physical_system is None:
        entries, radiators = [], list(imported_radiators)
        for item in candidate.generator_documents:
            result = all_geometry.get(item.id)
            if result is None or not item.mesh_enabled:
                continue
            name = generator_mesh_name(item)
            entries.append(
                InventoryEntry(
                    name=name,
                    source_file="" if result.mesh_data is not None else str(result.solver_mesh_path),
                    scale_factor=item.mesh_scale_factor,
                    translation_mm=item.mesh_translation_mm,
                    locked=True,
                    mesh_data=result.mesh_data,
                )
            )
            radiators.extend(replace(radiator, mesh=name) for radiator in result.radiators)
        entries.extend(
            InventoryEntry(
                name=item.name,
                source_file=item.source_file,
                cleaned_file=item.cleaned_file,
                scale_factor=item.scale_factor,
                translation_mm=item.translation_mm,
                enabled=item.enabled,
            )
            for item in candidate.imported_meshes
        )
        candidate.physical_system, candidate.component_channel_by_id = seed_exterior_system(
            inspect_system_meshes(tuple(entries)),
            tuple(radiators),
        )
    name = generator_mesh_name(document)
    mesh_ids = {item.id for item in candidate.physical_system.meshes if item.name == name}
    if not mesh_ids:
        raise ValueError("Generated mesh is not assigned to the physical system; configure its mesh resource first.")
    candidate.physical_system = replace(
        candidate.physical_system,
        meshes=tuple(
            replace(
                item,
                file="" if geometry.mesh_data is not None else str(geometry.solver_mesh_path),
                mesh_data=geometry.mesh_data,
                scale_to_m=document.mesh_scale_factor,
                translation_m=tuple(value / 1000 for value in document.mesh_translation_mm),
            )
            if item.id in mesh_ids
            else item
            for item in candidate.physical_system.meshes
        ),
    )
    candidate = apply_configuration_patch(
        candidate,
        completed.configuration_patch,
        document_id=document.id,
        mesh_ids=mesh_ids,
    )
    # Regeneration must not silently redirect an old assignment to another tag.
    resource = next(item for item in candidate.physical_system.meshes if item.id in mesh_ids)
    mesh = read_resource_mesh(resource)
    groups = {(str(group_name), int(value[0]), int(value[1])) for group_name, value in mesh.field_data.items()}
    refs = [boundary.group for boundary in candidate.physical_system.boundaries]
    refs.extend(group for region in candidate.physical_system.regions for group in region.volume_groups)
    for group in refs:
        if group.mesh_id not in mesh_ids:
            continue
        if not any(
            dimension == group.dimension
            and (group.name is None or group.name == group_name)
            and (group.tag is None or group.tag == tag)
            for group_name, tag, dimension in groups
        ):
            raise ValueError(f"Generated mesh no longer contains assigned physical group {group.name or group.tag!r}.")
    return candidate


def _stage_assembly(project, generated, completed, document, imported_radiators):
    geometry = completed.result
    if project.symmetry != "off":
        raise ValueError("Generated mesh assemblies currently require symmetry Off.")
    candidate = deepcopy(project)
    old_meshes = document.artifact.meshes if document.artifact else ()
    if document.artifact is not None and not old_meshes:
        raise ValueError("A legacy single-mesh design cannot be replaced by an assembly response.")
    old_ids = {generated_mesh_id(document.id, mesh.id) for mesh in old_meshes}
    new_ids = {generated_mesh_id(document.id, mesh.id) for mesh in geometry.meshes}
    if candidate.physical_system is None:
        # Keep existing legacy designs/imports when an assembly is first added.
        entries = [
            entry
            for item in candidate.generator_documents
            if item.id != document.id and item.mesh_enabled and item.id in generated
            for entry in generated_mesh_entries(item, generated[item.id])
        ]
        entries.extend(
            InventoryEntry(
                name=item.name,
                source_file=item.source_file,
                cleaned_file=item.cleaned_file,
                scale_factor=item.scale_factor,
                translation_mm=item.translation_mm,
                enabled=item.enabled,
            )
            for item in candidate.imported_meshes
        )
        inventory = inspect_system_meshes(tuple(entries))
        if any(not item.has_tetrahedra for item in inventory):
            radiators = list(imported_radiators)
            for item in candidate.generator_documents:
                if item.id != document.id and item.mesh_enabled and item.id in generated:
                    radiators.extend(replace(r, mesh=generator_mesh_name(item)) for r in generated[item.id].radiators)
            candidate.physical_system, routes = seed_exterior_system(inventory, tuple(radiators))
            candidate.component_channel_by_id.update(routes)
        else:
            candidate.physical_system = PhysicalSystem(
                id="system:loudspeaker", name="Loudspeaker", meshes=(), regions=(), boundaries=()
            )
    system = candidate.physical_system
    if any(mesh.id in new_ids - old_ids for mesh in system.meshes):
        raise ValueError("Generated assembly would overwrite an unowned mesh resource.")
    resources = tuple(
        MeshResource(
            id=generated_mesh_id(document.id, mesh.id),
            name=generated_mesh_name(document, mesh.id),
            file="" if mesh.mesh_data is not None else str(mesh.mesh_path),
            purpose=MeshPurpose(mesh.purpose),
            scale_to_m=document.mesh_scale_factor,
            translation_m=tuple(value / 1000 for value in document.mesh_translation_mm),
            mesh_data=mesh.mesh_data,
        )
        for mesh in geometry.meshes
    )
    other_names = {mesh.name for mesh in system.meshes if mesh.id not in old_ids}
    other_names.update(item.name for item in project.imported_meshes)
    if any(mesh.name in other_names for mesh in resources):
        raise ValueError("Generated assembly mesh names conflict with existing meshes.")
    # Keep retired resources until dependent entities have been explicitly removed.
    candidate.physical_system = replace(
        system, meshes=tuple(m for m in system.meshes if m.id not in new_ids) + resources
    )
    # A plugin may attach its own BEM resources to the shared exterior through
    # boundary assignments, without permission to rewrite the region's settings.
    assignments = completed.configuration_patch.get("surface_assignments", {}).get("upsert", [])
    exterior_ids = {mesh.id for mesh in resources if mesh.purpose == MeshPurpose.BEM_SURFACE}
    regions = []
    for region in system.regions:
        additions = {
            item.get("group", {}).get("mesh_id") for item in assignments if item.get("region_id") == region.id
        } & exterior_ids
        if region.kind == "unbounded_air" and additions:
            region = replace(region, mesh_ids=tuple(dict.fromkeys((*region.mesh_ids, *sorted(additions)))))
        regions.append(region)
    candidate.physical_system = replace(candidate.physical_system, regions=tuple(regions))
    candidate = apply_configuration_patch(
        candidate,
        completed.configuration_patch,
        document_id=document.id,
        mesh_ids=old_ids | new_ids,
    )
    if candidate.symmetry != "off":
        raise ValueError("Generated mesh assemblies currently require symmetry Off.")
    candidate.physical_system = replace(
        candidate.physical_system,
        meshes=tuple(m for m in candidate.physical_system.meshes if m.id not in old_ids - new_ids),
        regions=tuple(
            replace(region, mesh_ids=tuple(key for key in region.mesh_ids if key not in old_ids - new_ids))
            if region.kind == "unbounded_air" and not set(region.mesh_ids) <= old_ids | new_ids
            else region
            for region in candidate.physical_system.regions
        ),
    )
    _check_references(candidate.physical_system)
    used_meshes = {key for region in candidate.physical_system.regions for key in region.mesh_ids}
    if not new_ids <= used_meshes:
        raise ValueError("Every generated assembly mesh must be assigned to an acoustic region.")
    # Compilation validates purposes, groups, component assignments, coverage and
    # conforming interface topology before publishing any part of the assembly.
    PhysicalSystemCompiler().compile(candidate.physical_system)
    infer_physical_solve_kind(candidate.physical_system)
    candidate.generator_documents = replace_generator_document(
        candidate.generator_documents,
        document.id,
        artifact=geometry.to_reference(),
    )
    return candidate
