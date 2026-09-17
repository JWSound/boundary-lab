"""Stage provider configuration against generated artifacts without GUI state."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace

from blab.config import RadiatorConfig
from blab.generators.base import GeneratedGeometry, GenerationCompleted
from blab.generators.configuration import apply_configuration_patch, project_revision
from blab.mesh_cache import read_mesh
from blab.mesh_inventory import InventoryEntry, inspect_system_meshes
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
    candidate = deepcopy(project)
    candidate.generator_documents = replace_generator_document(
        candidate.generator_documents, document.id, artifact=geometry.to_reference(),
    )
    all_geometry = dict(generated) | {document.id: geometry}
    if candidate.physical_system is None:
        entries, radiators = [], list(imported_radiators)
        for item in candidate.generator_documents:
            result = all_geometry.get(item.id)
            if result is None or not item.mesh_enabled:
                continue
            name = generator_mesh_name(item)
            entries.append(InventoryEntry(
                name=name, source_file=str(result.solver_mesh_path), scale_factor=item.mesh_scale_factor,
                translation_mm=item.mesh_translation_mm, locked=True,
            ))
            radiators.extend(replace(radiator, mesh=name) for radiator in result.radiators)
        entries.extend(InventoryEntry(
            name=item.name, source_file=item.source_file, cleaned_file=item.cleaned_file,
            scale_factor=item.scale_factor, translation_mm=item.translation_mm, enabled=item.enabled,
        ) for item in candidate.imported_meshes)
        candidate.physical_system, candidate.component_channel_by_id = seed_exterior_system(
            inspect_system_meshes(tuple(entries)), tuple(radiators),
        )
    name = generator_mesh_name(document)
    mesh_ids = {item.id for item in candidate.physical_system.meshes if item.name == name}
    if not mesh_ids:
        raise ValueError("Generated mesh is not assigned to the physical system; configure its mesh resource first.")
    candidate.physical_system = replace(candidate.physical_system, meshes=tuple(
        replace(
            item, file=str(geometry.solver_mesh_path), scale_to_m=document.mesh_scale_factor,
            translation_m=tuple(value / 1000 for value in document.mesh_translation_mm),
        ) if item.id in mesh_ids else item
        for item in candidate.physical_system.meshes
    ))
    candidate = apply_configuration_patch(
        candidate, completed.configuration_patch, document_id=document.id, mesh_ids=mesh_ids,
    )
    # Regeneration must not silently redirect an old assignment to another tag.
    mesh = read_mesh(geometry.solver_mesh_path)
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
