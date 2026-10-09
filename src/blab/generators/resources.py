"""Qt-free identity and inventory contracts for generated mesh assemblies."""

from dataclasses import replace

from blab.generators.base import GeneratedGeometry, GeneratorDocument, generated_mesh_id
from blab.generators.configuration import _check_references, _owned_entities
from blab.mesh_inventory import InventoryEntry
from blab.project.model import generator_mesh_name


def generated_mesh_name(document: GeneratorDocument, local_id: str) -> str:
    return f"{generator_mesh_name(document)}__{local_id}"


def generated_mesh_names(document: GeneratorDocument) -> tuple[str, ...]:
    if document.artifact is not None and document.artifact.meshes:
        return tuple(generated_mesh_name(document, mesh.id) for mesh in document.artifact.meshes)
    return (generator_mesh_name(document),)


def generated_mesh_entries(document: GeneratorDocument, result: GeneratedGeometry, symmetry: str = "off"):
    options = dict(
        scale_factor=document.mesh_scale_factor,
        translation_mm=document.mesh_translation_mm,
        enabled=document.mesh_enabled,
        locked=True,
        assembly_id=document.id if result.meshes else None,
    )
    if result.meshes:
        if symmetry != "off" and document.mesh_enabled:
            raise ValueError(
                "Generated mesh assemblies currently require symmetry Off; automatic coupled reduction is unsupported."
            )
        return tuple(
            InventoryEntry(
                name=generated_mesh_name(document, mesh.id),
                source_file="" if mesh.mesh_data is not None else str(mesh.mesh_path),
                mesh_data=mesh.mesh_data,
                **options,
            )
            for mesh in result.meshes
        )
    return (
        InventoryEntry(
            name=generator_mesh_name(document),
            source_file="" if result.mesh_data is not None else str(result.solver_mesh_path_for_symmetry(symmetry)),
            mesh_data=result.solver_mesh_data_for_symmetry(symmetry),
            **options,
        ),
    )


def without_assembly(system, document):
    """Detach only this assembly's physical entities; reject cross-design dependencies."""
    if system is None or document.artifact is None or not document.artifact.meshes:
        return system
    ids = {generated_mesh_id(document.id, mesh.id) for mesh in document.artifact.meshes}
    owned = _owned_entities(system, ids)
    changes = {
        collection: tuple(item for item in getattr(system, collection) if item.id not in keys)
        for collection, keys in owned.items()
    }
    changes["meshes"] = tuple(mesh for mesh in system.meshes if mesh.id not in ids)
    changes["regions"] = tuple(
        replace(
            region,
            mesh_ids=tuple(key for key in region.mesh_ids if key not in ids),
            volume_groups=tuple(group for group in region.volume_groups if group.mesh_id not in ids),
        )
        for region in changes["regions"]
    )
    result = replace(system, **changes)
    _check_references(result)
    return result


def active_assembly_system(system, documents):
    for document in documents:
        if not document.mesh_enabled:
            system = without_assembly(system, document)
    return system
