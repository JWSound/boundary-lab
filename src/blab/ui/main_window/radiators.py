"""Radiator collection and exterior physical-system seeding."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from blab.ath import (
    read_surface_physical_names,
)
from blab.config import RadiatorConfig
from blab.generators.base import GeneratedGeometry, GeneratorDocument
from blab.ui.mesh_assembly import (
    STITCHED_MESH_NAME,
)
from blab.ui.physical_system_migration import AUTO_SEEDED_EXTERIOR_KEY, seed_exterior_system
from blab.ui.system_config import (
    inspect_system_meshes,
)


class RadiatorsMixin:
    """Radiator collection and exterior physical-system seeding.

    Mixed into :class:`~blab.ui.main_window.window.MainWindow`.
    """

    def _enabled_generated_geometry(self) -> tuple[tuple[GeneratorDocument, GeneratedGeometry], ...]:
        return self._geometry_store().enabled_geometry(self.generator_documents)

    def all_radiators(self) -> tuple[RadiatorConfig, ...]:
        return self._geometry_store().all_radiators(self.generator_documents)

    def ensure_seeded_exterior_system(self) -> None:
        current = self.project.physical_system
        if current is not None and not bool(current.metadata.get(AUTO_SEEDED_EXTERIOR_KEY, False)):
            return
        try:
            meshes = inspect_system_meshes(self._mesh_config_dialog_entries())
            if not any(not mesh.has_tetrahedra for mesh in meshes):
                return
            system, component_channels = seed_exterior_system(
                meshes,
                self._source_radiators_for_system_seed(),
            )
        except (OSError, ValueError):
            return
        self.project.physical_system = system
        self.project.component_channel_by_id = component_channels
        if all(
            not document.mesh_enabled or document.id in self.generated_geometry_by_document_id
            for document in self.generator_documents
        ):
            self.project.source_config_by_name = {}

    def _source_radiators_for_system_seed(self) -> tuple[RadiatorConfig, ...]:
        radiators = self.all_radiators()
        if not any(radiator.mesh == STITCHED_MESH_NAME for radiator in radiators):
            return radiators
        try:
            source_meshes = self._stitch_candidate_mesh_configs()
            stitched_map = self.mesh_service().stitched_radiator_map(source_meshes)
            source_name_by_key = {
                (mesh.name, int(tag)): f"{mesh.name}:{name}"
                for mesh in source_meshes
                for name, tag in read_surface_physical_names(Path(mesh.file)).items()
            }
        except (OSError, ValueError):
            return radiators
        reverse = {
            (str(stitched_name), int(stitched_tag)): (str(mesh_name), int(source_tag))
            for (mesh_name, source_tag), (stitched_name, stitched_tag) in stitched_map.items()
        }
        resolved = []
        for radiator in radiators:
            if radiator.mesh != STITCHED_MESH_NAME:
                resolved.append(radiator)
                continue
            source_key = reverse.get((radiator.name, int(radiator.tag)))
            if source_key is None:
                source_key = next(
                    (
                        candidate
                        for (stitched_name, stitched_tag), candidate in reverse.items()
                        if stitched_tag == int(radiator.tag)
                    ),
                    None,
                )
            if source_key is None:
                continue
            resolved.append(
                replace(
                    radiator,
                    name=source_name_by_key.get(source_key, radiator.name),
                    mesh=source_key[0],
                    tag=source_key[1],
                )
            )
        return tuple(resolved)

    def _apply_radiators_to_results(self, radiators: tuple[RadiatorConfig, ...]) -> None:
        self._geometry_store().apply_radiators(self.generator_documents, radiators)
