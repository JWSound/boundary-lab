"""Generated geometry and imported radiators, held in one place.

Several workflow modules read and write this state. Keeping it on the window
made the sharing invisible; holding it here makes it explicit, and lets the
derivations over it be tested without a window or a Qt application.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from blab.config import RadiatorConfig
from blab.generators.base import GeneratedGeometry, GeneratorDocument
from blab.ui.project_state import generator_mesh_name


@dataclass
class GeometryStore:
    """Geometry produced per design document, plus radiators from imported meshes."""

    generated_by_document_id: dict[str, GeneratedGeometry] = field(default_factory=dict)
    imported_radiators: tuple[RadiatorConfig, ...] = ()

    # -- derivations -------------------------------------------------------

    def enabled_geometry(
        self,
        documents: Iterable[GeneratorDocument],
    ) -> tuple[tuple[GeneratorDocument, GeneratedGeometry], ...]:
        """Documents that are mesh-enabled and have generated geometry available."""
        pairs = []
        for document in documents:
            if not document.mesh_enabled:
                continue
            result = self.generated_by_document_id.get(document.id)
            if result is not None:
                pairs.append((document, result))
        return tuple(pairs)

    def all_radiators(self, documents: Iterable[GeneratorDocument]) -> tuple[RadiatorConfig, ...]:
        """Every radiator in play, generated ones re-homed onto their mesh name."""
        radiators: list[RadiatorConfig] = []
        for document, result in self.enabled_geometry(documents):
            mesh_name = generator_mesh_name(document)
            radiators.extend(replace(radiator, mesh=mesh_name) for radiator in result.radiators)
        radiators.extend(self.imported_radiators)
        return tuple(radiators)

    # -- mutation ----------------------------------------------------------

    def apply_radiators(
        self,
        documents: Iterable[GeneratorDocument],
        radiators: tuple[RadiatorConfig, ...],
    ) -> None:
        """Distribute an edited radiator set back to its owning geometry.

        Radiators whose mesh is not one of the generated meshes are retained as
        imported radiators, so imported-mesh sources survive the round trip.
        """
        documents = tuple(documents)
        generated_mesh_names = {generator_mesh_name(document) for document in documents}
        for document, result in self.enabled_geometry(documents):
            mesh_name = generator_mesh_name(document)
            updated = [replace(radiator, mesh=mesh_name) for radiator in radiators if radiator.mesh == mesh_name]
            self.generated_by_document_id[document.id] = replace(result, radiators=tuple(updated))
        self.imported_radiators = tuple(radiator for radiator in radiators if radiator.mesh not in generated_mesh_names)

    def clear_imported_radiators(self) -> None:
        self.imported_radiators = ()

    def reassign_channels(self, valid_names: set[str], fallback: str) -> None:
        """Move radiators whose channel no longer exists onto a fallback channel.

        Called after the channel set is edited, so no radiator is left pointing
        at a deleted or renamed channel.
        """
        for document_id, result in tuple(self.generated_by_document_id.items()):
            self.generated_by_document_id[document_id] = replace(
                result,
                radiators=tuple(
                    radiator if radiator.channel in valid_names else replace(radiator, channel=fallback)
                    for radiator in result.radiators
                ),
            )


def radiator_channel_assignments(
    radiators: Iterable[RadiatorConfig],
) -> tuple[tuple[str | None, int, str], ...]:
    """A comparable snapshot of which channel each radiator is driven by."""
    return tuple((radiator.mesh, radiator.tag, radiator.channel) for radiator in radiators)
