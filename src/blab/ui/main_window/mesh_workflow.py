"""Imported/generated mesh selection, stitching, and 3D preview refresh."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
)

from blab.ath import (
    read_surface_physical_names,
)
from blab.config import MeshConfig, RadiatorConfig
from blab.generators.base import GeneratedGeometry, GeneratorDocument
from blab.generators.postprocess import ensure_reduced_geometry
from blab.mesh_topology import analyze_exterior_mesh_topology
from blab.ui.dialogs import (
    MeshDialogEntry,
)
from blab.ui.main_window.helpers import (
    _mesh_entries_with_file_overrides,
    _physical_system_preview_metadata,
)
from blab.ui.mesh_assembly import (
    STITCH_FAILURE_MESSAGE,
    STITCHED_MESH_NAME,
    MeshAssemblyService,
    PreparedMeshAssembly,
)
from blab.ui.project_state import (
    ImportedMeshState,
    generator_mesh_name,
    replace_generator_document,
)
from blab.ui.system_config import (
    INTERFACE_SEAM_SIMPLIFICATION_WARNING,
    inspect_system_meshes,
    interface_bem_mesh_names_for_changes,
    rebuild_configured_interfaces,
)


class MeshWorkflowMixin:
    """Imported/generated mesh selection, stitching, and 3D preview refresh.

    Mixed into :class:`~blab.ui.main_window.window.MainWindow`.
    """

    def _mesh_config_dialog_entries(self) -> tuple[MeshDialogEntry, ...]:
        return self.mesh_entries_for_symmetry("off")

    def mesh_entries_for_symmetry(
        self,
        symmetry: str,
    ) -> tuple[MeshDialogEntry, ...]:
        """Return mesh entries backed by solve-ready generated geometry."""

        entries = []
        for document in self.generator_documents:
            result = self.generated_geometry_by_document_id.get(document.id)
            if result is None:
                continue
            solver_result = self._generated_geometry_for_solver_symmetry(document, result, symmetry)
            entries.append(
                MeshDialogEntry(
                    name=generator_mesh_name(document),
                    source_file=str(solver_result.solver_mesh_path_for_symmetry(symmetry)),
                    scale_factor=float(document.mesh_scale_factor),
                    translation_mm=document.mesh_translation_mm,
                    enabled=document.mesh_enabled,
                    locked=True,
                )
            )
        entries.extend(self.imported_meshes)
        return tuple(entries)

    def _apply_mesh_config_dialog_entries(self, meshes: tuple[MeshDialogEntry, ...]) -> None:
        imported_meshes = []
        documents = self.generator_documents
        for mesh in meshes:
            document = self._generator_document_for_mesh_name(mesh.name)
            if document is not None:
                documents = replace_generator_document(
                    documents,
                    document.id,
                    mesh_enabled=bool(mesh.enabled),
                    mesh_translation_mm=mesh.translation_mm,
                    mesh_scale_factor=float(mesh.scale_factor),
                )
            else:
                imported_meshes.append(replace(mesh, locked=False))
        self.generator_documents = documents
        self.imported_meshes = tuple(imported_meshes)

    def has_solver_meshes(self) -> bool:
        return bool(self._enabled_generated_geometry()) or bool(self._active_imported_meshes())

    def _clean_imported_meshes(self, meshes: tuple[MeshDialogEntry, ...]) -> tuple[MeshDialogEntry, ...]:
        states = tuple(
            ImportedMeshState(
                name=mesh.name,
                source_file=mesh.source_file,
                cleaned_file=mesh.cleaned_file,
                scale_factor=mesh.scale_factor,
                translation_mm=mesh.translation_mm,
                enabled=mesh.enabled,
            )
            for mesh in meshes
        )
        cleaned = self.mesh_service().clean_imported_meshes(states)
        return tuple(
            replace(mesh, cleaned_file=state.cleaned_file) for mesh, state in zip(meshes, cleaned, strict=True)
        )

    def mesh_service(self) -> MeshAssemblyService:
        service = getattr(self, "mesh_assembly_service", None)
        if service is None:
            service = MeshAssemblyService(Path.cwd() / "runs" / "imported_meshes")
            self.mesh_assembly_service = service
        return service

    def _imported_mesh_needs_reload(self, mesh: MeshDialogEntry) -> bool:
        if not mesh.enabled:
            return False
        source_path = Path(mesh.source_file)
        if source_path.suffix.lower() != ".msh" or not source_path.exists():
            return False
        fingerprint = self._mesh_source_fingerprint(source_path)
        if fingerprint is None:
            return False
        previous = getattr(self, "_imported_mesh_source_fingerprints", {}).get(self._mesh_source_key(source_path))
        return previous is not None and fingerprint != previous

    @staticmethod
    def _mesh_source_key(source_path: Path) -> str:
        return str(source_path.resolve())

    @staticmethod
    def _mesh_source_fingerprint(source_path: Path) -> tuple[int, int] | None:
        try:
            stat = source_path.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _record_imported_mesh_source_fingerprints(self) -> None:
        previous = getattr(self, "_imported_mesh_source_fingerprints", {})
        fingerprints: dict[str, tuple[int, int]] = {}
        for mesh in self.imported_meshes:
            source_path = Path(mesh.source_file)
            if source_path.suffix.lower() != ".msh":
                continue
            key = self._mesh_source_key(source_path)
            fingerprint = self._mesh_source_fingerprint(source_path)
            if fingerprint is not None:
                fingerprints[key] = fingerprint
            elif key in previous:
                fingerprints[key] = previous[key]
        self._imported_mesh_source_fingerprints = fingerprints

    def _updated_imported_mesh_names(self) -> tuple[str, ...]:
        return tuple(mesh.name for mesh in self.imported_meshes if self._imported_mesh_needs_reload(mesh))

    def _reload_updated_imported_meshes_on_focus(self) -> None:
        if not self.imported_meshes or self.solve_controller.active:
            return

        now = time.monotonic()
        if now - self._last_imported_mesh_focus_check_at < 0.5:
            return
        self._last_imported_mesh_focus_check_at = now

        cursor_set = False
        quality_warning_interface_ids: tuple[str, ...] = ()
        reload_succeeded = False
        try:
            updated_names = self._updated_imported_mesh_names()
            if not updated_names:
                return
            QApplication.setOverrideCursor(Qt.WaitCursor)
            cursor_set = True
            self.show_status(f"Reloading updated mesh file{'s' if len(updated_names) != 1 else ''}...")
            physical_system = self._project_document().physical_system
            interface_bem_names = set(interface_bem_mesh_names_for_changes(physical_system, set(updated_names)))
            reload_candidates = tuple(
                replace(mesh, cleaned_file=None) if mesh.name in interface_bem_names else mesh
                for mesh in self.imported_meshes
            )
            reloaded_meshes = self._clean_imported_meshes(reload_candidates)
            rebuilt_interface_count = 0
            if physical_system is not None and interface_bem_names:
                generated_meshes = tuple(mesh for mesh in self.mesh_entries_for_symmetry("off") if mesh.locked)
                available_meshes = inspect_system_meshes((*generated_meshes, *reloaded_meshes))
                rebuild = rebuild_configured_interfaces(
                    physical_system,
                    available_meshes,
                    changed_mesh_names=set(updated_names),
                    interface_output_root=self.mesh_service().output_root,
                    symmetry_mode=self.symmetry,
                )
                reloaded_meshes = _mesh_entries_with_file_overrides(
                    reloaded_meshes,
                    rebuild.mesh_file_overrides_by_name,
                )
                self._project_document().physical_system = rebuild.system
                rebuilt_interface_count = len(rebuild.rebuilt_interface_ids)
                quality_warning_interface_ids = rebuild.quality_warning_interface_ids
            self.imported_meshes = reloaded_meshes
            self._record_imported_mesh_source_fingerprints()
            self.mesh_state_changed.emit("imported_mesh_files_reloaded")
            self.solve_results_invalidated.emit("imported_mesh_files_reloaded")
            names = ", ".join(updated_names)
            interface_text = (
                f"; rebuilt {rebuilt_interface_count} interface{'s' if rebuilt_interface_count != 1 else ''}"
                if rebuilt_interface_count
                else ("; interfaces verified" if interface_bem_names else "")
            )
            self.show_status(
                f"Reloaded updated mesh file{'s' if len(updated_names) != 1 else ''}: {names}{interface_text}"
            )
            reload_succeeded = True
        except Exception as exc:
            self.solve_results_invalidated.emit("imported_mesh_files_reloaded")
            self.show_status(f"Imported mesh reload failed: {exc}")
        finally:
            if cursor_set:
                QApplication.restoreOverrideCursor()
        if reload_succeeded and quality_warning_interface_ids:
            QMessageBox.warning(
                self,
                "Inspect Simplified Interface",
                INTERFACE_SEAM_SIMPLIFICATION_WARNING,
            )

    def _cleaned_imported_mesh_path(self, mesh: MeshDialogEntry) -> Path:
        return self.mesh_service().cleaned_imported_mesh_path(
            ImportedMeshState(name=mesh.name, source_file=mesh.source_file)
        )

    def _stitch_candidate_mesh_configs(self) -> tuple[MeshConfig, ...]:
        return (*self._generated_solver_mesh_configs_for_symmetry(self.symmetry), *self._imported_solver_mesh_configs())

    def _should_use_stitched_mesh(self) -> bool:
        return self.stitch_imported_meshes and len(self._stitch_candidate_mesh_configs()) > 1

    def _stitched_mesh_path(self, mesh_configs: tuple[MeshConfig, ...]) -> Path:
        return self.mesh_service().stitched_mesh_path(
            mesh_configs,
            self.preferences.stitch_tolerance_mm,
            self.symmetry,
        )

    def _mesh_for_stitching(self, mesh_cfg: MeshConfig):
        return self.mesh_service().mesh_for_stitching(mesh_cfg)

    def _stitch_ignored_boundary_axes(self) -> tuple[str, ...]:
        return self.mesh_service().ignored_boundary_axes(self.symmetry)

    def _stitched_solver_mesh_config(self) -> MeshConfig | None:
        if not self._should_use_stitched_mesh():
            return None
        assembly = self.prepare_mesh_assembly(())
        return assembly.mesh_configs[0] if assembly.mesh_configs else None

    def _active_imported_meshes(self) -> tuple[MeshDialogEntry, ...]:
        return tuple(mesh for mesh in self.imported_meshes if mesh.enabled)

    def _generated_geometry_for_solver_symmetry(
        self,
        document: GeneratorDocument,
        result: GeneratedGeometry,
        symmetry: str,
    ) -> GeneratedGeometry:
        if symmetry == "off":
            return result
        updated = ensure_reduced_geometry(result)
        self.generated_geometry_by_document_id[document.id] = updated
        self.generator_documents = replace_generator_document(
            self.generator_documents,
            document.id,
            artifact=updated.to_reference(),
        )
        return updated

    def _generated_solver_mesh_configs_for_symmetry(self, symmetry: str) -> tuple[MeshConfig, ...]:
        configs = []
        for document, result in self._enabled_generated_geometry():
            solver_result = self._generated_geometry_for_solver_symmetry(document, result, symmetry)
            configs.append(
                MeshConfig(
                    name=generator_mesh_name(document),
                    file=str(solver_result.solver_mesh_path_for_symmetry(symmetry)),
                    scale_factor=float(document.mesh_scale_factor),
                    translation_m=tuple(value / 1000.0 for value in document.mesh_translation_mm),
                )
            )
        return tuple(configs)

    def _generated_solver_mesh_configs(self) -> tuple[MeshConfig, ...]:
        return self._generated_solver_mesh_configs_for_symmetry(self.symmetry)

    def _imported_solver_mesh_configs(self) -> tuple[MeshConfig, ...]:
        configs = []
        for mesh in self._active_imported_meshes():
            mesh_file = self._mesh_file_for_imported(mesh)
            configs.append(
                MeshConfig(
                    name=mesh.name,
                    file=mesh_file,
                    scale_factor=float(mesh.scale_factor),
                    translation_m=tuple(value / 1000.0 for value in mesh.translation_mm),
                )
            )
        return tuple(configs)

    def _mesh_file_for_imported(self, mesh: MeshDialogEntry) -> str:
        if mesh.cleaned_file and Path(mesh.cleaned_file).exists():
            return mesh.cleaned_file
        return mesh.source_file

    def _solver_mesh_configs(self) -> tuple[MeshConfig, ...]:
        return self.prepare_mesh_assembly(self.all_radiators()).mesh_configs

    def prepare_mesh_assembly(
        self,
        radiators: tuple[RadiatorConfig, ...],
    ) -> PreparedMeshAssembly:
        assembly = self.mesh_service().prepare(
            generated_mesh_configs=self._generated_solver_mesh_configs(),
            imported_meshes=self._project_document().imported_meshes,
            radiators=radiators,
            stitch_imported_meshes=self.stitch_imported_meshes,
            stitch_tolerance_mm=self.preferences.stitch_tolerance_mm,
            symmetry=self.symmetry,
        )
        self._project_document().imported_meshes = assembly.imported_meshes
        return assembly

    def _unique_stitched_surface_name(
        self,
        surface_name: str,
        used_surface_names: set[str],
        mesh_index: int,
    ) -> str:
        return self.mesh_service().unique_surface_name(surface_name, used_surface_names, mesh_index)

    def _used_surface_tags_for_mesh(self, mesh_cfg: MeshConfig) -> tuple[int, ...]:
        return self.mesh_service().used_surface_tags(mesh_cfg)

    def _stitched_radiator_map(self) -> dict[tuple[str | None, int], tuple[str, int]]:
        return self.mesh_service().stitched_radiator_map(self._stitch_candidate_mesh_configs())

    def _radiators_for_solver_meshes(
        self,
        mesh_configs: tuple[MeshConfig, ...],
        radiators: tuple[RadiatorConfig, ...],
    ) -> tuple[RadiatorConfig, ...]:
        if len(mesh_configs) != 1 or mesh_configs[0].name != STITCHED_MESH_NAME:
            return radiators
        return self.mesh_service().radiators_for_stitched_mesh(
            self._stitch_candidate_mesh_configs(),
            radiators,
        )

    def show_stitch_or_generic_error(self, title: str, exc: Exception) -> None:
        if str(exc) != STITCH_FAILURE_MESSAGE:
            QMessageBox.critical(self, title, str(exc))
            return

        message = QMessageBox(QMessageBox.Critical, title, STITCH_FAILURE_MESSAGE, QMessageBox.Ok, self)
        if exc.__cause__ is not None:
            message.setDetailedText(str(exc.__cause__))
        message.exec()

    def show_mesh_quality_warning(self, result: GeneratedGeometry) -> None:
        warning = result.quality_warning
        if warning is None or not warning.has_warnings:
            return

        QMessageBox.warning(
            self,
            "Mesh quality warning",
            (
                "The cleaned mesh contains extremely thin triangles that may make the "
                "BEAT Engine produced non-finite results.\n\n"
                f"Thin triangles: {warning.sliver_triangles}\n"
                f"Float32-singular triangles: {warning.float32_singular_triangles}\n"
                f"Worst triangle: {warning.worst_triangle_index}\n"
                f"Worst altitude/edge ratio: {warning.worst_altitude_edge_ratio:.3g}\n\n"
                "Try increasing mesh resolution around sharp transitions or adjusting the geometry "
                "to avoid long, needle-like triangles."
            ),
        )

    def surface_tags_for_meshes(self) -> dict[str, tuple[str, int]]:
        return self.prepare_mesh_assembly(self.all_radiators()).surface_tags

    def _refresh_mesh_preview(self) -> None:
        self._sync_preview_region_actions()
        if not self.has_solver_meshes():
            self.clear_mesh_preview()
            return
        try:
            assembly = self.prepare_mesh_assembly(self.all_radiators())
            mesh_configs = assembly.mesh_configs
            if not mesh_configs:
                self.clear_mesh_preview()
                return
            interface_surfaces, component_surfaces, mesh_regions, has_interior = _physical_system_preview_metadata(
                self._project_document().physical_system,
                assembly.surface_tags_by_mesh,
            )
            driven_surfaces = {(radiator.mesh, radiator.tag) for radiator in assembly.radiators} | component_surfaces
            topology_report = self._mesh_preview_topology_report(
                mesh_configs,
                mesh_regions=mesh_regions,
                has_interior=has_interior,
            )
            self.show_mesh_preview(
                mesh_configs,
                driven_surfaces=driven_surfaces,
                surface_tags_by_mesh=assembly.surface_tags_by_mesh,
                interface_surfaces=interface_surfaces,
                mesh_regions=mesh_regions,
                symmetry=self.symmetry,
                topology_report=topology_report,
            )
        except Exception as exc:
            if str(exc) == STITCH_FAILURE_MESSAGE and self.stitch_imported_meshes:
                self._refresh_unstitched_mesh_preview_after_stitch_failure()
                return
            self.clear_mesh_preview()

    def _refresh_unstitched_mesh_preview_after_stitch_failure(self) -> None:
        try:
            mesh_configs = self._stitch_candidate_mesh_configs()
            if not mesh_configs:
                self.clear_mesh_preview()
                return
            surface_tags_by_mesh = {
                mesh_cfg.name: read_surface_physical_names(Path(mesh_cfg.file)) for mesh_cfg in mesh_configs
            }
            interface_surfaces, component_surfaces, mesh_regions, has_interior = _physical_system_preview_metadata(
                self._project_document().physical_system,
                surface_tags_by_mesh,
            )
            driven_surfaces = {(radiator.mesh, radiator.tag) for radiator in self.all_radiators()} | component_surfaces
            topology_report = self._mesh_preview_topology_report(
                mesh_configs,
                mesh_regions=mesh_regions,
                has_interior=has_interior,
            )
            self.show_mesh_preview(
                mesh_configs,
                driven_surfaces=driven_surfaces,
                surface_tags_by_mesh=surface_tags_by_mesh,
                interface_surfaces=interface_surfaces,
                mesh_regions=mesh_regions,
                symmetry=self.symmetry,
                topology_report=topology_report,
            )
            self.show_status("Mesh preview showing unstitched meshes; stitching failed")
        except Exception:
            self.clear_mesh_preview()

    def _mesh_preview_topology_report(
        self,
        mesh_configs: tuple[MeshConfig, ...],
        *,
        mesh_regions: dict[str, str],
        has_interior: bool,
    ):
        exterior_meshes = (
            tuple(mesh for mesh in mesh_configs if mesh_regions.get(mesh.name) == "exterior")
            if has_interior
            else mesh_configs
        )
        if not exterior_meshes:
            return None
        try:
            return analyze_exterior_mesh_topology(exterior_meshes, symmetry=self.symmetry)
        except (OSError, ValueError):
            # Preview rendering remains useful when a topology diagnostic cannot
            # be produced. The solve path reports the validation error directly.
            return None
