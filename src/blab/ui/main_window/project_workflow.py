"""Project lifecycle: new/open/save/import/export and payload serialization.

A controller, not a mixin. It reaches the UI only through :class:`WorkflowView`
— including the file pickers and the unsaved-changes prompt — and the domain
through :class:`ProjectInputs`. The open project lives in a
:class:`ProjectSession` and generated geometry in a :class:`GeometryStore`,
rather than as attributes on the window, so the rules over them are testable
without a Qt application.

Follows the shape of :mod:`blab.ui.main_window.solve_workflow`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from blab.generators.ath import ath_source_text, with_ath_source_text
from blab.observation_planes import observation_planes_from_payload
from blab.physical_model import (
    physical_system_from_dict,
    physical_system_to_dict,
)
from blab.ui.dialogs import (
    MeshDialogEntry,
)
from blab.ui.main_window.constants import (
    DEFAULT_MESH_SCALE_FACTOR,
)
from blab.ui.main_window.geometry_store import GeometryStore
from blab.ui.main_window.project_session import ProjectSession, canonical_payload
from blab.ui.main_window.workflow_view import (
    FrequencyRange,
    ProjectInputs,
    UnsavedChoice,
    WorkflowView,
)
from blab.ui.physical_system_migration import AUTO_SEEDED_EXTERIOR_KEY
from blab.ui.project_io import (
    PROJECT_DEFAULT_NAME,
    PROJECT_FILE_FILTER,
    build_project_payload,
    normalize_project_path,
    read_project_file,
    write_project_file,
)
from blab.ui.project_state import (
    ImportedMeshState,
    ProjectDocument,
    ProjectPreferencesState,
    generator_document_to_payload,
    generator_documents_from_payload,
    generator_mesh_name,
    new_generator_document,
    new_project_document,
    unique_generator_name,
)
from blab.ui.settings import (
    GuiPreferences,
    gui_preferences_with_project_preferences,
    project_preferences_from_gui,
)

#: Shared by the Ath design import and export pickers.
ATH_CONFIG_FILTER = "Ath config files (*.cfg);;All files (*)"


class ProjectWorkflowController(QObject):
    """Project lifecycle: new/open/save/import/export and payload serialization."""

    #: Emitted when the open project is replaced wholesale, so everything
    #: derived from it can be rebuilt.
    project_state_changed = Signal(str)

    #: Emitted when replacing the project invalidates existing solve results.
    solve_results_invalidated = Signal(str)

    def __init__(
        self,
        parent: QObject | None,
        *,
        view: WorkflowView,
        inputs: ProjectInputs,
        session: ProjectSession,
        geometry_store: GeometryStore,
        preferences: Callable[[], GuiPreferences],
        set_preferences: Callable[[GuiPreferences], None],
        save_preferences: Callable[[], None],
        save_frequency_settings: Callable[[], None],
        remember_recent: Callable[[Path], None],
        forget_recent: Callable[[Path], None],
    ) -> None:
        super().__init__(parent)
        self._view = view
        self._inputs = inputs
        self._session = session
        self._geometry_store = geometry_store
        self._read_preferences = preferences
        self._write_preferences = set_preferences
        self._save_preferences = save_preferences
        self._save_frequency_settings = save_frequency_settings
        self._remember_recent = remember_recent
        self._forget_recent = forget_recent

    @property
    def _project(self) -> ProjectDocument:
        return self._session.document

    # -- payload serialization ---------------------------------------------

    def _imported_meshes_payload(self) -> list[dict]:
        return [self._mesh_entry_to_payload(mesh, absolute_paths=True) for mesh in self._project.imported_meshes]

    def _mesh_entry_to_payload(self, mesh, *, absolute_paths: bool) -> dict:
        source_file = str(Path(mesh.source_file).resolve()) if absolute_paths and mesh.source_file else mesh.source_file
        cleaned_file = (
            None
            if mesh.cleaned_file is None
            else str(Path(mesh.cleaned_file).resolve())
            if absolute_paths
            else mesh.cleaned_file
        )
        return {
            "name": mesh.name,
            "source_file": source_file,
            "cleaned_file": cleaned_file,
            "scale_factor": float(mesh.scale_factor),
            "translation_mm": [int(round(value)) for value in mesh.translation_mm],
            "enabled": bool(mesh.enabled),
        }

    @staticmethod
    def _mesh_scale_from_payload(payload: object) -> float:
        if not isinstance(payload, dict):
            return DEFAULT_MESH_SCALE_FACTOR
        try:
            scale_factor = float(payload.get("scale_factor", DEFAULT_MESH_SCALE_FACTOR))
        except (TypeError, ValueError):
            return DEFAULT_MESH_SCALE_FACTOR
        return scale_factor if scale_factor > 0.0 else DEFAULT_MESH_SCALE_FACTOR

    def _mesh_entries_from_payload(self, payload: object) -> tuple[MeshDialogEntry, ...]:
        if not isinstance(payload, list):
            return ()

        meshes = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            source_file = str(item.get("source_file", "")).strip()
            name = str(item.get("name", "")).strip()
            generated_mesh_names = {generator_mesh_name(document) for document in self._project.generator_documents}
            if not source_file or not name or name in generated_mesh_names:
                continue
            translation = item.get("translation_mm", [0.0, 0.0, 0.0])
            if not isinstance(translation, list) or len(translation) != 3:
                translation = [0.0, 0.0, 0.0]
            meshes.append(
                MeshDialogEntry(
                    name=name,
                    source_file=source_file,
                    cleaned_file=None if item.get("cleaned_file") is None else str(item.get("cleaned_file")),
                    scale_factor=self._mesh_scale_from_payload(item),
                    translation_mm=tuple(float(int(round(float(value)))) for value in translation),
                    enabled=bool(item.get("enabled", True)),
                )
            )
        return tuple(meshes)

    def project_payload(self) -> dict:
        project = self._project
        project_preferences = self.current_project_preferences()
        project.project_preferences = project_preferences
        return build_project_payload(
            generator_documents=[
                generator_document_to_payload(document, absolute_paths=True) for document in project.generator_documents
            ],
            active_generator_document_id=project.active_generator_document_id,
            imported_meshes=self._imported_meshes_payload(),
            stitch_imported_meshes=project.stitch_imported_meshes,
            symmetry=project.symmetry,
            source_config_by_name=self._inputs.source_config_by_name(),
            channel_config_by_name=self._inputs.channel_config_by_name(),
            max_spl_limits_by_channel=project.max_spl_limits_by_channel,
            project_preferences=project_preferences.to_payload(),
            physical_system=(
                None if project.physical_system is None else physical_system_to_dict(project.physical_system)
            ),
            component_channel_by_id=project.component_channel_by_id,
            observation_planes=[plane.to_payload() for plane in project.observation_planes],
        )

    def canonical_project_payload(self) -> dict:
        return canonical_payload(self.project_payload())

    def current_project_preferences(self) -> ProjectPreferencesState:
        frequencies = self._view.frequency_range()
        return project_preferences_from_gui(
            self._read_preferences(),
            freq_min_hz=frequencies.min_hz,
            freq_max_hz=frequencies.max_hz,
            freq_count=frequencies.count,
        )

    # -- change tracking ----------------------------------------------------

    def mark_project_clean(self) -> None:
        self._session.mark_clean(self.project_payload())

    def has_unsaved_project_changes(self) -> bool:
        return self._session.has_unsaved_changes(self.project_payload())

    def confirm_unsaved_project_changes(self, action: str) -> bool:
        """Whether ``action`` may proceed, saving first if the user asks for it.

        Returns False only when the user cancels, which is what vetoes a close
        or abandons an open.
        """
        if not self.has_unsaved_project_changes():
            return True
        choice = self._view.ask_unsaved_changes(closing=action == "close")
        if choice is UnsavedChoice.SAVE:
            return self.save_project()
        return choice is UnsavedChoice.DISCARD

    # -- new / save / open --------------------------------------------------

    @Slot()
    def new_project(self) -> None:
        if not self.confirm_unsaved_project_changes("new_project"):
            return
        self._inputs.discard_channel_config_dialog()
        self._session.replace(
            new_project_document(project_preferences=self.current_project_preferences()),
            path=None,
        )
        self._geometry_store.generated_by_document_id = {}
        self._inputs.rebuild_generator_document_tabs()
        self._geometry_store.clear_imported_radiators()
        self.project_state_changed.emit("new_project")
        self.solve_results_invalidated.emit("new_project")
        self.mark_project_clean()
        self._view.show_status("New project")

    @Slot()
    def save_project(self) -> bool:
        if self._session.path is None:
            return self.save_project_as()
        return self._save_project_to_path(self._session.path)

    @Slot()
    def save_project_as(self) -> bool:
        suggested_filename = PROJECT_DEFAULT_NAME if self._session.path is None else self._session.path.name
        path = self._view.choose_save_file("Save Project", PROJECT_FILE_FILTER, suggested_filename)
        if path is None:
            return False
        return self._save_project_to_path(normalize_project_path(path))

    def _save_project_to_path(self, path: Path) -> bool:
        try:
            project_path = write_project_file(path, self.project_payload())
            self._session.path = project_path
            self._remember_recent(project_path)
            self.mark_project_clean()
            self._view.show_status(f"Saved project {project_path}")
            return True
        except Exception as exc:
            self._view.show_error("Save project failed", str(exc))
            return False

    @Slot()
    def load_project(self) -> None:
        if not self.confirm_unsaved_project_changes("open_project"):
            return
        path = self._view.choose_open_file("Open Project", PROJECT_FILE_FILTER)
        if path is None:
            return

        self._load_project_from_path(path)

    @Slot()
    def open_recent_project(self, path: Path) -> None:
        if not path.exists():
            self._forget_recent(path)
            self._view.warn("Open project failed", f"Recent project not found:\n{path}")
            return
        if not self.confirm_unsaved_project_changes("open_project"):
            return
        self._load_project_from_path(path)

    def _load_project_from_path(self, path: Path) -> None:
        try:
            payload = read_project_file(path)
            project_preferences = ProjectPreferencesState.from_payload(payload.get("project_preferences"))
            if self._confirm_apply_project_preferences(project_preferences):
                self._apply_project_preferences(project_preferences)
            self._apply_project_payload(payload)
            self._session.path = path
            self._remember_recent(path)
            self.mark_project_clean()
            self._view.show_status(f"Opened project {path}")
        except Exception as exc:
            self._view.show_error("Open project failed", str(exc))

    # -- project preferences ------------------------------------------------

    def _confirm_apply_project_preferences(self, project_preferences: ProjectPreferencesState | None) -> bool:
        if project_preferences is None or project_preferences == self.current_project_preferences():
            return False
        return self._view.confirm(
            "Project Preferences",
            "This project file contains unique application preferences. Would you like to apply them?",
        )

    def _apply_project_preferences(self, project_preferences: ProjectPreferencesState) -> None:
        self._write_preferences(gui_preferences_with_project_preferences(self._read_preferences(), project_preferences))
        self._view.set_frequency_range(
            FrequencyRange(
                min_hz=project_preferences.freq_min_hz,
                max_hz=project_preferences.freq_max_hz,
                count=project_preferences.freq_count,
            )
        )
        self._save_preferences()
        self._save_frequency_settings()
        self._project.project_preferences = self.current_project_preferences()

    # -- applying a loaded payload -----------------------------------------

    def _apply_project_payload(self, payload: dict) -> None:
        self._inputs.discard_channel_config_dialog()
        source_config = payload.get("source_config_by_name", {})
        if not isinstance(source_config, dict):
            source_config = {}
        channel_config = payload.get("channel_config_by_name", {})
        if not isinstance(channel_config, dict):
            channel_config = {}
        max_spl_limits = payload.get("max_spl_limits_by_channel", {})
        if not isinstance(max_spl_limits, dict):
            max_spl_limits = {}

        documents = generator_documents_from_payload(payload.get("generator_documents"))
        active_id = payload.get("active_generator_document_id")
        active_id = (
            active_id
            if any(document.id == active_id for document in documents)
            else (documents[0].id if documents else None)
        )
        symmetry = str(payload.get("symmetry", "off")).strip().lower()
        if symmetry not in {"off", "x", "xy"}:
            symmetry = "off"
        imported_meshes = self._mesh_entries_from_payload(payload.get("imported_meshes", []))
        raw_physical_system = payload.get("physical_system")
        physical_system = (
            physical_system_from_dict(raw_physical_system) if isinstance(raw_physical_system, dict) else None
        )
        if physical_system is not None and not bool(physical_system.metadata.get(AUTO_SEEDED_EXTERIOR_KEY, False)):
            source_config = {}
        component_channels = payload.get("component_channel_by_id", {})
        if not isinstance(component_channels, dict):
            component_channels = {}
        self._session.document = ProjectDocument(
            generator_documents=documents,
            active_generator_document_id=active_id,
            imported_meshes=tuple(
                ImportedMeshState(
                    name=mesh.name,
                    source_file=mesh.source_file,
                    cleaned_file=mesh.cleaned_file,
                    scale_factor=mesh.scale_factor,
                    translation_mm=mesh.translation_mm,
                    enabled=mesh.enabled,
                )
                for mesh in imported_meshes
            ),
            stitch_imported_meshes=bool(
                payload.get("stitch_exterior_meshes", payload.get("stitch_imported_meshes", False))
            ),
            symmetry=symmetry,
            source_config_by_name=source_config,
            channel_config_by_name=channel_config,
            max_spl_limits_by_channel=max_spl_limits,
            project_preferences=self.current_project_preferences(),
            physical_system=physical_system,
            component_channel_by_id={
                str(component_id): str(channel_name) for component_id, channel_name in component_channels.items()
            },
            observation_planes=observation_planes_from_payload(payload.get("observation_planes")),
        )
        self._geometry_store.generated_by_document_id = {}
        for document in self._project.generator_documents:
            result = self._inputs.result_from_generator_document(document)
            if result is not None:
                self._geometry_store.generated_by_document_id[document.id] = (
                    self._inputs.apply_saved_source_config_to_result(
                        result,
                        generator_mesh_name(document),
                    )
                )
        self._inputs.rebuild_generator_document_tabs()
        self._inputs.reconcile_symmetry_with_backend()
        self._geometry_store.clear_imported_radiators()
        try:
            self._inputs.apply_saved_imported_source_config(self._inputs.surface_tags_for_meshes())
        except Exception:
            self._geometry_store.clear_imported_radiators()
        self._inputs.ensure_seeded_exterior_system()

        self.project_state_changed.emit("project_loaded")
        self.solve_results_invalidated.emit("project_loaded")

    # -- Ath design import / export ----------------------------------------

    @Slot()
    def import_config(self) -> None:
        path = self._view.choose_open_file("Import Waveguide Design", ATH_CONFIG_FILTER)
        if path is None:
            return

        self.import_config_path(path)

    def import_config_path(self, path: Path, *, document_id: str | None = None) -> None:
        project = self._project
        try:
            config_text = path.read_text(encoding="utf-8")
            document = (
                next((item for item in project.generator_documents if item.id == document_id), None)
                if document_id
                else self._inputs.active_generator_document()
            )
            if document is None:
                document = new_generator_document(
                    unique_generator_name(path.stem, project.generator_documents),
                    config_text,
                )
                project.generator_documents = (*project.generator_documents, document)
                project.active_generator_document_id = document.id
            else:
                project.generator_documents = tuple(
                    with_ath_source_text(item, config_text) if item.id == document.id else item
                    for item in project.generator_documents
                )
            self._inputs.rebuild_generator_document_tabs()
            self._view.show_status(f"Imported {path}")
        except Exception as exc:
            self._view.show_error("Import failed", str(exc))

    @Slot()
    def export_config(self) -> None:
        path = self._view.choose_save_file("Export Waveguide Design", ATH_CONFIG_FILTER, "waveguide.cfg")
        if path is None:
            return

        if path.suffix == "":
            path = path.with_suffix(".cfg")

        try:
            document = self._inputs.active_generator_document()
            path.write_text("" if document is None else ath_source_text(document), encoding="utf-8")
            self._view.show_status(f"Exported {path}")
        except Exception as exc:
            self._view.show_error("Export failed", str(exc))
