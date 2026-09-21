"""Generator design-document tabs and their backing editor state."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QDockWidget,
    QInputDialog,
    QPlainTextEdit,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from blab.generators.application import stage_generation
from blab.generators.ath import ATH_PROVIDER_ID, with_ath_source_text
from blab.generators.base import GeneratedGeometry, GenerationCompleted, GeneratorDocument
from blab.generators.catalog import provider_catalog
from blab.generators.configuration import configuration_snapshot, project_revision
from blab.generators.registry import generator_info, restore_generator_document
from blab.project.model import (
    generator_mesh_name,
    new_generator_document,
    replace_generator_document,
    unique_generator_name,
)
from blab.ui.ath_editor import AthScriptEditor
from blab.ui.main_window.constants import (
    ADD_DESIGN_TAB_LABEL,
)
from blab.ui.main_window_widgets import TabCloseButton
from blab.ui.provider_editor import AthProviderEditor, DocumentHost
from blab.ui.settings import save_syntax_highlighting_enabled


class GeneratorDocumentsMixin:
    """Generator design-document tabs and their backing editor state.

    Mixed into :class:`~blab.ui.main_window.window.MainWindow`.
    """

    def _generation_project_snapshot(self):
        project = deepcopy(self.project)
        project.project_preferences = self.project_workflow.current_project_preferences()
        return project

    def generation_context(self) -> tuple[str, dict]:
        project = self._generation_project_snapshot()
        return project_revision(project), configuration_snapshot(project)

    def accept_generation(self, completed: GenerationCompleted) -> GeneratedGeometry:
        """Commit provider settings and artifact only after the candidate validates."""
        snapshot = self._generation_project_snapshot()
        request = completed.request
        if request.project_revision != project_revision(snapshot):
            raise ValueError("Generation discarded because project inputs changed.")
        document = next((item for item in snapshot.generator_documents if item.id == request.document_id), None)
        if document is None or document.source != request.source or document.provider_id != request.provider_id:
            raise ValueError("Generation discarded because the provider source changed.")
        result = self.apply_saved_source_config_to_result(completed.result, request.mesh_name)
        completed = replace(completed, result=result)
        # Legacy providers retain their existing exterior migration behavior.
        # A configuration response is staged independently before any live mutation.
        if not completed.legacy_response:
            candidate = stage_generation(
                snapshot,
                self.generated_geometry_by_document_id,
                completed,
                imported_radiators=self.imported_radiators,
            )
            self.project = candidate
            self.generated_geometry_by_document_id[request.document_id] = result
            stitching = completed.configuration_patch.get("stitching_config", {})
            if "tolerance_mm" in stitching:
                self.preferences.stitch_tolerance_mm = candidate.project_preferences.stitch_tolerance_mm
            self.discard_channel_config_dialog()
        else:
            self.record_generated_geometry(request.document_id, result)
            self.ensure_seeded_exterior_system()
        return result

    def dispose_provider_editors(self) -> None:
        for editor, host in getattr(self, "_provider_editors", {}).values():
            host.dispose()
            try:
                editor.dispose()
            except Exception:
                logging.getLogger(__name__).exception("Provider editor disposal failed")
        self._provider_editors = {}

    def update_provider_editor_states(self, _state=None) -> None:
        state = self.geometry_controller.state if self.geometry_controller.active else self.solve_controller.state
        for editor, _host in getattr(self, "_provider_editors", {}).values():
            try:
                editor.set_operation_state(state)
            except Exception:
                logging.getLogger(__name__).exception("Provider editor state update failed")

    def rebuild_generator_document_tabs(self) -> None:
        self.dispose_provider_editors()
        self.editor_tabs.blockSignals(True)
        while self.editor_tabs.count():
            widget = self.editor_tabs.widget(0)
            self.editor_tabs.removeTab(0)
            widget.deleteLater()
        for document in self.generator_documents:
            host = DocumentHost(self, document)
            adapter = None
            container = None
            try:
                info = generator_info(document.provider_id)
                if document.provider_schema_version != info.source_schema_version:
                    raise ValueError("Incompatible provider source schema; source has been preserved.")
                if document.provider_id == ATH_PROVIDER_ID:
                    adapter = AthProviderEditor(
                        self.editor_tabs, host, highlight_syntax=self.syntax_highlighting_enabled
                    )
                    adapter.widget.configDropped.connect(
                        lambda path, document_id=document.id: self.import_config_path(
                            Path(path), document_id=document_id
                        )
                    )
                else:
                    factory = provider_catalog().factory(document.provider_id, "editor")
                    if factory is None:
                        raise ValueError("This provider does not supply a custom editor.")
                    # Own even partially constructed child widgets if a factory raises.
                    container = QWidget(self.editor_tabs)
                    container.hide()
                    layout = QVBoxLayout(container)
                    layout.setContentsMargins(0, 0, 0, 0)
                    adapter = factory(container, host)
                if (
                    not isinstance(adapter.widget, QWidget)
                    or isinstance(adapter.widget, QDockWidget)
                    or adapter.widget is self.editor_tabs
                    or adapter.widget is container
                ):
                    raise TypeError("Provider must supply its own widget.")
                for method in ("apply_source", "set_operation_state", "dispose"):
                    if not callable(getattr(adapter, method, None)):
                        raise TypeError(f"Provider editor is missing {method}().")
                snapshot = host.snapshot()
                adapter.apply_source(snapshot.source, snapshot.revision)
                editor = adapter.widget
                if container is not None:
                    layout.addWidget(editor)
                    editor = container
                self._provider_editors[document.id] = (adapter, host)
            except Exception as exc:
                host.dispose()
                if container is not None:
                    container.deleteLater()
                if adapter is not None:
                    try:
                        adapter.dispose()
                        adapter.widget.deleteLater()
                    except Exception:
                        logging.getLogger(__name__).exception("Failed editor cleanup")
                editor = QPlainTextEdit()
                editor.setReadOnly(True)
                editor.setPlainText(
                    f"{document.provider_id}: {exc}\n\n" + json.dumps(document.source, indent=2, sort_keys=True)
                )
            self._install_tab_close_button(self.editor_tabs.addTab(editor, document.name), document.name)
        add_tab = AthScriptEditor(highlight_syntax=False)
        add_tab.setReadOnly(True)
        add_tab.configDropped.connect(lambda path: self.import_config_path(Path(path)))
        add_index = self.editor_tabs.addTab(add_tab, ADD_DESIGN_TAB_LABEL)
        self.editor_tabs.tabBar().setTabButton(add_index, QTabBar.ButtonPosition.RightSide, None)
        self.editor_tabs.tabBar().setTabToolTip(add_index, "Add waveguide design")
        active_index = self.active_generator_document_index()
        if active_index >= 0:
            self.editor_tabs.setCurrentIndex(active_index)
        self.editor_tabs.blockSignals(False)
        self.update_provider_editor_states()
        self._refresh_generate_availability()

    def _install_tab_close_button(self, index: int, name: str) -> None:
        button = TabCloseButton(f"Close {name}")
        # Resolved on click: removing a tab renumbers the rest.
        button.clicked.connect(lambda: self._remove_generator_document_at(self._tab_index_of_close_button(button)))
        self.editor_tabs.tabBar().setTabButton(index, QTabBar.ButtonPosition.RightSide, button)

    def _tab_index_of_close_button(self, button: TabCloseButton) -> int:
        tab_bar = self.editor_tabs.tabBar()
        for index in range(tab_bar.count()):
            if tab_bar.tabButton(index, QTabBar.ButtonPosition.RightSide) is button:
                return index
        return -1

    @Slot(bool)
    def set_syntax_highlighting_enabled(self, enabled: bool) -> None:
        self.syntax_highlighting_enabled = bool(enabled)
        save_syntax_highlighting_enabled(self.settings, self.syntax_highlighting_enabled)
        for index, document in enumerate(self.generator_documents):
            editor = self.editor_tabs.widget(index)
            if document.provider_id == ATH_PROVIDER_ID and isinstance(editor, AthScriptEditor):
                editor.set_syntax_highlighting_enabled(self.syntax_highlighting_enabled)

    def active_generator_document_index(self) -> int:
        for index, document in enumerate(self.generator_documents):
            if document.id == self.active_generator_document_id:
                return index
        return 0 if self.generator_documents else -1

    def active_generator_document(self) -> GeneratorDocument | None:
        if not self.generator_documents:
            return None
        index = self.active_generator_document_index()
        return self.generator_documents[index] if index >= 0 else None

    def record_generated_geometry(self, document_id: str, result) -> None:
        """File fresh geometry against its design and reference it on the document."""
        self.generated_geometry_by_document_id[document_id] = result
        self.generator_documents = replace_generator_document(
            self.generator_documents,
            document_id,
            artifact=result.to_reference(),
        )

    def _update_generator_source_text(self, document_id: str, editor: QPlainTextEdit) -> None:
        self.generator_documents = tuple(
            with_ath_source_text(document, editor.toPlainText()) if document.id == document_id else document
            for document in self.generator_documents
        )

    def _on_active_generator_tab_changed(self, index: int) -> None:
        if index == len(self.generator_documents):
            self.add_generator_document()
            return
        if 0 <= index < len(self.generator_documents):
            self.active_generator_document_id = self.generator_documents[index].id
            self._refresh_generate_availability()

    def _refresh_generate_availability(self):
        if not hasattr(self, "generate_button"):
            return
        document = self.active_generator_document()
        try:
            info = generator_info(document.provider_id) if document else None
            available = bool(info and info.available and info.source_schema_version == document.provider_schema_version)
        except ValueError:
            available = False
        self.generate_button.setEnabled(
            available
            and not self.geometry_controller.active
            and not self.solve_controller.active
            and not self.preparations.active
        )

    def new_default_generator_document(self, name):
        provider_id = self.preferences.default_geometry_provider
        if provider_id == ATH_PROVIDER_ID:
            return new_generator_document(name, "")
        # Defaults are declarative: creating a design never instantiates a backend.
        manifest = provider_catalog().package(provider_id).manifest
        return replace(
            new_generator_document(
                name,
                provider_id=provider_id,
                provider_schema_version=manifest.source_schema_version,
                source=deepcopy(manifest.default_source),
            ),
            mesh_scale_factor=manifest.mesh_scale_factor,
        )

    @Slot()
    def add_generator_document(self) -> None:
        name = unique_generator_name("waveguide", self.generator_documents)
        try:
            document = self.new_default_generator_document(name)
        except ValueError as exc:
            self.show_error("Geometry provider unavailable", str(exc))
            self.editor_tabs.setCurrentIndex(self.active_generator_document_index())
            return
        self.generator_documents = (*self.generator_documents, document)
        self.active_generator_document_id = document.id
        self.rebuild_generator_document_tabs()

    @Slot()
    def rename_active_generator_document(self) -> None:
        document = self.active_generator_document()
        if document is None:
            return
        name, accepted = QInputDialog.getText(
            self,
            "Rename Waveguide Design",
            "Design name:",
            text=document.name,
        )
        if not accepted:
            return
        name = name.strip()
        if not name:
            return
        self.generator_documents = replace_generator_document(
            self.generator_documents,
            document.id,
            name=unique_generator_name(
                name,
                tuple(item for item in self.generator_documents if item.id != document.id),
            ),
        )
        self.rebuild_generator_document_tabs()
        self.mesh_state_changed.emit("generator_document_renamed")
        self.solve_results_invalidated.emit("generator_document_renamed")

    def _remove_generator_document_at(self, index: int) -> None:
        if not (0 <= index < len(self.generator_documents)):
            return
        document = self.generator_documents[index]
        self.generator_documents = tuple(item for item in self.generator_documents if item.id != document.id)
        self.generated_geometry_by_document_id.pop(document.id, None)
        self.active_generator_document_id = (
            self.generator_documents[min(index, len(self.generator_documents) - 1)].id
            if self.generator_documents
            else None
        )
        self.rebuild_generator_document_tabs()
        self.mesh_state_changed.emit("generator_document_removed")
        self.solve_results_invalidated.emit("generator_document_removed")

    def _generator_document_for_mesh_name(self, mesh_name: str) -> GeneratorDocument | None:
        return next(
            (document for document in self.generator_documents if generator_mesh_name(document) == mesh_name),
            None,
        )

    def result_from_generator_document(self, document: GeneratorDocument) -> GeneratedGeometry | None:
        try:
            return restore_generator_document(document)
        except Exception:
            return None
