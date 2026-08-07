"""Generator design-document tabs and their backing editor state."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QInputDialog,
    QPlainTextEdit,
    QTabBar,
)

from blab.generators.ath import ATH_PROVIDER_ID, ath_source_text, with_ath_source_text
from blab.generators.base import GeneratedGeometry, GeneratorDocument
from blab.generators.registry import create_generator
from blab.ui.main_window.constants import (
    ADD_DESIGN_TAB_LABEL,
)
from blab.ui.main_window_widgets import (
    AthScriptEditor,
)
from blab.ui.project_state import (
    generator_mesh_name,
    new_generator_document,
    replace_generator_document,
    unique_generator_name,
)


class GeneratorDocumentsMixin:
    """Generator design-document tabs and their backing editor state.

    Mixed into :class:`~blab.ui.main_window.window.MainWindow`.
    """

    def rebuild_generator_document_tabs(self) -> None:
        self.editor_tabs.blockSignals(True)
        self.editor_tabs.clear()
        for document in self.generator_documents:
            editor = AthScriptEditor()
            editor.setFont(QFont("Consolas", 10))
            if document.provider_id == ATH_PROVIDER_ID:
                editor.setPlainText(ath_source_text(document))
                editor.textChanged.connect(
                    lambda document_id=document.id, editor=editor: self._update_generator_source_text(
                        document_id, editor
                    )
                )
                editor.configDropped.connect(
                    lambda path, document_id=document.id: self.import_config_path(Path(path), document_id=document_id)
                )
            else:
                editor.setPlainText(json.dumps(document.source, indent=2, sort_keys=True))
                editor.setReadOnly(True)
            self.editor_tabs.addTab(editor, document.name)
        add_tab = AthScriptEditor()
        add_tab.setReadOnly(True)
        add_tab.configDropped.connect(lambda path: self.import_config_path(Path(path)))
        add_index = self.editor_tabs.addTab(add_tab, ADD_DESIGN_TAB_LABEL)
        self.editor_tabs.tabBar().setTabButton(add_index, QTabBar.ButtonPosition.RightSide, None)
        self.editor_tabs.tabBar().setTabToolTip(add_index, "Add waveguide design")
        active_index = self.active_generator_document_index()
        if active_index >= 0:
            self.editor_tabs.setCurrentIndex(active_index)
        self.editor_tabs.blockSignals(False)

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

    @Slot()
    def add_generator_document(self) -> None:
        name = unique_generator_name("waveguide", self.generator_documents)
        document = new_generator_document(name, "")
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
            return create_generator(document.provider_id).restore(document)
        except Exception:
            return None
