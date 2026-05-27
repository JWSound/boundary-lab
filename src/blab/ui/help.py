"""Simple Markdown help browser for bundled Boundary Lab docs."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class HelpBrowserDialog(QDialog):
    def __init__(self, docs_dir: Path, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Help")
        self.resize(980, 650)
        self.docs_dir = docs_dir
        self.docs_by_title = _discover_docs(docs_dir)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(240)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)

        for title, path in self.docs_by_title.items():
            item = QTreeWidgetItem([title])
            item.setData(0, 0x0100, str(path))
            self.tree.addTopLevelItem(item)

        self.tree.currentItemChanged.connect(self._show_current_document)

        content = QHBoxLayout()
        content.addWidget(self.tree)
        content.addWidget(self.browser, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(content, 1)
        layout.addWidget(buttons)

        if self.tree.topLevelItemCount() > 0:
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
        else:
            self.browser.setMarkdown("# Boundary Lab Help\n\nHelp content has not been added yet.")

    def _show_current_document(self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None) -> None:
        if current is None:
            return
        path = Path(str(current.data(0, 0x0100)))
        try:
            self.browser.setMarkdown(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.browser.setPlainText(f"Unable to load help document: {exc}")


def _discover_docs(docs_dir: Path) -> dict[str, Path]:
    if not docs_dir.exists():
        return {}
    return {
        _doc_title(path): path
        for path in sorted(docs_dir.glob("*.md"))
        if path.is_file()
    }


def _doc_title(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line[2:].strip() or path.stem.replace("-", " ").title()
    except Exception:
        pass
    return path.stem.replace("-", " ").title()
