"""Simple Markdown help browser for bundled Boundary Lab docs."""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QTextDocument
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


_HELP_STYLESHEET = """
body {
    font-family: "Segoe UI", "Arial", sans-serif;
    font-size: 14px;
    line-height: 1.45;
}
h1 {
    font-size: 28px;
    margin-top: 0;
    margin-bottom: 18px;
}
h2 {
    font-size: 21px;
    margin-top: 26px;
    margin-bottom: 12px;
}
h3 {
    font-size: 17px;
    margin-top: 20px;
    margin-bottom: 8px;
}
p {
    margin-top: 0;
    margin-bottom: 12px;
}
ul, ol {
    margin-top: 4px;
    margin-bottom: 14px;
}
li {
    margin-bottom: 4px;
}
code {
    font-family: "Cascadia Mono", "Consolas", monospace;
    background-color: #f1f3f5;
}
pre {
    font-family: "Cascadia Mono", "Consolas", monospace;
    background-color: #f6f8fa;
    border: 1px solid #d0d7de;
    padding: 10px;
}
img {
    max-width: 100%;
}
"""

_MATH_BLOCK_RE = re.compile(r"^\$\$\s*\n(.*?)\n\$\$\s*$", re.MULTILINE | re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$([^$\n]+)\$(?!\$)")
_HTML_IMG_RE = re.compile(r"<img\s+([^>]*?)\s*/?>", re.IGNORECASE)
_HTML_ATTR_RE = re.compile(r"""([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*["']([^"']*)["']""")
_MARKDOWN_FEATURES = QTextDocument.MarkdownDialectGitHub | QTextDocument.MarkdownNoHTML


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
        self.browser.document().setDefaultStyleSheet(_HELP_STYLESHEET)

        default_item = None
        for title, path in self.docs_by_title.items():
            item = QTreeWidgetItem([title])
            item.setData(0, 0x0100, str(path))
            self.tree.addTopLevelItem(item)
            if title == "User Guide":
                default_item = item

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
            self.tree.setCurrentItem(default_item or self.tree.topLevelItem(0))
        else:
            _set_browser_markdown(self.browser, "# Boundary Lab Help\n\nHelp content has not been added yet.")

    def _show_current_document(self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None) -> None:
        if current is None:
            return
        path = Path(str(current.data(0, 0x0100)))
        try:
            self.browser.document().setBaseUrl(_document_base_url(path))
            _set_browser_markdown(self.browser, path.read_text(encoding="utf-8"))
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
    if path.stem.casefold() == "user guide":
        return "User Guide"
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line[2:].strip() or path.stem.replace("-", " ").title()
    except Exception:
        pass
    return path.stem.replace("-", " ").title()


def _document_base_url(path: Path) -> QUrl:
    return QUrl.fromLocalFile(str(path.resolve().parent) + "/")


def _markdown_for_qt(markdown: str) -> str:
    """Adjust Markdown features that QTextBrowser does not render well."""

    markdown = _HTML_IMG_RE.sub(_html_image_as_markdown, markdown)
    markdown = _MATH_BLOCK_RE.sub(_math_block_as_code, markdown)
    markdown = _INLINE_MATH_RE.sub(r"`\1`", markdown)
    return markdown


def _set_browser_markdown(browser: QTextBrowser, markdown: str) -> None:
    browser.document().setMarkdown(_markdown_for_qt(markdown), _MARKDOWN_FEATURES)


def _html_image_as_markdown(match: re.Match[str]) -> str:
    attrs = {key.casefold(): value for key, value in _HTML_ATTR_RE.findall(match.group(1))}
    src = attrs.get("src")
    if not src:
        return ""
    alt = attrs.get("alt", "")
    return f"![{alt}]({src})"


def _math_block_as_code(match: re.Match[str]) -> str:
    body = match.group(1).strip("\n")
    return f"```math\n{body}\n```"
