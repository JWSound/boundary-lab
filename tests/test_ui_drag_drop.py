import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication, QTabWidget

from blab.ui.dialogs import MeshDropTable
from blab.ui.main_window import ADD_SCRIPT_TAB_LABEL, AthScriptEditor, MainWindow

_APP = QApplication.instance() or QApplication([])


class _MimeData:
    def __init__(self, urls: list[QUrl]):
        self._urls = urls

    def hasUrls(self) -> bool:
        return bool(self._urls)

    def urls(self) -> list[QUrl]:
        return self._urls


class _DropEvent:
    def __init__(self, paths: list[Path]):
        self._mime_data = _MimeData([QUrl.fromLocalFile(str(path)) for path in paths])

    def mimeData(self) -> _MimeData:
        return self._mime_data


def test_ath_script_editor_accepts_cfg_drop_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "waveguide.CFG"
    txt_path = tmp_path / "notes.txt"

    assert AthScriptEditor._cfg_drop_path(_DropEvent([txt_path, cfg_path])) == cfg_path
    assert AthScriptEditor._cfg_drop_path(_DropEvent([txt_path])) is None


def test_mesh_drop_table_filters_msh_paths(tmp_path: Path) -> None:
    mesh_path = tmp_path / "cabinet.MSH"
    cfg_path = tmp_path / "waveguide.cfg"

    assert MeshDropTable._msh_drop_paths(_DropEvent([cfg_path, mesh_path])) == [mesh_path]
    assert MeshDropTable._msh_drop_paths(_DropEvent([cfg_path])) == []


def test_empty_script_tabs_keep_add_drop_target() -> None:
    window = MainWindow.__new__(MainWindow)
    window.editor_tabs = QTabWidget()
    window.ath_scripts = ()
    window.active_ath_script_id = None

    window._rebuild_ath_script_tabs()

    assert window.editor_tabs.count() == 1
    assert window.editor_tabs.tabText(0) == ADD_SCRIPT_TAB_LABEL
    assert isinstance(window.editor_tabs.widget(0), AthScriptEditor)
    assert window.editor_tabs.widget(0).isReadOnly()
