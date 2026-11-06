import sys
import time

import pytest
from PySide6.QtCore import QSettings

from blab.ath_setup import ATH_PATH_KEY, WINE_PATH_KEY
from blab.ui.ath_setup import AthSetupDialog


@pytest.fixture
def setup_dialog(qapp, tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    monkeypatch.setattr("blab.ui.ath_setup.wine_prefix", lambda: tmp_path / "prefix")
    return AthSetupDialog(settings)


def test_cancel_does_not_persist_paths(setup_dialog):
    setup_dialog.ath.setText("not saved.exe")
    setup_dialog.reject()
    assert not setup_dialog.settings.contains(ATH_PATH_KEY)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable fixture")
@pytest.mark.parametrize("success", [True, False])
def test_probe_checks_generation_and_saves_only_on_accept(setup_dialog, tmp_path, qapp, success):
    ath = tmp_path / "ath test.exe"
    ath.write_text("fixture")
    wine = tmp_path / "wine test"
    wine.write_text(
        '#!/bin/sh\nif [ "$1" = "--version" ]; then echo wine-test; exit 0; fi\n'
        + ('echo "Point(1) = {0,0,0,1};"\n' if success else 'echo "generation failed" >&2; exit 1\n')
    )
    wine.chmod(0o700)
    setup_dialog.ath.setText(str(ath))
    setup_dialog.wine.setText(str(wine))
    setup_dialog._test()
    deadline = time.monotonic() + 5
    while setup_dialog._stage and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert not setup_dialog._stage
    assert ("test passed" in setup_dialog.status.text()) is success
    assert setup_dialog._temporary is None
    assert not setup_dialog.settings.contains(ATH_PATH_KEY)
    if success:
        setup_dialog.accept()
        assert setup_dialog.settings.value(ATH_PATH_KEY) == str(ath)
        assert setup_dialog.settings.value(WINE_PATH_KEY) == str(wine)


def test_setup_menu_is_macos_only(main_window):
    labels = [
        action.text() for menu in main_window.menuBar().actions() if menu.menu() for action in menu.menu().actions()
    ]
    assert ("Ath Setup…" in labels) == (sys.platform == "darwin")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable fixture")
def test_probe_cancellation_releases_process_and_scratch(setup_dialog, tmp_path, qapp):
    ath, wine = tmp_path / "ath.exe", tmp_path / "wine"
    ath.touch()
    wine.write_text("#!/bin/sh\nexec /bin/sleep 30\n")
    wine.chmod(0o700)
    setup_dialog.ath.setText(str(ath))
    setup_dialog.wine.setText(str(wine))
    setup_dialog._test()
    assert setup_dialog.process.waitForStarted(1000)
    setup_dialog.reject()
    deadline = time.monotonic() + 5
    while setup_dialog._stage and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert not setup_dialog._stage
    assert setup_dialog._temporary is None
    assert setup_dialog.status.text() == "Test cancelled."
    assert not setup_dialog.settings.contains(ATH_PATH_KEY)
