import pytest

pytest.importorskip("PySide6")

from blab.ui.diagnostics import collect_diagnostics, format_diagnostics
from blab.ui.dialogs import DONATE_BLURB, DONATE_QR_PATH, DONATE_URL
from blab.ui.settings import GuiPreferences


def test_diagnostics_report_includes_preferences_and_dependencies() -> None:
    diagnostics = collect_diagnostics(GuiPreferences(gmres_tolerance=5e-4))
    text = format_diagnostics(diagnostics)

    assert "Boundary Lab:" in text
    assert "Python:" in text
    assert "Dependencies:" in text
    assert "gmres_tolerance: 0.0005" in text
    assert "worker_count" not in text


def test_donate_dialog_content_points_to_asset_and_paypal() -> None:
    assert DONATE_QR_PATH.exists()
    assert DONATE_QR_PATH.name == "donateqr.png"
    assert DONATE_URL == "https://www.paypal.com/donate/?hosted_button_id=ZVC2HAFBJNPDW"
    assert "Boundary Lab is free open source software." in DONATE_BLURB

