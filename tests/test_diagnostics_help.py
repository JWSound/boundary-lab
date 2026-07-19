from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

import blab.ui.diagnostics as diagnostics_module
from blab.ui.application_state import OperationState
from blab.ui.diagnostics import DiagnosticsDialog, collect_diagnostics, format_diagnostics
from blab.ui.dialogs import DONATE_BLURB, DONATE_QR_PATH, DONATE_URL
from blab.ui.main_window import MainWindow
from blab.ui.settings import GuiPreferences


def test_diagnostics_report_includes_preferences_and_dependencies() -> None:
    diagnostics = collect_diagnostics(GuiPreferences(gmres_tolerance=5e-4))
    text = format_diagnostics(diagnostics)

    assert "Boundary Lab:" in text
    assert "Python:" in text
    assert "Dependencies:" in text
    assert "gmsh:" in text
    assert "gmres_tolerance: 0.0005" in text
    assert "worker_count" not in text


def test_diagnostics_hides_server_url_from_preferences_and_context() -> None:
    server_url = "https://user:secret@example.test/solve"
    diagnostics = collect_diagnostics(
        GuiPreferences(solve_server_url=server_url),
        {
            "backend": {"server_url": server_url, "label": "Boundary Lab Server"},
            "operations": {"last_error": f"Request to {server_url} failed"},
        },
    )
    text = format_diagnostics(diagnostics)

    assert server_url not in text
    assert "solve_server_url" not in text
    assert "server_url" not in text
    assert "label: Boundary Lab Server" in text
    assert "Request to [hidden] failed" in text


def test_copy_to_clipboard_generates_report_when_output_is_empty(monkeypatch) -> None:
    copied: list[str] = []

    class OutputStub:
        text = ""

        def toPlainText(self) -> str:
            return self.text

    output = OutputStub()
    dialog = SimpleNamespace(
        output=output,
        run_diagnostics=lambda: setattr(output, "text", "diagnostic report"),
    )
    application = SimpleNamespace(clipboard=lambda: SimpleNamespace(setText=copied.append))
    monkeypatch.setattr(diagnostics_module, "QApplication", application)

    DiagnosticsDialog.copy_to_clipboard(dialog)

    assert copied == ["diagnostic report"]


def test_main_window_diagnostic_context_summarizes_current_state() -> None:
    controller = SimpleNamespace(state=OperationState(), last_error=None)
    window = SimpleNamespace(
        preferences=GuiPreferences(solve_backend="local"),
        geometry_controller=controller,
        solve_controller=SimpleNamespace(
            state=OperationState(),
            last_error=None,
            last_completion=None,
        ),
        ath_scripts=(),
        ath_results_by_script_id={},
        imported_meshes=(),
        project_path=None,
        _has_unsaved_project_changes=lambda: False,
        _all_radiators=lambda: (),
        _channel_configs=lambda: (),
        symmetry="none",
        stitch_imported_meshes=False,
        freq_min_spin=SimpleNamespace(value=lambda: 20),
        freq_max_spin=SimpleNamespace(value=lambda: 20_000),
        freq_count_spin=SimpleNamespace(value=lambda: 100),
        live_dataset=None,
    )

    context = MainWindow._diagnostic_context(window)

    assert context["backend"]["id"] == "local"
    assert context["project"]["file"] == "unsaved"
    assert context["operations"]["solve"]["phase"] == "idle"
    assert context["results"]["solved frequencies"] == 0


def test_donate_dialog_content_points_to_asset_and_paypal() -> None:
    assert DONATE_QR_PATH.exists()
    assert DONATE_QR_PATH.name == "donateqr.png"
    assert DONATE_URL == "https://www.paypal.com/donate/?hosted_button_id=ZVC2HAFBJNPDW"
    assert "Boundary Lab is free open source software." in DONATE_BLURB
