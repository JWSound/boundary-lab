import sys
from types import SimpleNamespace

from blab import cli, plotting, postprocess, solver


def test_cli_dispatches_subcommand_with_prefixed_prog(monkeypatch) -> None:
    calls = {}

    def fake_import_module(name: str):
        calls["module"] = name

        def main(args, prog=None):
            calls["args"] = args
            calls["prog"] = prog

        return SimpleNamespace(main=main)

    monkeypatch.setattr(cli, "import_module", fake_import_module)
    monkeypatch.setattr(sys, "argv", ["blab", "plot", "--help"])

    cli.main()

    assert calls == {
        "module": "blab.plotting",
        "args": ["--help"],
        "prog": "blab plot",
    }


def test_cli_exposes_interface_conformer() -> None:
    assert cli.COMMAND_MODULES["conform-interface"] == "blab.interface_conform"


def test_cli_exposes_rocm_configuration() -> None:
    assert cli.COMMAND_MODULES["rocm"] == "blab.rocm"


def test_retired_solve_command_explains_physical_project_workflow() -> None:
    import pytest

    with pytest.raises(SystemExit, match="retired.*blab project validate.*blab project solve"):
        solver.main([])


def test_server_help_describes_physical_cpu_preview(capsys):
    import pytest

    from blab import server

    with pytest.raises(SystemExit) as exc:
        server.main(["--help"])
    assert exc.value.code == 0
    assert "private networks or hosted deployments" in " ".join(capsys.readouterr().out.split())


def test_postprocess_public_options_are_trimmed() -> None:
    help_text = postprocess._build_arg_parser().format_help()

    assert "--octave-smoothing" in help_text
    assert "--hor-ref-angle" in help_text
    assert "--vert-ref-angle" in help_text
    assert "--isobar-angle-samples-smooth" not in help_text
    assert "--isobar-freq-samples-smooth" not in help_text


def test_plot_public_options_are_trimmed() -> None:
    help_text = plotting._build_arg_parser().format_help()

    assert "--output-dir" in help_text
    assert "--isobar-interp-freq-factor" not in help_text
