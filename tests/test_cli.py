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


def test_solver_accepts_short_public_option_names() -> None:
    args = solver._build_arg_parser().parse_args(
        [
            "mesh.msh",
            "--output-npz",
            "out.npz",
            "--step-size",
            "5",
            "--min-angle",
            "-90",
            "--max-angle",
            "90",
            "--axial-offset",
            "0.2",
            "--gmres-tol",
            "1e-4",
        ]
    )

    assert args.output_npz == "out.npz"
    assert args.step_size == 5
    assert args.min_angle == -90
    assert args.max_angle == 90
    assert args.axial_offset == 0.2
    assert args.gmres_tol == 1e-4


def test_solver_rejects_removed_legacy_option_names() -> None:
    parser = solver._build_arg_parser()

    for option in (
        "--output-npz-base-path",
        "--polar-angle-step-deg",
        "--polar-angle-min-deg",
        "--polar-angle-max-deg",
        "--observation-axial-offset-m",
    ):
        try:
            parser.parse_args(["mesh.msh", option, "1"])
        except SystemExit:
            continue
        raise AssertionError(f"{option} should not be accepted")


def test_solver_help_uses_short_public_option_names() -> None:
    help_text = solver._build_arg_parser().format_help()

    assert "--output-npz" in help_text
    assert "--step-size" in help_text
    assert "--min-angle" in help_text
    assert "--max-angle" in help_text
    assert "--axial-offset" in help_text
    assert "--gmres-tol" in help_text
    assert "--output-npz-base-path" not in help_text
    assert "--polar-angle-step-deg" not in help_text
    assert "--observation-axial-offset-m" not in help_text


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
