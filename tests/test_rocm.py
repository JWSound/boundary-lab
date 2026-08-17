from __future__ import annotations

import json
from pathlib import Path

import pytest

from blab.rocm import (
    default_rocm_config_path,
    discover_rocm,
    inspect_rocm_root,
    main,
    save_configured_rocm_path,
)


def _fake_rocm(root: Path, *, hip_runtime: str = "amdhip64_7.dll") -> Path:
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    for name in (hip_runtime, "rocblas.dll", "rocsolver.dll", "rocsparse.dll", "hipconfig.exe"):
        (bin_dir / name).touch()
    return root


def test_inspect_rocm_root_accepts_versioned_windows_hip_runtime(tmp_path: Path) -> None:
    root = _fake_rocm(tmp_path / "ROCm" / "7.2")

    installation = inspect_rocm_root(root)

    assert installation.valid
    assert installation.root == root.resolve()
    assert installation.missing == ()


def test_inspect_rocm_root_reports_missing_solver_libraries(tmp_path: Path) -> None:
    root = tmp_path / "incomplete"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "amdhip64.dll").touch()
    (root / "bin" / "hipInfo.exe").touch()

    installation = inspect_rocm_root(root)

    assert not installation.valid
    assert installation.missing == ("rocBLAS", "rocSOLVER", "rocSPARSE")


def test_discovery_prefers_boundary_lab_override_then_saved_config(tmp_path: Path) -> None:
    override = _fake_rocm(tmp_path / "override")
    configured = _fake_rocm(tmp_path / "configured")
    hip_path = _fake_rocm(tmp_path / "hip")
    config_path = tmp_path / "rocm_path.txt"
    save_configured_rocm_path(configured, config_path)

    selected = discover_rocm(
        environ={"BLAB_ROCM_PATH": str(override), "HIP_PATH": str(hip_path)},
        config_path=config_path,
        program_files_roots=(),
    )
    configured_selected = discover_rocm(
        environ={"HIP_PATH": str(hip_path)},
        config_path=config_path,
        program_files_roots=(),
    )

    assert selected is not None
    assert selected.root == override.resolve()
    assert selected.source == "BLAB_ROCM_PATH"
    assert configured_selected is not None
    assert configured_selected.root == configured.resolve()
    assert configured_selected.source == "config"


def test_discovery_skips_invalid_environment_candidate(tmp_path: Path) -> None:
    valid = _fake_rocm(tmp_path / "valid")

    selected = discover_rocm(
        environ={"BLAB_ROCM_PATH": str(tmp_path / "missing"), "HIP_PATH": str(valid)},
        config_path=tmp_path / "absent.txt",
        program_files_roots=(),
    )

    assert selected is not None
    assert selected.root == valid.resolve()
    assert selected.source == "HIP_PATH"


def test_program_files_candidates_are_sorted_by_numeric_version(tmp_path: Path) -> None:
    rocm_parent = tmp_path / "AMD" / "ROCm"
    _fake_rocm(rocm_parent / "7.9")
    newest = _fake_rocm(rocm_parent / "10.1")
    _fake_rocm(rocm_parent / "7.12")

    selected = discover_rocm(
        environ={"ProgramFiles": str(tmp_path)},
        config_path=tmp_path / "absent.txt",
    )

    assert selected is not None
    assert selected.root == newest.resolve()
    assert selected.source == "program_files"


def test_default_config_path_uses_local_app_data() -> None:
    path = default_rocm_config_path({"LOCALAPPDATA": r"C:\Users\Tester\AppData\Local"})

    assert path == Path(r"C:\Users\Tester\AppData\Local") / "Boundary Lab" / "rocm_path.txt"


def test_cli_detect_writes_root_file_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _fake_rocm(tmp_path / "sdk")
    root_file = tmp_path / "selected.txt"

    with pytest.raises(SystemExit) as exit_info:
        main(["detect", "--path", str(root), "--root-file", str(root_file), "--json"], prog="blab rocm")

    assert exit_info.value.code == 0
    assert root_file.read_text(encoding="utf-8").strip() == str(root.resolve())
    assert json.loads(capsys.readouterr().out)["selected"]["valid"] is True
