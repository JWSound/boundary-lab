"""Ath4 command-line integration helpers."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import meshio

from blab.config import RadiatorConfig
from blab.mesh_clean import (
    AREA_TOL,
    MERGE_TOL,
    MeshQualityWarning,
    clean_mesh_file,
    triangle_quality_warning,
)

DRIVEN_DIAPHRAGM_PHYSICAL_NAME = "SD1D1001"
COMPLEX_RADIATOR_DRIVES_DB = {
    "SD1D1003": 0.0,
    "SD1D1002": -2.5,
    "SD1D1001": -12.0,
}
COMPLEX_RADIATOR_NAMES = {
    "SD1D1003": "dome",
    "SD1D1002": "surround_inner",
    "SD1D1001": "surround_outer",
}
DEFAULT_CLEAN_SUFFIX = "_clean"
ATH_CFG_OUTPUT_ROOT_KEY = "OutputRootDir"
ATH_CFG_MESH_CMD_KEY = "MeshCmd"
SOLVING_SYM_RE = re.compile(r"\bSym\s*=\s*([A-Za-z]+)\b")


class AthCancelledError(RuntimeError):
    """Raised when an active Ath generation is cancelled by the user."""


@dataclass(frozen=True)
class AthRunResult:
    output_dir: Path
    msh_path: Path
    config_path: Path
    driven_tag: int
    radiators: tuple[RadiatorConfig, ...]
    cleaned_msh_path: Path | None = None
    reduced_cleaned_msh_path: Path | None = None
    quality_warning: MeshQualityWarning | None = None

    @property
    def solver_msh_path(self) -> Path:
        return self.cleaned_msh_path or self.msh_path

    def solver_msh_path_for_symmetry(self, symmetry: str) -> Path:
        if str(symmetry or "off").strip().lower() == "off":
            return self.solver_msh_path
        return self.reduced_cleaned_msh_path or self.msh_path


class AthProcessRunner:
    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._cancel_requested = False

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    def run(
        self,
        *,
        ath_exe: Path,
        config_text: str,
        run_root: Path,
        case_name: str = "waveguide",
        timeout_s: float | None = None,
    ) -> AthRunResult:
        ath_exe = ath_exe.resolve()
        if not ath_exe.exists():
            raise FileNotFoundError(f"Ath executable not found: {ath_exe}")

        ath_companion_config = ath_exe.parent / "ath.cfg"
        if not ath_companion_config.exists():
            raise FileNotFoundError(
                f"Ath companion config not found: {ath_companion_config}. Ath expects ath.cfg beside ath.exe."
            )

        run_root.mkdir(parents=True, exist_ok=True)
        config_path = run_root / f"{case_name}.cfg"
        config_path.write_text(config_text, encoding="utf-8")
        output_root = read_ath_output_root(ath_companion_config) or ath_exe.parent

        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        self._cancel_requested = False
        self._process = subprocess.Popen(
            [str(ath_exe), str(config_path)],
            cwd=ath_exe.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags,
        )
        try:
            stdout, stderr = self._process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.stop()
            raise
        finally:
            process = self._process
            self._process = None

        if self._cancel_requested:
            raise AthCancelledError("Ath generation cancelled")

        returncode = 0 if process is None else process.returncode
        if returncode != 0:
            message = stderr.strip() or stdout.strip()
            raise RuntimeError(f"Ath failed with exit code {returncode}: {message}")

        return discover_ath_output(
            run_root=output_root,
            case_name=case_name,
            config_path=config_path,
        )

    def stop(self) -> None:
        self._cancel_requested = True
        process = self._process
        if process is None or process.poll() is not None:
            return
        if os.name == "nt":
            try:
                completed = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5.0,
                )
                if completed.returncode == 0:
                    return
            except (OSError, subprocess.TimeoutExpired):
                pass
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)


def run_ath(
    *,
    ath_exe: Path,
    config_text: str,
    run_root: Path,
    case_name: str = "waveguide",
    timeout_s: float | None = None,
) -> AthRunResult:
    return AthProcessRunner().run(
        ath_exe=ath_exe,
        config_text=config_text,
        run_root=run_root,
        case_name=case_name,
        timeout_s=timeout_s,
    )


def read_ath_output_root(ath_cfg_path: Path) -> Path | None:
    with ath_cfg_path.open("r", encoding="utf-8", errors="replace") as cfg_file:
        for raw_line in cfg_file:
            line = raw_line.strip()
            if not line or line.startswith(";") or "=" not in line:
                continue
            key, value = line.split("=", maxsplit=1)
            if key.strip() != ATH_CFG_OUTPUT_ROOT_KEY:
                continue
            output_root = value.strip().strip('"')
            if not output_root:
                return None
            return Path(output_root)
    return None


def _write_ath_cfg_value(ath_cfg_path: Path, key_name: str, value: str) -> None:
    config_line = f'{key_name} = "{value}"\n'
    ath_cfg_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    replaced = False
    if ath_cfg_path.exists():
        lines = ath_cfg_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        for index, raw_line in enumerate(lines):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith(";") or "=" not in stripped:
                continue
            key, _value = stripped.split("=", maxsplit=1)
            if key.strip() == key_name:
                lines[index] = config_line
                replaced = True
                break

    if not replaced:
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] = f"{lines[-1]}\n"
        lines.insert(0, config_line)

    ath_cfg_path.write_text("".join(lines), encoding="utf-8", newline="")


def write_ath_output_root(ath_cfg_path: Path, output_root: Path) -> Path:
    """Ensure ath.cfg contains an absolute OutputRootDir value."""
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_ath_cfg_value(ath_cfg_path, ATH_CFG_OUTPUT_ROOT_KEY, str(output_root))
    return output_root


def write_ath_gmsh_path(ath_cfg_path: Path, gmsh_exe_path: Path) -> Path:
    """Ensure ath.cfg points Ath's MeshCmd at an absolute Gmsh executable."""
    gmsh_exe_path = gmsh_exe_path.resolve()
    _write_ath_cfg_value(ath_cfg_path, ATH_CFG_MESH_CMD_KEY, f"{gmsh_exe_path} %f -")
    return gmsh_exe_path


def discover_ath_output(*, run_root: Path, case_name: str, config_path: Path | None = None) -> AthRunResult:
    output_dir = run_root / case_name
    if not output_dir.exists():
        candidates = sorted(run_root.glob(f"{case_name}*"))
        dirs = [path for path in candidates if path.is_dir()]
        if not dirs:
            raise FileNotFoundError(f"Ath output directory not found under {run_root}")
        output_dir = dirs[0]

    msh_root = output_dir / "ABEC_FreeStanding"
    msh_path = msh_root / f"{case_name}.msh"
    if not msh_path.exists():
        msh_matches = sorted(output_dir.rglob("*.msh"))
        if not msh_matches:
            raise FileNotFoundError(f"No MSH output found in {output_dir}")
        msh_path = msh_matches[0]

    physical_names = read_physical_names(msh_path)
    driven_tag = physical_names[DRIVEN_DIAPHRAGM_PHYSICAL_NAME]
    return AthRunResult(
        output_dir=output_dir,
        msh_path=msh_path,
        config_path=config_path or output_dir / "config.txt",
        driven_tag=driven_tag,
        radiators=detect_ath_radiators(msh_path),
    )


def find_physical_tag_by_name(msh_path: Path, physical_name: str) -> int:
    physical_names = read_physical_names(msh_path)
    try:
        return physical_names[physical_name]
    except KeyError as exc:
        raise ValueError(f"Physical group '{physical_name}' not found in {msh_path}") from exc


def read_physical_names(msh_path: Path) -> dict[str, int]:
    physical_names = {}
    in_names = False
    with msh_path.open("r", encoding="utf-8", errors="replace") as mesh_file:
        for raw_line in mesh_file:
            line = raw_line.strip()
            if line == "$PhysicalNames":
                in_names = True
                continue
            if line == "$EndPhysicalNames":
                break
            if not in_names or not line or line.isdigit():
                continue

            parts = line.split(maxsplit=2)
            if len(parts) != 3:
                continue
            _, tag_text, name_text = parts
            name = name_text.strip().strip('"')
            physical_names[name] = int(tag_text)

    return physical_names


def read_surface_physical_names(msh_path: Path) -> dict[str, int]:
    surface_names = {}
    in_names = False
    with msh_path.open("r", encoding="utf-8", errors="replace") as mesh_file:
        for raw_line in mesh_file:
            line = raw_line.strip()
            if line == "$PhysicalNames":
                in_names = True
                continue
            if line == "$EndPhysicalNames":
                break
            if not in_names or not line or line.isdigit():
                continue

            parts = line.split(maxsplit=2)
            if len(parts) != 3:
                continue
            dimension_text, tag_text, name_text = parts
            if int(dimension_text) != 2:
                continue
            name = name_text.strip().strip('"')
            surface_names[name] = int(tag_text)

    return surface_names


def detect_ath_radiators(msh_path: Path) -> tuple[RadiatorConfig, ...]:
    physical_names = read_physical_names(msh_path)
    if set(COMPLEX_RADIATOR_DRIVES_DB).issubset(physical_names):
        return tuple(
            RadiatorConfig(
                name=COMPLEX_RADIATOR_NAMES[physical_name],
                tag=physical_names[physical_name],
                level_db=level_db,
            )
            for physical_name, level_db in COMPLEX_RADIATOR_DRIVES_DB.items()
        )

    if DRIVEN_DIAPHRAGM_PHYSICAL_NAME in physical_names:
        return (
            RadiatorConfig(
                name="throat",
                tag=physical_names[DRIVEN_DIAPHRAGM_PHYSICAL_NAME],
                level_db=0.0,
            ),
        )

    return ()


def ath_mirror_axes_from_solving_file(solving_path: Path) -> tuple[str, ...]:
    if not solving_path.exists():
        return ()

    text = solving_path.read_text(encoding="utf-8", errors="replace")
    match = SOLVING_SYM_RE.search(text)
    if match is None:
        return ()

    axes = []
    for axis in match.group(1).lower():
        if axis not in {"x", "y", "z"}:
            continue
        if axis not in axes:
            axes.append(axis)
    return tuple(axes)


def ath_mirror_axes_for_result(result: AthRunResult) -> tuple[str, ...]:
    candidates = (
        result.msh_path.parent / "solving.txt",
        result.output_dir / "solving.txt",
    )
    for solving_path in candidates:
        axes = ath_mirror_axes_from_solving_file(solving_path)
        if axes:
            return axes
    return ()


def clean_ath_mesh_output(
    result: AthRunResult,
    *,
    output_path: Path | None = None,
    merge_tol: float = MERGE_TOL,
    area_tol: float = AREA_TOL,
    mirror_x: bool | None = None,
    mirror_axes: tuple[str, ...] | None = None,
) -> AthRunResult:
    if mirror_axes is None:
        mirror_axes = ath_mirror_axes_for_result(result) if mirror_x is None else (("x",) if mirror_x else ())

    cleaned_path = output_path or result.msh_path.with_name(
        f"{result.msh_path.stem}{DEFAULT_CLEAN_SUFFIX}{result.msh_path.suffix}"
    )
    clean_mesh_file(
        str(result.msh_path),
        str(cleaned_path),
        merge_tol=merge_tol,
        area_tol=area_tol,
        mirror_x=False,
        mirror_axes=mirror_axes,
        binary=False,
    )
    quality_warning = triangle_quality_warning(meshio.read(cleaned_path))
    physical_names = read_physical_names(cleaned_path)
    driven_tag = physical_names[DRIVEN_DIAPHRAGM_PHYSICAL_NAME]
    return replace(
        result,
        cleaned_msh_path=cleaned_path,
        driven_tag=driven_tag,
        radiators=detect_ath_radiators(cleaned_path),
        quality_warning=quality_warning if quality_warning.has_warnings else None,
    )


def clean_ath_reduced_mesh_output(
    result: AthRunResult,
    *,
    output_path: Path | None = None,
    merge_tol: float = MERGE_TOL,
    area_tol: float = AREA_TOL,
) -> AthRunResult:
    reduced_path = output_path or result.msh_path.with_name(
        f"{result.msh_path.stem}{DEFAULT_CLEAN_SUFFIX}_reduced{result.msh_path.suffix}"
    )
    clean_mesh_file(
        str(result.msh_path),
        str(reduced_path),
        merge_tol=merge_tol,
        area_tol=area_tol,
        mirror_x=False,
        mirror_axes=(),
        binary=False,
    )
    return replace(result, reduced_cleaned_msh_path=reduced_path)


def copy_existing_ath_output(*, source_dir: Path, run_root: Path, case_name: str) -> AthRunResult:
    """Convenience helper for loading known outputs into the app workspace."""
    target_dir = run_root / case_name
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    return discover_ath_output(run_root=run_root, case_name=case_name)
