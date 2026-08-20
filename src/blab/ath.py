"""Ath4 command-line integration helpers."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np

from blab.config import RadiatorConfig
from blab.mesh_clean import (
    AREA_TOL,
    MERGE_TOL,
    MeshQualityWarning,
    clean_mesh,
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
WINE_PLATFORMS = {"linux", "darwin"}
ATH_QUADRANTS_RE = re.compile(r"^\s*Mesh\.Quadrants\s*=\s*(\S+)\s*$", re.IGNORECASE)
ATH_GEO_MESH_RE = re.compile(r"^\s*Mesh\s+2\s*;\s*$", re.IGNORECASE)
ATH_GEO_SAVE_RE = re.compile(r'^\s*Save\s+"[^"]+"\s*;\s*$', re.IGNORECASE)
ATH_GMSH_LOCK = threading.RLock()


class AthCancelledError(RuntimeError):
    """Raised when an active Ath generation is cancelled by the user."""


@dataclass(frozen=True)
class AthDriveDefinition:
    """One drive record emitted in Ath blab-mode GEO metadata."""

    internal_id: int
    user_id: int
    name: str
    ref_elements: str
    weight_absolute: float
    weight_db: float
    driving_direction: int


def parse_ath_drive_definitions(geo_text: str) -> tuple[AthDriveDefinition, ...]:
    """Parse ``//#// DRV`` records from an Ath blab-mode GEO footer."""
    definitions = []
    referenced_elements: set[str] = set()
    for line_number, raw_line in enumerate(str(geo_text).splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped.startswith("//#//"):
            continue
        fields = stripped.removeprefix("//#//").split()
        if not fields or fields[0] != "DRV":
            continue
        if len(fields) != 8:
            raise ValueError(
                f"Malformed Ath DRV metadata on GEO line {line_number}: expected 8 fields, found {len(fields)}."
            )
        try:
            definition = AthDriveDefinition(
                internal_id=int(fields[1]),
                user_id=int(fields[2]),
                name=fields[3],
                ref_elements=fields[4],
                weight_absolute=float(fields[5]),
                weight_db=float(fields[6]),
                driving_direction=int(fields[7]),
            )
        except ValueError as exc:
            raise ValueError(f"Invalid Ath DRV metadata on GEO line {line_number}: {raw_line}") from exc
        if not math.isfinite(definition.weight_absolute) or not math.isfinite(definition.weight_db):
            raise ValueError(f"Ath DRV metadata on GEO line {line_number} contains a non-finite weight.")
        if definition.ref_elements in referenced_elements:
            raise ValueError(f"Ath DRV metadata references {definition.ref_elements!r} more than once.")
        referenced_elements.add(definition.ref_elements)
        definitions.append(definition)
    return tuple(definitions)


def ath_mirror_axes_from_config_text(config_text: str) -> tuple[str, ...]:
    """Return the symmetry planes implied by an Ath ``Mesh.Quadrants`` assignment."""
    quadrant_values = []
    for raw_line in str(config_text).splitlines():
        active_line = raw_line.split(";", maxsplit=1)[0].strip()
        if not active_line:
            continue
        match = ATH_QUADRANTS_RE.match(active_line)
        if match is not None:
            quadrant_values.append(match.group(1))

    if len(quadrant_values) > 1:
        raise ValueError(
            "Ath source contains multiple active Mesh.Quadrants assignments. "
            "Keep exactly one assignment, or remove all of them for a fully expanded mesh."
        )

    value = quadrant_values[0] if quadrant_values else "1234"
    if value == "1":
        return ("x", "y")
    if value == "14":
        return ("x",)
    if value == "1234":
        return ()
    if value == "12":
        raise ValueError(
            "Mesh.Quadrants = 12 requests Y-only symmetry, which Boundary Lab does not support. "
            "Use 1 (XY), 14 (X), or 1234/remove the setting (no symmetry)."
        )
    raise ValueError(
        f"Invalid Mesh.Quadrants value {value!r}. Use 1 (XY), 14 (X), or 1234/remove the setting "
        "(no symmetry). Mesh.Quadrants = 12 is valid in Ath but Y-only symmetry is not supported "
        "by Boundary Lab."
    )


def _ath_process_command(ath_exe: Path, config_path: Path) -> list[str]:
    if sys.platform not in WINE_PLATFORMS:
        return [str(ath_exe), str(config_path), "-b"]

    wine_exe = shutil.which("wine")
    if wine_exe is None:
        raise RuntimeError(
            "Ath.exe execution on Linux/macOS requires Wine, but no 'wine' executable was found on PATH."
        )
    return [wine_exe, str(ath_exe), str(config_path), "-b"]


@dataclass(frozen=True)
class AthRunResult:
    output_dir: Path
    msh_path: Path
    config_path: Path
    driven_tag: int
    radiators: tuple[RadiatorConfig, ...]
    drive_definitions: tuple[AthDriveDefinition, ...] = ()
    geo_path: Path | None = None
    log_path: Path | None = None
    mirror_axes: tuple[str, ...] = ()
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
        status_callback: Callable[[str], None] | None = None,
    ) -> AthRunResult:
        ath_exe = ath_exe.resolve()
        if not ath_exe.exists():
            raise FileNotFoundError(f"Ath executable not found: {ath_exe}")

        mirror_axes = ath_mirror_axes_from_config_text(config_text)
        run_root.mkdir(parents=True, exist_ok=True)
        run_root = run_root.resolve()
        output_dir = run_root / case_name
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / f"{case_name}.cfg"
        geo_path = output_dir / f"{case_name}.geo"
        log_path = output_dir / "ath.log"
        config_path.write_text(config_text, encoding="utf-8")

        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        self._cancel_requested = False
        self._process = subprocess.Popen(
            _ath_process_command(ath_exe, config_path),
            cwd=output_dir,
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

        # Preserve Ath diagnostics even when generation fails.  Some geometry
        # errors are only explained on stderr in blab mode.
        log_path.write_text(stderr, encoding="utf-8", newline="")
        if stdout:
            geo_path.write_text(stdout, encoding="utf-8", newline="")

        returncode = 0 if process is None else process.returncode
        if returncode != 0:
            message = stderr.strip() or stdout.strip()
            raise RuntimeError(f"Ath failed with exit code {returncode}: {message}")

        if not stdout.strip():
            raise RuntimeError("Ath blab mode produced no Gmsh GEO output.")

        if status_callback is not None:
            status_callback("Meshing Ath geometry...")
        return self._run_gmsh_worker(
            output_dir=output_dir,
            case_name=case_name,
            config_path=config_path,
            geo_path=geo_path,
            log_path=log_path,
            mirror_axes=mirror_axes,
            timeout_s=timeout_s,
        )

    def _run_gmsh_worker(
        self,
        *,
        output_dir: Path,
        case_name: str,
        config_path: Path,
        geo_path: Path,
        log_path: Path,
        mirror_axes: tuple[str, ...],
        timeout_s: float | None,
    ) -> AthRunResult:
        if self._cancel_requested:
            raise AthCancelledError("Geometry generation cancelled")
        result_path = output_dir / ".ath_gmsh_result.json"
        result_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            "-m",
            "blab.ath_gmsh_worker",
            "--geo",
            str(geo_path),
            "--config",
            str(config_path),
            "--log",
            str(log_path),
            "--output-dir",
            str(output_dir),
            "--case-name",
            case_name,
            "--mirror-axes",
            "".join(mirror_axes),
            "--result",
            str(result_path),
        ]
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        self._process = subprocess.Popen(
            command,
            cwd=output_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags,
        )
        if self._cancel_requested:
            self.stop()
        try:
            stdout, stderr = self._process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.stop()
            raise
        finally:
            process = self._process
            self._process = None

        if self._cancel_requested:
            raise AthCancelledError("Geometry generation cancelled")
        returncode = 0 if process is None else process.returncode
        if returncode != 0:
            message = stderr.strip() or stdout.strip()
            raise RuntimeError(f"Gmsh worker failed with exit code {returncode}: {message}")
        if not result_path.exists():
            raise RuntimeError("Gmsh worker completed without returning an Ath mesh result.")
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            return ath_run_result_from_payload(payload)
        finally:
            result_path.unlink(missing_ok=True)

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
    status_callback: Callable[[str], None] | None = None,
) -> AthRunResult:
    return AthProcessRunner().run(
        ath_exe=ath_exe,
        config_text=config_text,
        run_root=run_root,
        case_name=case_name,
        timeout_s=timeout_s,
        status_callback=status_callback,
    )


def _strip_ath_geo_batch_commands(geo_text: str) -> str:
    """Remove Ath's batch mesh/save footer so Gmsh remains under application control."""
    kept_lines = []
    for raw_line in str(geo_text).splitlines():
        stripped = raw_line.strip()
        if ATH_GEO_MESH_RE.match(stripped) or ATH_GEO_SAVE_RE.match(stripped):
            continue
        kept_lines.append(raw_line)
    return "\n".join(kept_lines) + "\n"


def _meshio_from_current_gmsh_model(gmsh_module) -> meshio.Mesh:
    physical_groups = gmsh_module.model.getPhysicalGroups(2)
    if not physical_groups:
        raise RuntimeError("Ath GEO contains no named physical surface groups.")

    field_data: dict[str, np.ndarray] = {}
    physical_tags_by_entity: dict[int, list[int]] = {}
    for dimension, physical_tag in physical_groups:
        name = gmsh_module.model.getPhysicalName(dimension, physical_tag).strip()
        if not name:
            raise RuntimeError(f"Ath GEO physical surface group {physical_tag} has no name.")
        field_data[name] = np.array([int(physical_tag), 2], dtype=np.int32)
        for entity_tag in gmsh_module.model.getEntitiesForPhysicalGroup(dimension, physical_tag):
            physical_tags_by_entity.setdefault(int(entity_tag), []).append(int(physical_tag))

    if DRIVEN_DIAPHRAGM_PHYSICAL_NAME not in field_data:
        raise RuntimeError(
            f"Ath GEO does not define the required {DRIVEN_DIAPHRAGM_PHYSICAL_NAME!r} physical surface group."
        )

    node_tags, coordinates, _parameters = gmsh_module.model.mesh.getNodes()
    coordinate_rows = np.asarray(coordinates, dtype=np.float64).reshape((-1, 3))
    coordinates_by_tag = {int(node_tag): coordinate_rows[index] for index, node_tag in enumerate(np.asarray(node_tags))}

    triangle_node_tags: list[np.ndarray] = []
    triangle_physical_tags: list[np.ndarray] = []
    triangle_geometrical_tags: list[np.ndarray] = []
    for _dimension, entity_tag in gmsh_module.model.getEntities(2):
        element_types, element_tags, element_nodes = gmsh_module.model.mesh.getElements(2, entity_tag)
        triangle_count = 0
        for element_type, tags, nodes in zip(element_types, element_tags, element_nodes, strict=True):
            element_name, dimension, order, node_count, _local_coords, _primary_nodes = (
                gmsh_module.model.mesh.getElementProperties(int(element_type))
            )
            if int(element_type) != 2 or int(dimension) != 2 or int(order) != 1 or int(node_count) != 3:
                raise RuntimeError(
                    "Ath GEO produced unsupported surface elements "
                    f"{element_name!r} (Gmsh type {int(element_type)}); first-order triangles are required."
                )
            block = np.asarray(nodes, dtype=np.int64).reshape((-1, 3))
            triangle_node_tags.append(block)
            triangle_count += len(tags)

        if triangle_count == 0:
            continue
        entity_physical_tags = physical_tags_by_entity.get(int(entity_tag), [])
        if len(entity_physical_tags) != 1:
            detail = "no physical surface group" if not entity_physical_tags else "multiple physical surface groups"
            raise RuntimeError(f"Ath GEO surface {int(entity_tag)} belongs to {detail}.")
        triangle_physical_tags.append(np.full(triangle_count, entity_physical_tags[0], dtype=np.int32))
        triangle_geometrical_tags.append(np.full(triangle_count, int(entity_tag), dtype=np.int32))

    if not triangle_node_tags:
        raise RuntimeError("Gmsh generated no surface triangles from the Ath GEO output.")

    gmsh_triangles = np.vstack(triangle_node_tags)
    used_node_tags = np.unique(gmsh_triangles)
    missing_node_tags = [int(tag) for tag in used_node_tags if int(tag) not in coordinates_by_tag]
    if missing_node_tags:
        raise RuntimeError(f"Gmsh omitted coordinates for Ath mesh node {missing_node_tags[0]}.")
    node_index_by_tag = {int(tag): index for index, tag in enumerate(used_node_tags)}
    points = np.asarray([coordinates_by_tag[int(tag)] for tag in used_node_tags], dtype=np.float64)
    triangles = np.asarray(
        [[node_index_by_tag[int(tag)] for tag in triangle] for triangle in gmsh_triangles],
        dtype=np.int64,
    )
    return meshio.Mesh(
        points=points,
        cells=[("triangle", triangles)],
        cell_data={
            "gmsh:physical": [np.concatenate(triangle_physical_tags)],
            "gmsh:geometrical": [np.concatenate(triangle_geometrical_tags)],
        },
        field_data=field_data,
    )


def mesh_ath_geo_text(geo_text: str, *, output_dir: Path, case_name: str) -> meshio.Mesh:
    """Mesh Ath's stdout GEO with the in-process Gmsh API and return it in memory."""
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError("Ath geometry generation requires the Python gmsh package.") from exc

    parser_path = output_dir / f".{case_name}_gmsh_input.geo"
    parser_path.write_text(_strip_ath_geo_batch_commands(geo_text), encoding="utf-8", newline="")
    with ATH_GMSH_LOCK:
        was_initialized = bool(gmsh.isInitialized())
        previous_model = gmsh.model.getCurrent() if was_initialized else ""
        model_name = ""
        if not was_initialized:
            gmsh.initialize(readConfigFiles=False, interruptible=False)
        previous_terminal = gmsh.option.getNumber("General.Terminal")
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(str(parser_path))
            model_name = gmsh.model.getCurrent()
            gmsh.model.mesh.generate(2)
            return _meshio_from_current_gmsh_model(gmsh)
        except Exception as exc:
            raise RuntimeError(f"Gmsh could not mesh Ath's GEO output: {exc}") from exc
        finally:
            if model_name and gmsh.model.getCurrent() == model_name:
                gmsh.model.remove()
            gmsh.option.setNumber("General.Terminal", previous_terminal)
            if was_initialized:
                if previous_model and previous_model in gmsh.model.list():
                    gmsh.model.setCurrent(previous_model)
            else:
                gmsh.finalize()
            parser_path.unlink(missing_ok=True)


def _radiators_from_physical_names(
    physical_names: dict[str, int],
    drive_definitions: tuple[AthDriveDefinition, ...] = (),
) -> tuple[RadiatorConfig, ...]:
    if drive_definitions:
        missing = [
            definition.ref_elements for definition in drive_definitions if definition.ref_elements not in physical_names
        ]
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise RuntimeError(f"Ath DRV metadata references missing physical surfaces: {names}.")
        return tuple(
            RadiatorConfig(
                name=definition.ref_elements,
                tag=physical_names[definition.ref_elements],
                drive_group=f"ath:{definition.internal_id}",
                drive_group_name=definition.name,
                velocity_offset_db=definition.weight_db,
            )
            for definition in drive_definitions
        )

    if set(COMPLEX_RADIATOR_DRIVES_DB).issubset(physical_names):
        return tuple(
            RadiatorConfig(
                name=COMPLEX_RADIATOR_NAMES[physical_name],
                tag=physical_names[physical_name],
                velocity_offset_db=level_db,
            )
            for physical_name, level_db in COMPLEX_RADIATOR_DRIVES_DB.items()
        )

    if DRIVEN_DIAPHRAGM_PHYSICAL_NAME in physical_names:
        return (
            RadiatorConfig(
                name="throat",
                tag=physical_names[DRIVEN_DIAPHRAGM_PHYSICAL_NAME],
            ),
        )
    return ()


def build_ath_mesh_artifacts(
    raw_mesh: meshio.Mesh,
    *,
    output_dir: Path,
    case_name: str,
    config_path: Path,
    geo_path: Path | None = None,
    log_path: Path | None = None,
    drive_definitions: tuple[AthDriveDefinition, ...] = (),
    mirror_axes: tuple[str, ...] = (),
    merge_tol: float = MERGE_TOL,
    area_tol: float = AREA_TOL,
) -> AthRunResult:
    """Clean an in-memory Ath mesh and materialize only solve-ready artifacts."""
    reduced_mesh, _changes, _before, _after = clean_mesh(
        raw_mesh,
        merge_tol=merge_tol,
        area_tol=area_tol,
        mirror_x=False,
        mirror_axes=(),
    )
    if mirror_axes:
        expanded_mesh, _changes, _before, _after = clean_mesh(
            raw_mesh,
            merge_tol=merge_tol,
            area_tol=area_tol,
            mirror_x=False,
            mirror_axes=mirror_axes,
        )
    else:
        expanded_mesh = reduced_mesh

    expanded_path = output_dir / f"{case_name}.msh"
    reduced_path = output_dir / f"{case_name}_reduced.msh" if mirror_axes else expanded_path
    meshio.write(expanded_path, expanded_mesh, file_format="gmsh22", binary=False)
    if reduced_path != expanded_path:
        meshio.write(reduced_path, reduced_mesh, file_format="gmsh22", binary=False)

    physical_names = {
        name: int(value[0]) for name, value in raw_mesh.field_data.items() if len(value) >= 2 and int(value[1]) == 2
    }
    try:
        driven_tag = physical_names[DRIVEN_DIAPHRAGM_PHYSICAL_NAME]
    except KeyError as exc:
        raise RuntimeError(
            f"Ath mesh does not contain the required {DRIVEN_DIAPHRAGM_PHYSICAL_NAME!r} surface."
        ) from exc
    quality_warning = triangle_quality_warning(expanded_mesh)
    return AthRunResult(
        output_dir=output_dir,
        msh_path=reduced_path,
        config_path=config_path,
        driven_tag=driven_tag,
        radiators=_radiators_from_physical_names(physical_names, drive_definitions),
        drive_definitions=drive_definitions,
        geo_path=geo_path,
        log_path=log_path,
        mirror_axes=mirror_axes,
        cleaned_msh_path=expanded_path,
        reduced_cleaned_msh_path=reduced_path,
        quality_warning=quality_warning if quality_warning.has_warnings else None,
    )


def ath_run_result_to_payload(result: AthRunResult) -> dict:
    warning = result.quality_warning
    return {
        "output_dir": str(result.output_dir),
        "msh_path": str(result.msh_path),
        "config_path": str(result.config_path),
        "driven_tag": int(result.driven_tag),
        "radiators": [
            {
                "name": radiator.name,
                "tag": int(radiator.tag),
                "drive_group": radiator.drive_group,
                "drive_group_name": radiator.drive_group_name,
                "velocity_offset_db": float(radiator.velocity_offset_db),
                "level_db": float(radiator.level_db),
            }
            for radiator in result.radiators
        ],
        "drive_definitions": [
            {
                "internal_id": definition.internal_id,
                "user_id": definition.user_id,
                "name": definition.name,
                "ref_elements": definition.ref_elements,
                "weight_absolute": definition.weight_absolute,
                "weight_db": definition.weight_db,
                "driving_direction": definition.driving_direction,
            }
            for definition in result.drive_definitions
        ],
        "geo_path": None if result.geo_path is None else str(result.geo_path),
        "log_path": None if result.log_path is None else str(result.log_path),
        "mirror_axes": list(result.mirror_axes),
        "cleaned_msh_path": None if result.cleaned_msh_path is None else str(result.cleaned_msh_path),
        "reduced_cleaned_msh_path": (
            None if result.reduced_cleaned_msh_path is None else str(result.reduced_cleaned_msh_path)
        ),
        "quality_warning": (
            None
            if warning is None
            else {
                "sliver_triangles": int(warning.sliver_triangles),
                "float32_singular_triangles": int(warning.float32_singular_triangles),
                "worst_triangle_index": int(warning.worst_triangle_index),
                "worst_altitude_edge_ratio": float(warning.worst_altitude_edge_ratio),
                "worst_area": float(warning.worst_area),
                "worst_longest_edge": float(warning.worst_longest_edge),
            }
        ),
    }


def ath_run_result_from_payload(payload: dict) -> AthRunResult:
    warning_payload = payload.get("quality_warning")
    warning = None if warning_payload is None else MeshQualityWarning(**warning_payload)
    return AthRunResult(
        output_dir=Path(payload["output_dir"]),
        msh_path=Path(payload["msh_path"]),
        config_path=Path(payload["config_path"]),
        driven_tag=int(payload["driven_tag"]),
        radiators=tuple(
            RadiatorConfig(
                name=str(radiator["name"]),
                tag=int(radiator["tag"]),
                drive_group=(None if radiator.get("drive_group") is None else str(radiator["drive_group"])),
                drive_group_name=(
                    None if radiator.get("drive_group_name") is None else str(radiator["drive_group_name"])
                ),
                velocity_offset_db=float(radiator.get("velocity_offset_db", 0.0)),
                level_db=float(radiator.get("level_db", 0.0)),
            )
            for radiator in payload.get("radiators", [])
        ),
        drive_definitions=tuple(
            AthDriveDefinition(
                internal_id=int(definition["internal_id"]),
                user_id=int(definition["user_id"]),
                name=str(definition["name"]),
                ref_elements=str(definition["ref_elements"]),
                weight_absolute=float(definition["weight_absolute"]),
                weight_db=float(definition["weight_db"]),
                driving_direction=int(definition["driving_direction"]),
            )
            for definition in payload.get("drive_definitions", [])
        ),
        geo_path=Path(payload["geo_path"]) if payload.get("geo_path") else None,
        log_path=Path(payload["log_path"]) if payload.get("log_path") else None,
        mirror_axes=tuple(str(axis) for axis in payload.get("mirror_axes", [])),
        cleaned_msh_path=(Path(payload["cleaned_msh_path"]) if payload.get("cleaned_msh_path") else None),
        reduced_cleaned_msh_path=(
            Path(payload["reduced_cleaned_msh_path"]) if payload.get("reduced_cleaned_msh_path") else None
        ),
        quality_warning=warning,
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


def detect_ath_radiators(
    msh_path: Path,
    drive_definitions: tuple[AthDriveDefinition, ...] = (),
) -> tuple[RadiatorConfig, ...]:
    return _radiators_from_physical_names(read_physical_names(msh_path), drive_definitions)
