"""Portable solve-job bundles for cloud execution."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from blab.cloud.protocol import array_from_payload, array_to_payload, config_from_payload, config_to_payload
from blab.config import SimulationConfig


BUNDLE_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
INPUT_DIR = "inputs"


def write_solve_bundle(
    bundle_path: str | Path,
    *,
    config: SimulationConfig,
    frequencies: np.ndarray,
) -> Path:
    """Write a self-contained cloud solve bundle.

    The bundle contains the simulation config, requested frequency order, and
    every mesh file referenced by the config. Mesh paths in the manifest are
    rewritten to relative bundle paths so the server never needs client-local
    filesystem paths.
    """
    output_path = Path(bundle_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_payload = config_to_payload(config)
    file_map = _build_file_map(config)
    rewritten_config = _rewrite_config_paths_for_bundle(config_payload, file_map)
    manifest = {
        "format": "boundary-lab.solve-bundle",
        "version": BUNDLE_FORMAT_VERSION,
        "config": rewritten_config,
        "frequencies": array_to_payload(np.asarray(frequencies, dtype=np.float32)),
    }

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
        for source_path, bundle_name in file_map.items():
            archive.write(source_path, bundle_name)

    return output_path


def load_solve_bundle(bundle_path: str | Path, workspace_dir: str | Path) -> tuple[SimulationConfig, np.ndarray]:
    """Extract a solve bundle and return a workspace-local config."""
    bundle = Path(bundle_path)
    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(bundle, "r") as archive:
        _validate_archive_members(archive)
        archive.extractall(workspace)

    manifest_path = workspace / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "boundary-lab.solve-bundle":
        raise ValueError("Unsupported solve bundle format.")
    if int(manifest.get("version", 0)) != BUNDLE_FORMAT_VERSION:
        raise ValueError(f"Unsupported solve bundle version: {manifest.get('version')}")

    config_payload = _rewrite_config_paths_for_workspace(manifest["config"], workspace)
    frequencies = array_from_payload(manifest["frequencies"])
    if frequencies is None:
        raise ValueError("Solve bundle is missing frequencies.")
    return config_from_payload(config_payload), np.asarray(frequencies, dtype=np.float32)


def load_solve_bundle_bytes(data: bytes, workspace_dir: str | Path) -> tuple[SimulationConfig, np.ndarray]:
    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    bundle_path = workspace / "solve_bundle.zip"
    bundle_path.write_bytes(data)
    return load_solve_bundle(bundle_path, workspace / "extracted")


def _build_file_map(config: SimulationConfig) -> dict[Path, str]:
    source_paths = [Path(config.mesh_file)]
    source_paths.extend(Path(mesh.file) for mesh in config.meshes)

    file_map: dict[Path, str] = {}
    used_names: set[str] = set()
    for index, source in enumerate(source_paths):
        resolved = source.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Mesh file does not exist: {source}")
        if resolved in file_map:
            continue
        safe_name = _safe_bundle_filename(index, resolved.name, used_names)
        file_map[resolved] = f"{INPUT_DIR}/{safe_name}"
    return file_map


def _rewrite_config_paths_for_bundle(config_payload: dict[str, Any], file_map: dict[Path, str]) -> dict[str, Any]:
    rewritten = json.loads(json.dumps(config_payload))
    rewritten["mesh_file"] = file_map[Path(rewritten["mesh_file"]).expanduser().resolve()]
    for mesh in rewritten.get("meshes", []):
        mesh["file"] = file_map[Path(mesh["file"]).expanduser().resolve()]
    return rewritten


def _rewrite_config_paths_for_workspace(config_payload: dict[str, Any], workspace: Path) -> dict[str, Any]:
    rewritten = json.loads(json.dumps(config_payload))
    rewritten["mesh_file"] = str(_bundle_member_to_workspace_path(workspace, rewritten["mesh_file"]))
    for mesh in rewritten.get("meshes", []):
        mesh["file"] = str(_bundle_member_to_workspace_path(workspace, mesh["file"]))
    return rewritten


def _bundle_member_to_workspace_path(workspace: Path, member: str) -> Path:
    path = (workspace / member).resolve()
    workspace_root = workspace.resolve()
    if path != workspace_root and workspace_root not in path.parents:
        raise ValueError(f"Bundle path escapes workspace: {member}")
    return path


def _safe_bundle_filename(index: int, filename: str, used_names: set[str]) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filename).stem).strip("._") or "mesh"
    safe_suffix = re.sub(r"[^A-Za-z0-9.]+", "", Path(filename).suffix) or ".msh"
    candidate = f"{index:03d}_{safe_stem}{safe_suffix}"
    while candidate in used_names:
        candidate = f"{index:03d}_{safe_stem}_{len(used_names)}{safe_suffix}"
    used_names.add(candidate)
    return candidate


def _validate_archive_members(archive: zipfile.ZipFile) -> None:
    for member in archive.infolist():
        path = Path(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe solve bundle member path: {member.filename}")
