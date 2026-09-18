"""Resolve scene package assignments and their common exported frequencies."""

from pathlib import Path
from typing import Any

import numpy as np

from blab.deploy.assets import DeployPackageData, DeploySolveCache, _load_deploy_package_data


def scene_packages(payload: dict[str, Any], cache: DeploySolveCache | None = None) -> dict[str, DeployPackageData]:
    paths = payload.get("packagePaths")
    if paths is None:
        paths = {"": payload.get("packagePath", "")}
    elif not isinstance(paths, dict) or not paths or "" in paths:
        raise ValueError("Deploy packagePaths must be a non-empty package ID to path mapping.")
    sources = payload.get("sources", [])
    ids = list(dict.fromkeys(str(s.get("packageId", "")) for s in sources)) if "packagePaths" in payload else [""]
    if not ids:
        raise ValueError("Deploy requires at least one source.")
    packages = {}
    for package_id in ids:
        if package_id not in paths:
            raise ValueError(f"Deploy source references missing package {package_id!r}.")
        path = Path(str(paths[package_id])).expanduser().resolve()
        if path.suffix.lower() != ".blabsp":
            raise ValueError(f"Deploy requires an existing .blabsp package: {path}")
        packages[package_id] = cache.load_package(path) if cache is not None else _load_deploy_package_data(path)
    if len(packages) == 1:
        return packages
    media = [p.manifest.get("medium", {}) for p in packages.values()]
    for medium in media[1:]:
        for key, default in (("density_kg_per_m3", 1.21), ("sound_speed_m_per_s", 343.0)):
            if not np.isclose(float(medium.get(key, default)), float(media[0].get(key, default)), rtol=1e-6):
                raise ValueError("Deploy packages must use the same exterior air density and sound speed.")
    return packages


def source_package(packages: dict[str, DeployPackageData], source: dict[str, Any]) -> DeployPackageData:
    return packages[""] if "" in packages else packages[str(source.get("packageId", ""))]


def common_frequencies(packages: dict[str, DeployPackageData], *, coupled: bool) -> list[float]:
    grids = []
    for package in packages.values():
        grids.append(package.frequencies)
        if coupled:
            model = package.coupled_model
            if not isinstance(model, dict) or model.get("representation") != "parity_petrov_galerkin_rom":
                raise ValueError(f"Package {package.path.name} requires a parity Petrov–Galerkin ROM for Level 3.")
            grids.append(np.asarray(model["arrays"]["frequencies_hz"]))
    frequencies = sorted({float(v) for v in grids[0] if np.isfinite(v) and v > 0})
    return [v for v in frequencies if all(np.any(np.abs(grid - v) <= max(1e-4, abs(v) * 1e-6)) for grid in grids)]
