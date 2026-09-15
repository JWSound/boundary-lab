"""Speaker package loading, rigid mesh validation, and reusable Deploy asset caches."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import meshio
import numpy as np

from blab.phasor import LEGACY_PHASOR_CONVENTION, SOLVER_PHASOR_CONVENTION, convert_phasor
from blab.speaker_package import validate_speaker_package


@dataclass(frozen=True)
class DeployPackageData:
    path: Path
    fingerprint: tuple[str, int, int]
    manifest: dict[str, Any]
    frequencies: np.ndarray
    triangles: np.ndarray
    points: np.ndarray
    pressure: np.ndarray
    normal: np.ndarray
    geometry_bytes: bytes
    coupled_model: dict[str, Any] | None = None


@dataclass(frozen=True)
class DeployRigidMeshData:
    path: Path
    fingerprint: tuple[str, int, int]
    points: np.ndarray
    triangles: np.ndarray


@dataclass(frozen=True)
class DeployRomSweepStage:
    binary_path: Path
    frequency_descriptors: tuple[dict[str, dict[str, object]], ...]
    binary_bytes: int


@dataclass
class DeploySolveCache:
    packages: dict[tuple[str, int, int], DeployPackageData] = field(default_factory=dict)
    rigid_meshes: dict[tuple[str, int, int], DeployRigidMeshData] = field(default_factory=dict)
    ground_image_pairs: dict[tuple[Any, ...], list[Any]] = field(default_factory=dict)
    sweep_geometries: dict[str, tuple[dict[str, Any], str]] = field(default_factory=dict)
    rom_sweep_stages: dict[tuple[Any, ...], DeployRomSweepStage] = field(default_factory=dict)
    _rom_sweep_temp: tempfile.TemporaryDirectory = field(
        default_factory=lambda: tempfile.TemporaryDirectory(prefix="blab-deploy-rom-cache-"),
        init=False,
        repr=False,
    )

    def _reset_rom_sweep_stages(self) -> None:
        self.rom_sweep_stages.clear()
        self._rom_sweep_temp.cleanup()
        self._rom_sweep_temp = tempfile.TemporaryDirectory(prefix="blab-deploy-rom-cache-")

    def close(self) -> None:
        self.rom_sweep_stages.clear()
        self._rom_sweep_temp.cleanup()

    def stage_rom_sweep_arrays(
        self,
        package: DeployPackageData,
        frequency_pairs: list[tuple[float, int]],
        array_names: tuple[str, ...],
    ) -> tuple[DeployRomSweepStage, bool]:
        arrays = package.coupled_model.get("arrays") if isinstance(package.coupled_model, dict) else None
        if not isinstance(arrays, dict):
            raise ValueError("Deploy parity-ROM package did not load its reduced arrays.")
        cache_key = (
            package.fingerprint,
            tuple(index for _frequency, index in frequency_pairs),
            array_names,
        )
        cached = self.rom_sweep_stages.get(cache_key)
        if cached is not None and cached.binary_path.is_file():
            return cached, True

        binary_values: dict[str, np.ndarray] = {}
        descriptor_names: list[dict[str, str]] = []
        for sweep_index, (_frequency_hz, array_index) in enumerate(frequency_pairs):
            names: dict[str, str] = {}
            for name in array_names:
                binary_name = f"{name}_{sweep_index}"
                binary_values[binary_name] = np.asarray(arrays[name][array_index], dtype=np.complex64)
                names[name] = binary_name
            descriptor_names.append(names)

        key_text = json.dumps(cache_key, sort_keys=True, separators=(",", ":"), default=str)
        binary_path = Path(self._rom_sweep_temp.name) / f"{hashlib.sha256(key_text.encode('utf-8')).hexdigest()}.bin"
        all_descriptors = _write_deploy_binary_arrays(binary_path, binary_values)
        stage = DeployRomSweepStage(
            binary_path=binary_path,
            frequency_descriptors=tuple(
                {name: all_descriptors[binary_name] for name, binary_name in names.items()}
                for names in descriptor_names
            ),
            binary_bytes=binary_path.stat().st_size,
        )
        self.rom_sweep_stages[cache_key] = stage
        return stage, False

    def load_package(self, package_path: Path) -> DeployPackageData:
        stat = package_path.stat()
        fingerprint = (str(package_path), int(stat.st_mtime_ns), int(stat.st_size))
        cached = self.packages.get(fingerprint)
        if cached is not None:
            return cached
        package = _load_deploy_package_data(package_path, fingerprint)
        self.packages.clear()
        self.packages[fingerprint] = package
        self.ground_image_pairs.clear()
        self.sweep_geometries.clear()
        self._reset_rom_sweep_stages()
        return package

    def load_rigid_mesh(self, mesh_path: Path) -> DeployRigidMeshData:
        stat = mesh_path.stat()
        fingerprint = (str(mesh_path), int(stat.st_mtime_ns), int(stat.st_size))
        cached = self.rigid_meshes.get(fingerprint)
        if cached is not None:
            return cached
        mesh = _load_rigid_mesh_data(mesh_path, fingerprint)
        self.rigid_meshes[fingerprint] = mesh
        return mesh


def _load_deploy_package_data(
    package_path: Path,
    fingerprint: tuple[str, int, int] | None = None,
) -> DeployPackageData:
    stat = package_path.stat()
    package_fingerprint = fingerprint or (str(package_path), int(stat.st_mtime_ns), int(stat.st_size))
    manifest = validate_speaker_package(package_path)
    source_convention = manifest.get("phasor_convention", LEGACY_PHASOR_CONVENTION)
    fixed_file = manifest.get("files", {}).get("fixed_sources", {})
    fixed_path = str(fixed_file.get("path", ""))
    geometry_path = str(fixed_file.get("geometry_mesh", ""))
    if not fixed_path or not geometry_path:
        raise ValueError("Speaker package does not declare fixed-source data and geometry.")
    with zipfile.ZipFile(package_path, "r") as archive:
        try:
            fixed_bytes = archive.read(fixed_path)
            geometry_bytes = archive.read(geometry_path)
        except KeyError as exc:
            raise ValueError(f"Speaker package is missing {exc.args[0]!r}.") from exc
        coupled_model = _read_coupled_descriptor(archive, manifest)
    with np.load(io.BytesIO(fixed_bytes), allow_pickle=False) as fixed:
        triangles = np.asarray(fixed["triangles"], dtype=np.int64)
        points = np.asarray(fixed["points_m"], dtype=np.float64)
        pressure = np.asarray(fixed["pressure_pa"])
        normal = np.asarray(fixed["normal_derivative_pa_per_m"])
    pressure = convert_phasor(pressure, source_convention)
    normal = convert_phasor(normal, source_convention)
    if coupled_model is not None and "arrays" in coupled_model:
        coupled_model["arrays"] = {
            key: convert_phasor(value, source_convention) if np.iscomplexobj(value) else value
            for key, value in coupled_model["arrays"].items()
        }
    manifest = dict(manifest, phasor_convention=SOLVER_PHASOR_CONVENTION, source_phasor_convention=source_convention)
    frequencies = np.asarray(manifest.get("frequencies_hz", ()), dtype=np.float64)
    return DeployPackageData(
        path=package_path,
        fingerprint=package_fingerprint,
        manifest=manifest,
        frequencies=frequencies,
        triangles=triangles,
        points=points,
        pressure=pressure,
        normal=normal,
        geometry_bytes=geometry_bytes,
        coupled_model=coupled_model,
    )


def _read_coupled_descriptor(
    archive: zipfile.ZipFile,
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    declaration = manifest.get("files", {}).get("coupled_model", {})
    representation = declaration.get("representation")
    if representation == "parity_petrov_galerkin_rom":
        model_path = str(declaration.get("path", ""))
        if not model_path:
            raise ValueError("Parity-ROM Level-3 package does not declare its model path.")
        try:
            payload = archive.read(model_path)
        except KeyError as exc:
            raise ValueError(f"Speaker package is missing {model_path!r}.") from exc
        with np.load(io.BytesIO(payload), allow_pickle=False) as model:
            arrays = {name: np.asarray(model[name]) for name in model.files}
        required = {
            "frequencies_hz",
            "k",
            "c",
            "d",
            "b",
            "e",
            "velocity",
            "current",
            "velocity_drive",
            "current_drive",
        }
        missing = sorted(required - arrays.keys())
        if missing:
            raise ValueError(f"Parity-ROM Level-3 model is missing arrays: {', '.join(missing)}.")
        return {**copy.deepcopy(declaration), "arrays": arrays}
    if representation != "exact_frequency_parametric_fem":
        return None
    descriptor_path = str(declaration.get("path", ""))
    if not descriptor_path:
        raise ValueError("Exact Level-3 package does not declare its system descriptor path.")
    try:
        descriptor = json.loads(archive.read(descriptor_path))
    except KeyError as exc:
        raise ValueError(f"Speaker package is missing {descriptor_path!r}.") from exc
    if descriptor.get("representation") != "exact_frequency_parametric_fem":
        raise ValueError("Exact Level-3 descriptor has an unsupported representation.")
    mesh_members = descriptor.get("mesh_members")
    if not isinstance(mesh_members, dict) or not mesh_members:
        raise ValueError("Exact Level-3 descriptor does not contain mesh members.")
    archive_members = set(archive.namelist())
    for member in mesh_members.values():
        path = Path(str(member))
        if path.is_absolute() or ".." in path.parts or str(member) not in archive_members:
            raise ValueError(f"Exact Level-3 descriptor references invalid mesh member {member!r}.")
    return descriptor


def _load_rigid_mesh_data(
    mesh_path: Path,
    fingerprint: tuple[str, int, int] | None = None,
) -> DeployRigidMeshData:
    try:
        mesh = meshio.read(mesh_path)
    except Exception as exc:
        raise ValueError(f"Rigid mesh {mesh_path.name!r} could not be read: {exc}") from exc
    triangle_blocks = [np.asarray(block.data, dtype=np.int64) for block in mesh.cells if block.type == "triangle"]
    if not triangle_blocks:
        raise ValueError(f"Rigid mesh {mesh_path.name!r} must contain linear triangular surface elements.")
    triangles = np.concatenate(triangle_blocks, axis=0)
    source_points = np.asarray(mesh.points, dtype=np.float64)
    if source_points.ndim != 2 or source_points.shape[1] < 3 or not np.all(np.isfinite(source_points[:, :3])):
        raise ValueError(f"Rigid mesh {mesh_path.name!r} contains invalid vertices.")
    used = np.unique(triangles.reshape(-1))
    if used.size < 4 or np.any(used < 0) or np.any(used >= source_points.shape[0]):
        raise ValueError(f"Rigid mesh {mesh_path.name!r} contains invalid triangle connectivity.")
    remap = np.full(source_points.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    triangles = remap[triangles]
    # Raw Gmsh assets use the conventional Z-up frame. Deploy uses Y-up.
    raw = source_points[used, :3]
    points = np.column_stack((raw[:, 0], raw[:, 2], raw[:, 1]))
    face_points = points[triangles]
    doubled_areas = np.linalg.norm(
        np.cross(face_points[:, 1] - face_points[:, 0], face_points[:, 2] - face_points[:, 0]),
        axis=1,
    )
    if np.any(doubled_areas <= 1e-12):
        raise ValueError(f"Rigid mesh {mesh_path.name!r} contains degenerate triangles.")
    edge_counts: dict[tuple[int, int], int] = {}
    directed_edges: set[tuple[int, int]] = set()
    for face in triangles:
        for start, end in ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))):
            edge = (min(start, end), max(start, end))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            if (start, end) in directed_edges:
                raise ValueError(f"Rigid mesh {mesh_path.name!r} has inconsistent face orientation.")
            directed_edges.add((start, end))
    if any(count != 2 for count in edge_counts.values()):
        raise ValueError(f"Rigid mesh {mesh_path.name!r} must be a closed two-manifold surface.")
    if any((end, start) not in directed_edges for start, end in directed_edges):
        raise ValueError(f"Rigid mesh {mesh_path.name!r} has inconsistent face orientation.")
    signed_volume = float(
        np.sum(np.einsum("ij,ij->i", face_points[:, 0], np.cross(face_points[:, 1], face_points[:, 2]))) / 6.0
    )
    if abs(signed_volume) <= 1e-12:
        raise ValueError(f"Rigid mesh {mesh_path.name!r} has zero enclosed volume.")
    if signed_volume < 0.0:
        triangles = triangles[:, [0, 2, 1]]
    stat = mesh_path.stat()
    return DeployRigidMeshData(
        path=mesh_path,
        fingerprint=fingerprint or (str(mesh_path), int(stat.st_mtime_ns), int(stat.st_size)),
        points=points,
        triangles=triangles,
    )


def _write_deploy_binary_arrays(
    path: Path,
    arrays: dict[str, np.ndarray],
) -> dict[str, dict[str, object]]:
    descriptors: dict[str, dict[str, object]] = {}
    offset = 0
    with path.open("wb") as stream:
        for name, values in arrays.items():
            array = np.ascontiguousarray(values, dtype=np.complex64)
            payload = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes(order="C")
            stream.write(payload)
            descriptors[name] = {
                "file": str(path.resolve()),
                "offset": offset,
                "nbytes": len(payload),
                "dtype": "complex64",
                "shape": list(array.shape),
                "order": "C",
                "byte_order": "little",
            }
            offset += len(payload)
    return descriptors
