"""Qt-free preparation for Boundary Lab Deploy boundary and coupled solves."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import meshio
import numpy as np

from blab.deploy_geometry import (
    first_surface_pair_within,
    surface_face_pairs_within,
    transform_package_points,
)
from blab.speaker_package import validate_speaker_package

DEPLOY_SOLVE_SCHEMA = "boundary_lab_deploy_solve"
DEPLOY_SOLVE_SCHEMA_VERSION = 2
DEPLOY_COUPLED_SCHEMA = "boundary_lab_deploy_coupled"
DEPLOY_FIELD_SCHEMA = "boundary_lab_deploy_field"
DEPLOY_MICROPHONE_SWEEP_SCHEMA = "boundary_lab_deploy_microphone_sweep"
SOURCE_SURFACE_PADDING_M = 0.01
CLOSE_PAIR_DISTANCE_M = 0.05
CLOSE_PAIR_QUADRATURE_ORDER = 8
GROUND_TOLERANCE_M = 1e-6
GROUND_IMAGE_SINGULAR_TOLERANCE_M = 1e-8


@dataclass(frozen=True)
class DeploySourcePlacement:
    id: str
    position_x_m: float
    position_height_m: float
    position_z_m: float
    pitch_deg: float
    yaw_deg: float
    roll_deg: float
    level_db: float
    delay_ms: float
    polarity: int
    muted: bool

    @classmethod
    def from_payload(cls, raw: object) -> "DeploySourcePlacement":
        if not isinstance(raw, dict):
            raise ValueError("Deploy source must be an object.")
        polarity = int(raw.get("polarity", 1))
        if polarity not in (-1, 1):
            raise ValueError("Deploy source polarity must be -1 or +1.")
        values = cls(
            id=str(raw.get("id", "")).strip(),
            position_x_m=float(raw.get("positionX", 0.0)),
            position_height_m=float(raw.get("positionHeightM", 0.0)),
            position_z_m=float(raw.get("positionZ", 0.0)),
            pitch_deg=float(raw.get("pitchDeg", 0.0)),
            yaw_deg=float(raw.get("yawDeg", 0.0)),
            roll_deg=float(raw.get("rollDeg", 0.0)),
            level_db=float(raw.get("levelDb", 0.0)),
            delay_ms=float(raw.get("delayMs", 0.0)),
            polarity=polarity,
            muted=bool(raw.get("muted", False)),
        )
        if not values.id:
            raise ValueError("Deploy source id must not be empty.")
        if not all(
            math.isfinite(value)
            for value in (
                values.position_x_m,
                values.position_height_m,
                values.position_z_m,
                values.pitch_deg,
                values.yaw_deg,
                values.roll_deg,
                values.level_db,
                values.delay_ms,
            )
        ):
            raise ValueError("Deploy source values must be finite.")
        return values


@dataclass(frozen=True)
class DeployRigidPlacement:
    id: str
    mesh_path: Path
    scale_to_meters: float
    position_x_m: float
    position_height_m: float
    position_z_m: float
    pitch_deg: float
    yaw_deg: float
    roll_deg: float

    @classmethod
    def from_payload(cls, raw: object) -> "DeployRigidPlacement":
        if not isinstance(raw, dict):
            raise ValueError("Deploy rigid object must be an object.")
        mesh_path = Path(str(raw.get("meshPath", ""))).expanduser().resolve()
        value = cls(
            id=str(raw.get("id", "")).strip(),
            mesh_path=mesh_path,
            scale_to_meters=float(raw.get("scaleToMeters", 0.001)),
            position_x_m=float(raw.get("positionX", 0.0)),
            position_height_m=float(raw.get("positionHeightM", 0.0)),
            position_z_m=float(raw.get("positionZ", 0.0)),
            pitch_deg=float(raw.get("pitchDeg", 0.0)),
            yaw_deg=float(raw.get("yawDeg", 0.0)),
            roll_deg=float(raw.get("rollDeg", 0.0)),
        )
        if not value.id:
            raise ValueError("Deploy rigid object id must not be empty.")
        if value.mesh_path.suffix.lower() != ".msh" or not value.mesh_path.is_file():
            raise ValueError(f"Deploy rigid object {value.id!r} requires an existing .msh file.")
        if (
            not all(
                math.isfinite(item)
                for item in (
                    value.scale_to_meters,
                    value.position_x_m,
                    value.position_height_m,
                    value.position_z_m,
                    value.pitch_deg,
                    value.yaw_deg,
                    value.roll_deg,
                )
            )
            or value.scale_to_meters <= 0.0
        ):
            raise ValueError("Deploy rigid object values must be finite and its scale must be positive.")
        return value


@dataclass(frozen=True)
class DeployObservationPlane:
    width_m: float
    depth_m: float
    center_x_m: float
    near_m: float
    height_m: float
    pitch_deg: float
    yaw_deg: float
    roll_deg: float
    columns: int
    rows: int

    @classmethod
    def from_payload(cls, raw: object) -> "DeployObservationPlane":
        if not isinstance(raw, dict):
            raise ValueError("Deploy observation plane must be an object.")
        value = cls(
            width_m=float(raw.get("widthM", 0.0)),
            depth_m=float(raw.get("depthM", 0.0)),
            center_x_m=float(raw.get("centerXM", 0.0)),
            near_m=float(raw.get("nearM", 0.0)),
            height_m=float(raw.get("heightM", 0.0)),
            pitch_deg=float(raw.get("pitchDeg", 0.0)),
            yaw_deg=float(raw.get("yawDeg", 0.0)),
            roll_deg=float(raw.get("rollDeg", 0.0)),
            columns=int(raw.get("columns", 0)),
            rows=int(raw.get("rows", 0)),
        )
        if not all(
            math.isfinite(item)
            for item in (
                value.width_m,
                value.depth_m,
                value.center_x_m,
                value.near_m,
                value.height_m,
                value.pitch_deg,
                value.yaw_deg,
                value.roll_deg,
            )
        ):
            raise ValueError("Deploy observation plane values must be finite.")
        if value.width_m <= 0.0 or value.depth_m <= 0.0:
            raise ValueError("Deploy observation plane width and depth must be positive.")
        if value.columns < 2 or value.rows < 2:
            raise ValueError("Deploy observation plane must contain at least two rows and columns.")
        if value.columns * value.rows > 250_000:
            raise ValueError("Deploy observation plane exceeds the 250,000 point limit.")
        return value

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(-self.width_m / 2.0, self.width_m / 2.0, self.columns, dtype=np.float32)
        z = np.linspace(-self.depth_m / 2.0, self.depth_m / 2.0, self.rows, dtype=np.float32)
        local_x, local_z = np.meshgrid(x, z, indexing="xy")
        yaw = math.radians(self.yaw_deg)
        roll = math.radians(self.roll_deg)
        roll_cosine = math.cos(roll)
        roll_sine = math.sin(roll)
        rolled_x = roll_cosine * local_x
        rolled_y = roll_sine * local_x
        pitch = math.radians(self.pitch_deg)
        pitch_cosine = math.cos(pitch)
        pitch_sine = math.sin(pitch)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        center_z_m = self.near_m + self.depth_m / 2.0
        pitched_y = pitch_cosine * rolled_y - pitch_sine * local_z
        pitched_z = pitch_sine * rolled_y + pitch_cosine * local_z
        world_x = self.center_x_m + cosine * rolled_x + sine * pitched_z
        world_z = center_z_m - sine * rolled_x + cosine * pitched_z
        points = np.column_stack(
            (
                world_x.reshape(-1),
                self.height_m + pitched_y.reshape(-1),
                world_z.reshape(-1),
            )
        ).astype(np.float32, copy=False)
        sample_indices = np.flatnonzero(points[:, 1] >= -GROUND_TOLERANCE_M).astype(np.int64)
        return points[sample_indices], sample_indices

    def wire(self) -> dict[str, float | int]:
        return {
            "width_m": self.width_m,
            "depth_m": self.depth_m,
            "center_x_m": self.center_x_m,
            "near_m": self.near_m,
            "height_m": self.height_m,
            "pitch_deg": self.pitch_deg,
            "yaw_deg": self.yaw_deg,
            "roll_deg": self.roll_deg,
            "columns": self.columns,
            "rows": self.rows,
            "ground_tolerance_m": GROUND_TOLERANCE_M,
        }


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
class DeployBoundaryComponent:
    id: str
    kind: str
    fingerprint: tuple[str, int, int]
    points: np.ndarray
    triangles: np.ndarray
    face_offset: int
    vertex_offset: int
    q_neumann: np.ndarray
    reference_pressure: np.ndarray


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


def _logical_excitation_indices(
    manifest: dict[str, Any],
    excitation_count: int,
    selected_index: int = 0,
) -> tuple[int, ...]:
    """Return symmetry-expanded ports belonging to one logical package input."""

    if not 0 <= selected_index < excitation_count:
        raise ValueError("Selected speaker-package excitation index is out of range.")
    port_ids = manifest.get("excitation_port_ids")
    if not isinstance(port_ids, list) or len(port_ids) != excitation_count:
        return (selected_index,)
    physical_system = manifest.get("physical_system")
    if not isinstance(physical_system, dict):
        return (selected_index,)
    metadata = physical_system.get("metadata")
    if not isinstance(metadata, dict):
        return (selected_index,)
    expansion = metadata.get("speaker_export_symmetry_expansion")
    if not isinstance(expansion, dict):
        return (selected_index,)
    source_ids = expansion.get("excitation_port_source_ids")
    if not isinstance(source_ids, dict):
        return (selected_index,)
    selected_port_id = str(port_ids[selected_index])
    logical_source_id = source_ids.get(selected_port_id)
    if not isinstance(logical_source_id, str) or not logical_source_id:
        return (selected_index,)
    grouped = tuple(
        index for index, port_id in enumerate(port_ids) if source_ids.get(str(port_id)) == logical_source_id
    )
    return grouped or (selected_index,)


def _combined_excitation_trace(
    values: np.ndarray,
    frequency_index: int,
    excitation_indices: tuple[int, ...],
) -> np.ndarray:
    selected = np.asarray(values[frequency_index, excitation_indices, :])
    return selected[0] if len(excitation_indices) == 1 else np.sum(selected, axis=0)


def _load_deploy_package_data(
    package_path: Path,
    fingerprint: tuple[str, int, int] | None = None,
) -> DeployPackageData:
    stat = package_path.stat()
    package_fingerprint = fingerprint or (str(package_path), int(stat.st_mtime_ns), int(stat.st_size))
    manifest = validate_speaker_package(package_path)
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


def stage_exact_coupled_system(
    package: DeployPackageData,
    work_dir: str | Path,
) -> dict[str, Any]:
    """Extract an exact Level-3 system into a worker-owned temporary directory.

    The returned descriptor is a detached JSON value whose compiled mesh paths
    point at extracted local files. Numeric operators remain lazy: the Julia
    worker assembles/factors them only when a coupled frequency is requested.
    """

    if package.coupled_model is None:
        raise ValueError("Speaker package does not contain an exact Level-3 interior system.")
    descriptor = json.loads(json.dumps(package.coupled_model))
    compiled = descriptor.get("compiled_system")
    mesh_members = descriptor.get("mesh_members")
    if not isinstance(compiled, dict) or not isinstance(mesh_members, dict):
        raise ValueError("Exact Level-3 descriptor is incomplete.")
    target_dir = Path(work_dir).resolve() / "speaker-interior"
    target_dir.mkdir(parents=True, exist_ok=True)
    meshes = compiled.get("meshes", ())
    with zipfile.ZipFile(package.path, "r") as archive:
        for index, mesh in enumerate(meshes):
            mesh_id = str(mesh.get("id", ""))
            member = str(mesh_members.get(mesh_id, ""))
            if not member:
                raise ValueError(f"Exact Level-3 descriptor has no mesh member for {mesh_id!r}.")
            suffix = Path(member).suffix or ".msh"
            target = target_dir / f"{index:03d}{suffix}"
            target.write_bytes(archive.read(member))
            mesh["file"] = str(target)
    descriptor["mesh_path_kind"] = "local_file"
    return descriptor


def prepare_deploy_coupled_request(
    payload: object,
    work_dir: str | Path,
    *,
    cache: DeploySolveCache | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Build one full coupled system containing every Level-3 cabinet instance.

    The production coupled backend factors the combined system once and solves
    every requested package port as multiple right-hand sides. The output
    carries complex cabinet weights so Julia evaluates only the synthesized
    audience field rather than returning one field per port.
    """

    if not isinstance(payload, dict):
        raise ValueError("Deploy Level 3 request must be an object.")
    package_path = Path(str(payload.get("packagePath", ""))).expanduser().resolve()
    package = cache.load_package(package_path) if cache is not None else _load_deploy_package_data(package_path)
    if package.coupled_model is None:
        raise ValueError("Deploy Level 3 requires an exact coupled speaker package.")
    declaration = package.manifest.get("files", {}).get("coupled_model", {})
    if declaration.get("representation") != "exact_frequency_parametric_fem":
        raise ValueError("Deploy Level 3 requires the exact frequency-parametric representation.")

    raw_frequencies = payload.get("frequenciesHz")
    if raw_frequencies is None:
        frequencies = [float(payload.get("frequencyHz", 0.0))]
    elif isinstance(raw_frequencies, list) and raw_frequencies:
        frequencies = [float(value) for value in raw_frequencies]
    else:
        raise ValueError("Deploy Level 3 frequenciesHz must be a non-empty array.")
    if any(not math.isfinite(value) or value <= 0.0 for value in frequencies):
        raise ValueError("Deploy Level 3 frequencies must be finite and positive.")
    if len(set(frequencies)) != len(frequencies):
        raise ValueError("Deploy Level 3 frequencies must be unique.")
    requested_frequency = frequencies[0]
    band = package.coupled_model.get("frequency_band_hz", ())
    if isinstance(band, list) and len(band) == 2:
        lower, upper = map(float, band)
        tolerance = max(1e-4, max(abs(lower), abs(upper)) * 1e-6)
        outside = next((value for value in frequencies if value < lower - tolerance or value > upper + tolerance), None)
        if outside is not None:
            raise ValueError(
                f"Deploy Level 3 frequency {outside:g} Hz is outside the package band {lower:g}-{upper:g} Hz."
            )

    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Deploy Level 3 requires at least one source.")
    if len(raw_sources) > 8:
        raise ValueError("Deploy Level 3 currently supports at most 8 sources.")
    sources = [DeploySourcePlacement.from_payload(raw) for raw in raw_sources]
    if len({source.id for source in sources}) != len(sources):
        raise ValueError("Deploy Level 3 source ids must be unique.")

    raw_rigid_objects = payload.get("rigidObjects", [])
    if not isinstance(raw_rigid_objects, list):
        raise ValueError("Deploy Level 3 rigidObjects must be an array.")
    rigid_objects = [DeployRigidPlacement.from_payload(raw) for raw in raw_rigid_objects]
    raw_observation_points = payload.get("observationPointsM")
    observation = None
    if raw_observation_points is None:
        observation = DeployObservationPlane.from_payload(payload.get("observation"))
        observation_points, observation_sample_indices = observation.points()
        observation_shape = [observation.rows, observation.columns]
    else:
        observation_points = np.asarray(raw_observation_points, dtype=np.float32)
        if observation_points.ndim != 2 or observation_points.shape[1] != 3 or observation_points.shape[0] == 0:
            raise ValueError("Deploy Level 3 observationPointsM must contain one or more XYZ points.")
        if observation_points.shape[0] > 1_024 or not np.all(np.isfinite(observation_points)):
            raise ValueError("Deploy Level 3 observationPointsM must contain at most 1,024 finite points.")
        if np.any(observation_points[:, 1] < -GROUND_TOLERANCE_M):
            raise ValueError("Deploy Level 3 observation points cannot be below the ground plane.")
        observation_sample_indices = np.arange(observation_points.shape[0], dtype=np.int64)
        observation_shape = [1, int(observation_points.shape[0])]
    if observation_points.shape[0] == 0:
        raise ValueError("Deploy Level 3 has no audience samples on or above the ground plane.")

    work_path = Path(work_dir).resolve()
    work_path.mkdir(parents=True, exist_ok=True)
    staged = stage_exact_coupled_system(package, work_path)
    base_system = staged["compiled_system"]
    base_meshes = list(base_system.get("meshes", ()))
    base_regions = list(base_system.get("regions", ()))
    base_boundaries = list(base_system.get("boundaries", ()))
    base_interfaces = list(base_system.get("interfaces", ()))
    base_components = list(base_system.get("components", ()))
    base_ports = list(base_system.get("excitation_ports", ()))
    unbounded = [region for region in base_regions if region.get("kind") == "unbounded_air"]
    if len(unbounded) != 1:
        raise ValueError("Exact Level-3 package must contain one unbounded acoustic region.")
    base_unbounded_id = str(unbounded[0]["id"])
    combined_meshes: list[dict[str, Any]] = []
    combined_regions: list[dict[str, Any]] = []
    combined_boundaries: list[dict[str, Any]] = []
    combined_interfaces: list[dict[str, Any]] = []
    combined_components: list[dict[str, Any]] = []
    combined_ports: list[dict[str, Any]] = []
    unbounded_mesh_ids: list[str] = []
    excitation_ids: list[str] = []
    excitation_weights: list[dict[str, float]] = []
    excitation_weights_sweep: list[list[dict[str, float]]] = [[] for _ in frequencies]
    instance_dir = work_path / "coupled-instances"
    instance_dir.mkdir(parents=True, exist_ok=True)

    if status_callback is not None:
        status_callback(f"Staging {len(sources)} exact cabinet interiors")
    for source_index, source in enumerate(sources):
        prefix = f"deploy:{source_index}:{source.id}:"
        id_maps = {
            category: {str(item["id"]): prefix + str(item["id"]) for item in items}
            for category, items in (
                ("mesh", base_meshes),
                ("region", base_regions),
                ("boundary", base_boundaries),
                ("interface", base_interfaces),
                ("component", base_components),
                ("port", base_ports),
            )
        }
        source_mesh_dir = instance_dir / f"{source_index:02d}"
        source_mesh_dir.mkdir(parents=True, exist_ok=True)
        for mesh_index, base_mesh in enumerate(base_meshes):
            cloned = copy.deepcopy(base_mesh)
            mesh_id = str(base_mesh["id"])
            cloned["id"] = id_maps["mesh"][mesh_id]
            cloned["name"] = f"{source.id} / {base_mesh.get('name', mesh_id)}"
            source_file = Path(str(base_mesh["file"]))
            target = source_mesh_dir / f"{mesh_index:03d}-{source_file.name}"
            _write_transformed_coupled_mesh(source_file, target, base_mesh, source)
            cloned["file"] = str(target)
            cloned["scale_to_m"] = 1.0
            cloned["translation_m"] = [0.0, 0.0, 0.0]
            combined_meshes.append(cloned)
            if mesh_id in unbounded[0].get("mesh_ids", ()):
                unbounded_mesh_ids.append(str(cloned["id"]))

        for base_region in base_regions:
            if str(base_region["id"]) == base_unbounded_id:
                continue
            cloned = copy.deepcopy(base_region)
            cloned["id"] = id_maps["region"][str(base_region["id"])]
            cloned["name"] = f"{source.id} / {base_region.get('name', base_region['id'])}"
            cloned["mesh_ids"] = [id_maps["mesh"][str(value)] for value in base_region.get("mesh_ids", ())]
            for group in cloned.get("volume_groups", ()):
                group["mesh_id"] = id_maps["mesh"][str(group["mesh_id"])]
            combined_regions.append(cloned)

        for base_boundary in base_boundaries:
            cloned = copy.deepcopy(base_boundary)
            boundary_id = str(base_boundary["id"])
            cloned["id"] = id_maps["boundary"][boundary_id]
            cloned["name"] = f"{source.id} / {base_boundary.get('name', boundary_id)}"
            base_region_id = str(base_boundary["region_id"])
            cloned["region_id"] = (
                "deploy:exterior" if base_region_id == base_unbounded_id else id_maps["region"][base_region_id]
            )
            cloned["group"]["mesh_id"] = id_maps["mesh"][str(base_boundary["group"]["mesh_id"])]
            combined_boundaries.append(cloned)

        for base_interface in base_interfaces:
            cloned = copy.deepcopy(base_interface)
            cloned["id"] = id_maps["interface"][str(base_interface["id"])]
            cloned["name"] = f"{source.id} / {base_interface.get('name', base_interface['id'])}"
            cloned["bounded_boundary_id"] = id_maps["boundary"][str(base_interface["bounded_boundary_id"])]
            cloned["unbounded_boundary_id"] = id_maps["boundary"][str(base_interface["unbounded_boundary_id"])]
            # Julia reconstructs correspondence from the staged meshes; omit
            # unstable flattened element indices and avoid duplicating them N times.
            cloned["topology"] = {
                "fem_vertex_indices": [],
                "fem_to_bem_vertex_indices": [],
                "fem_face_indices": [],
                "bem_face_indices": [],
                "normal_sign": [],
                "max_coordinate_error": 0.0,
                "fem_facets_on_tetra_boundary": 0,
                "bem_boundary_edges": 0,
            }
            combined_interfaces.append(cloned)

        for base_component in base_components:
            cloned = copy.deepcopy(base_component)
            component_id = str(base_component["id"])
            cloned["id"] = id_maps["component"][component_id]
            cloned["name"] = f"{source.id} / {base_component.get('name', component_id)}"
            cloned["boundary_ids"] = [
                id_maps["boundary"][str(value)] for value in base_component.get("boundary_ids", ())
            ]
            parameters = cloned.get("parameters", {})
            for key in ("boundary_motion_weights", "boundary_motion_signs"):
                mapping = parameters.get(key)
                if isinstance(mapping, dict):
                    parameters[key] = {id_maps["boundary"][str(name)]: value for name, value in mapping.items()}
            if "motion_axis" in parameters:
                parameters["motion_axis"] = _transform_scene_vector(
                    np.asarray(parameters["motion_axis"], dtype=np.float64), source
                ).tolist()
            combined_components.append(cloned)

        for base_port in base_ports:
            cloned = copy.deepcopy(base_port)
            port_id = str(base_port["id"])
            cloned["id"] = id_maps["port"][port_id]
            cloned["name"] = f"{source.id} / {base_port.get('name', port_id)}"
            cloned["component_id"] = id_maps["component"][str(base_port["component_id"])]
            combined_ports.append(cloned)
            excitation_ids.append(str(cloned["id"]))
            for frequency_index, frequency_hz in enumerate(frequencies):
                gain_phase = 2.0 * math.pi * frequency_hz * source.delay_ms / 1000.0
                gain = (0.0 if source.muted else source.polarity * 10.0 ** (source.level_db / 20.0)) * np.exp(
                    1j * gain_phase
                )
                wire_gain = {"real": float(gain.real), "imag": float(gain.imag)}
                excitation_weights_sweep[frequency_index].append(wire_gain)
                if frequency_index == 0:
                    excitation_weights.append(wire_gain)

    for rigid_index, rigid in enumerate(rigid_objects):
        asset = cache.load_rigid_mesh(rigid.mesh_path) if cache is not None else _load_rigid_mesh_data(rigid.mesh_path)
        points = _transform_scene_points(
            asset.points * rigid.scale_to_meters,
            position_x_m=rigid.position_x_m,
            position_height_m=rigid.position_height_m,
            position_z_m=rigid.position_z_m,
            pitch_deg=rigid.pitch_deg,
            yaw_deg=rigid.yaw_deg,
            roll_deg=rigid.roll_deg,
        )
        mesh_id = f"deploy:rigid-mesh:{rigid_index}:{rigid.id}"
        boundary_id = f"deploy:rigid-boundary:{rigid_index}:{rigid.id}"
        target = instance_dir / f"rigid-{rigid_index:02d}.msh"
        _write_gmsh22_surface(target, [(points, asset.triangles)])
        combined_meshes.append(
            {
                "id": mesh_id,
                "name": rigid.id,
                "file": str(target),
                "purpose": "bem_surface",
                "scale_to_m": 1.0,
                "translation_m": [0.0, 0.0, 0.0],
            }
        )
        unbounded_mesh_ids.append(mesh_id)
        combined_boundaries.append(
            {
                "id": boundary_id,
                "name": rigid.id,
                "region_id": "deploy:exterior",
                "kind": "rigid",
                "group": {"mesh_id": mesh_id, "dimension": 2, "tag": 1, "name": None},
                "parameters": {},
            }
        )

    exterior = copy.deepcopy(unbounded[0])
    exterior["id"] = "deploy:exterior"
    exterior["name"] = "Deploy exterior"
    exterior["mesh_ids"] = unbounded_mesh_ids
    combined_regions.insert(0, exterior)
    combined_system = {
        **{
            key: copy.deepcopy(value)
            for key, value in base_system.items()
            if key
            not in {"id", "name", "meshes", "regions", "boundaries", "interfaces", "components", "excitation_ports"}
        },
        "id": "deploy:coupled-array",
        "name": "Deploy coupled array",
        "meshes": combined_meshes,
        "regions": combined_regions,
        "boundaries": combined_boundaries,
        "interfaces": combined_interfaces,
        "components": combined_components,
        "excitation_ports": combined_ports,
    }
    request: dict[str, Any] = {
        "schema_version": 1,
        "schema": DEPLOY_COUPLED_SCHEMA,
        "compiled_system": combined_system,
        "frequencies_hz": frequencies,
        "excitation_port_ids": excitation_ids,
        "outputs": [
            {
                "id": "deploy:field-pressure",
                "quantity": "exterior_pressure",
                "target_ids": [],
                "options": {
                    "points_m": observation_points.tolist(),
                    "excitation_weights": excitation_weights,
                    "excitation_weights_sweep": excitation_weights_sweep,
                },
            }
        ],
        "solver_options": {
            "precision": "float32",
            "bem_backend": str(payload.get("backend", "cuda")).strip().lower(),
            "symmetry": "ground",
            "quadrature_order": int(payload.get("quadratureOrder", 2)),
            "singular_order": int(payload.get("singularOrder", 3)),
            "static_condensation": True,
            "validation_diagnostics": False,
            "cache_frequency_invariant": True,
            "transducer_reference_voltage_v": 2.83,
        },
        "deploy": {
            "frequency_hz": requested_frequency,
            "rows": observation_shape[0],
            "columns": observation_shape[1],
            "sample_indices": observation_sample_indices.tolist(),
            "source_count": len(sources),
            "rigid_object_count": len(rigid_objects),
            "solution_key": str(payload.get("solutionKey", "")),
        },
    }
    if status_callback is not None:
        status_callback("Serializing exact Level 3 array request")
    request_path = work_path / "coupled-request.json"
    request_path.write_text(json.dumps(request, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return request_path, request


def _write_transformed_coupled_mesh(
    source_path: Path,
    target_path: Path,
    mesh_resource: dict[str, Any],
    placement: DeploySourcePlacement,
) -> None:
    mesh = meshio.read(source_path)
    points = np.asarray(mesh.points, dtype=np.float64) * float(mesh_resource.get("scale_to_m", 1.0))
    points += np.asarray(mesh_resource.get("translation_m", (0.0, 0.0, 0.0)), dtype=np.float64)
    transformed = _transform_scene_points(
        points,
        position_x_m=placement.position_x_m,
        position_height_m=placement.position_height_m,
        position_z_m=placement.position_z_m,
        pitch_deg=placement.pitch_deg,
        yaw_deg=placement.yaw_deg,
        roll_deg=placement.roll_deg,
    )
    output = meshio.Mesh(
        points=transformed,
        cells=mesh.cells,
        point_data=mesh.point_data,
        cell_data=mesh.cell_data,
        field_data=mesh.field_data,
        cell_sets=mesh.cell_sets,
    )
    purpose = str(mesh_resource.get("purpose", ""))
    if purpose == "bem_surface":
        meshio.write(target_path, output, file_format="gmsh22", binary=False)
    else:
        meshio.write(target_path, output, file_format="gmsh", binary=False)


def _transform_scene_vector(vector: np.ndarray, placement: DeploySourcePlacement) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64).reshape(1, 3)
    origin = np.zeros((1, 3), dtype=np.float64)
    transformed_vector = _transform_scene_points(
        values,
        position_x_m=placement.position_x_m,
        position_height_m=placement.position_height_m,
        position_z_m=placement.position_z_m,
        pitch_deg=placement.pitch_deg,
        yaw_deg=placement.yaw_deg,
        roll_deg=placement.roll_deg,
    )
    transformed_origin = _transform_scene_points(
        origin,
        position_x_m=placement.position_x_m,
        position_height_m=placement.position_height_m,
        position_z_m=placement.position_z_m,
        pitch_deg=placement.pitch_deg,
        yaw_deg=placement.yaw_deg,
        roll_deg=placement.roll_deg,
    )
    result = transformed_vector[0] - transformed_origin[0]
    norm = float(np.linalg.norm(result))
    return result if norm == 0.0 else result / norm


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


def _transform_scene_points(
    points_m: np.ndarray,
    *,
    position_x_m: float,
    position_height_m: float,
    position_z_m: float,
    pitch_deg: float,
    yaw_deg: float,
    roll_deg: float,
) -> np.ndarray:
    package_frame = np.column_stack((points_m[:, 0], points_m[:, 2], -points_m[:, 1]))
    return transform_package_points(
        package_frame,
        position_x_m=position_x_m,
        position_height_m=position_height_m,
        position_z_m=position_z_m,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        roll_deg=roll_deg,
    )


def _write_gmsh22_surface(path: Path, components: list[tuple[np.ndarray, np.ndarray]]) -> None:
    point_count = sum(points.shape[0] for points, _ in components)
    face_count = sum(triangles.shape[0] for _, triangles in components)
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(point_count)]
    vertex_offset = 0
    for points, _ in components:
        lines.extend(
            f"{vertex_offset + index + 1} {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}"
            for index, point in enumerate(points)
        )
        vertex_offset += points.shape[0]
    lines.extend(("$EndNodes", "$Elements", str(face_count)))
    vertex_offset = 0
    element_index = 1
    for component_index, (points, triangles) in enumerate(components, start=1):
        for face in triangles:
            a, b, c = (int(value) + vertex_offset + 1 for value in face)
            lines.append(f"{element_index} 2 2 {component_index} {component_index} {a} {b} {c}")
            element_index += 1
        vertex_offset += points.shape[0]
    lines.extend(("$EndElements", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_deploy_solve_request(
    payload: object,
    work_dir: str | Path,
    *,
    cache: DeploySolveCache | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Validate a renderer request and stage fixed-source instances for BEAT."""

    if not isinstance(payload, dict):
        raise ValueError("Deploy solve request must be an object.")
    if status_callback is not None:
        status_callback("Preparing scene geometry")
    package_path = Path(str(payload.get("packagePath", ""))).expanduser().resolve()
    if package_path.suffix.lower() != ".blabsp" or not package_path.is_file():
        raise ValueError("Deploy Level 2 requires an existing .blabsp package path.")

    package_data = cache.load_package(package_path) if cache is not None else _load_deploy_package_data(package_path)
    manifest = package_data.manifest
    if int(manifest.get("fidelity_level", 0)) < 2:
        raise ValueError("Deploy Level 2 requires a package containing fixed distributed sources.")

    requested_frequency = float(payload.get("frequencyHz", 0.0))
    if not math.isfinite(requested_frequency) or requested_frequency <= 0.0:
        raise ValueError("Deploy solve frequency must be finite and positive.")
    frequencies = package_data.frequencies
    if frequencies.size == 0:
        raise ValueError("Speaker package contains no frequencies.")
    frequency_index = int(np.argmin(np.abs(frequencies - requested_frequency)))
    frequency_hz = float(frequencies[frequency_index])
    tolerance_hz = max(1e-4, abs(frequency_hz) * 1e-6)
    if abs(frequency_hz - requested_frequency) > tolerance_hz:
        raise ValueError("Deploy Level 2 initially requires an exact exported package frequency.")
    close_pair_quadrature_override = payload.get("closePairQuadratureOrder")
    close_pair_quadrature_order = int(
        CLOSE_PAIR_QUADRATURE_ORDER if close_pair_quadrature_override is None else close_pair_quadrature_override
    )
    if not 4 <= close_pair_quadrature_order <= 16:
        raise ValueError("Deploy close-pair quadrature order must be between 4 and 16.")

    raw_sources = payload.get("sources")
    if raw_sources is None and payload.get("source") is not None:
        raw_sources = [{"id": "subwoofer-1", **payload["source"]}]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Deploy solve requires at least one source.")
    if len(raw_sources) > 16:
        raise ValueError("Deploy solve supports at most 16 sources.")
    sources = [DeploySourcePlacement.from_payload(raw) for raw in raw_sources]
    source_ids = [source.id for source in sources]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Deploy source ids must be unique.")
    raw_rigid_objects = payload.get("rigidObjects", [])
    if not isinstance(raw_rigid_objects, list):
        raise ValueError("Deploy rigidObjects must be an array.")
    if len(raw_rigid_objects) > 16:
        raise ValueError("Deploy solve supports at most 16 rigid objects.")
    rigid_objects = [DeployRigidPlacement.from_payload(raw) for raw in raw_rigid_objects]
    rigid_ids = [item.id for item in rigid_objects]
    if len(set(rigid_ids)) != len(rigid_ids) or set(rigid_ids).intersection(source_ids):
        raise ValueError("Deploy boundary object ids must be unique.")
    backend = str(payload.get("backend", "cuda")).strip().lower()
    raw_observation_points = payload.get("observationPointsM")
    observation = None
    if raw_observation_points is None:
        observation = DeployObservationPlane.from_payload(payload.get("observation"))
        points_m, observation_sample_indices = observation.points()
        observation_shape = [observation.rows, observation.columns]
    else:
        points_m = np.asarray(raw_observation_points, dtype=np.float32)
        if points_m.ndim != 2 or points_m.shape[1] != 3 or points_m.shape[0] == 0:
            raise ValueError("Deploy observationPointsM must contain one or more XYZ points.")
        if points_m.shape[0] > 1024:
            raise ValueError("Deploy observationPointsM supports at most 1,024 points.")
        if not np.all(np.isfinite(points_m)):
            raise ValueError("Deploy observationPointsM values must be finite.")
        if np.any(points_m[:, 1] < -GROUND_TOLERANCE_M):
            raise ValueError("Deploy observation points cannot be below the ground plane.")
        observation_sample_indices = np.arange(points_m.shape[0], dtype=np.int64)
        observation_shape = [1, int(points_m.shape[0])]
    solution_key = str(payload.get("solutionKey", "")).strip() or json.dumps(
        {
            "package": str(package_path),
            "frequency_hz": frequency_hz,
            "sources": raw_sources,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    triangles = package_data.triangles
    points = package_data.points
    pressure = package_data.pressure
    normal = package_data.normal
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Fixed-source points must have shape (node, 3).")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Fixed-source triangles must have shape (face, 3).")
    if normal.ndim != 3 or normal.shape[0] != frequencies.size or normal.shape[2] != triangles.shape[0]:
        raise ValueError("Fixed-source Neumann traces do not align with package frequencies and faces.")
    if pressure.ndim != 3 or pressure.shape[0] != frequencies.size or pressure.shape[2] != points.shape[0]:
        raise ValueError("Fixed-source pressure traces do not align with package frequencies and nodes.")
    if normal.shape[1] < 1:
        raise ValueError("Fixed-source package contains no excitation ports.")
    if pressure.shape[1] != normal.shape[1]:
        raise ValueError("Fixed-source pressure and Neumann traces have different excitation counts.")
    excitation_indices = _logical_excitation_indices(manifest, normal.shape[1])
    logical_normal = _combined_excitation_trace(normal, frequency_index, excitation_indices)
    logical_pressure = _combined_excitation_trace(pressure, frequency_index, excitation_indices)

    components: list[DeployBoundaryComponent] = []
    face_offset = 0
    vertex_offset = 0
    for source in sources:
        transformed = transform_package_points(
            points,
            position_x_m=source.position_x_m,
            position_height_m=source.position_height_m,
            position_z_m=source.position_z_m,
            pitch_deg=source.pitch_deg,
            roll_deg=source.roll_deg,
            yaw_deg=source.yaw_deg,
        )
        phase = 2.0 * math.pi * frequency_hz * source.delay_ms / 1000.0
        gain = (0.0 if source.muted else source.polarity * 10.0 ** (source.level_db / 20.0)) * np.exp(1j * phase)
        component = DeployBoundaryComponent(
            id=source.id,
            kind="speaker",
            fingerprint=package_data.fingerprint,
            points=transformed,
            triangles=triangles,
            face_offset=face_offset,
            vertex_offset=vertex_offset,
            q_neumann=np.asarray(logical_normal * gain, dtype=np.complex64),
            reference_pressure=np.asarray(logical_pressure * gain, dtype=np.complex64),
        )
        components.append(component)
        face_offset += triangles.shape[0]
        vertex_offset += points.shape[0]

    rigid_meshes = [
        cache.load_rigid_mesh(item.mesh_path) if cache is not None else _load_rigid_mesh_data(item.mesh_path)
        for item in rigid_objects
    ]
    for rigid, mesh in zip(rigid_objects, rigid_meshes, strict=True):
        scaled_points = mesh.points * rigid.scale_to_meters
        transformed = _transform_scene_points(
            scaled_points,
            position_x_m=rigid.position_x_m,
            position_height_m=rigid.position_height_m,
            position_z_m=rigid.position_z_m,
            pitch_deg=rigid.pitch_deg,
            yaw_deg=rigid.yaw_deg,
            roll_deg=rigid.roll_deg,
        )
        component = DeployBoundaryComponent(
            id=rigid.id,
            kind="rigid",
            fingerprint=mesh.fingerprint,
            points=transformed,
            triangles=mesh.triangles,
            face_offset=face_offset,
            vertex_offset=vertex_offset,
            q_neumann=np.zeros(mesh.triangles.shape[0], dtype=np.complex64),
            reference_pressure=np.zeros(mesh.points.shape[0], dtype=np.complex64),
        )
        components.append(component)
        face_offset += mesh.triangles.shape[0]
        vertex_offset += mesh.points.shape[0]

    for component in components:
        minimum_y = float(np.min(component.points[:, 1]))
        if minimum_y < -GROUND_TOLERANCE_M:
            raise ValueError(
                f"Deploy boundary object {component.id!r} extends {abs(minimum_y):.6f} m below the ground plane."
            )

    proximity_pairs: list[dict[str, Any]] = []
    close_face_pairs: list[list[int]] = []
    minimum_surface_distance_m: float | None = None
    if status_callback is not None:
        status_callback("Validating boundary spacing")
    for first_index in range(len(components)):
        for second_index in range(first_index + 1, len(components)):
            first = components[first_index]
            second = components[second_index]
            first_minimum = np.min(first.points, axis=0)
            first_maximum = np.max(first.points, axis=0)
            second_minimum = np.min(second.points, axis=0)
            second_maximum = np.max(second.points, axis=0)
            object_separation = np.maximum(
                0.0,
                np.maximum(first_minimum - second_maximum, second_minimum - first_maximum),
            )
            object_distance_m = float(np.linalg.norm(object_separation))
            violation = (
                first_surface_pair_within(
                    first.points,
                    first.triangles,
                    second.points,
                    second.triangles,
                    max(0.0, SOURCE_SURFACE_PADDING_M - GROUND_TOLERANCE_M),
                )
                if object_distance_m < SOURCE_SURFACE_PADDING_M
                else None
            )
            if violation is not None:
                raise ValueError(
                    f"Deploy boundary objects {first.id!r} and {second.id!r} have "
                    f"{violation.distance_m * 1000.0:.3f} mm surface spacing; at least "
                    f"{SOURCE_SURFACE_PADDING_M * 1000.0:.1f} mm is required."
                )
            face_pairs = (
                surface_face_pairs_within(
                    first.points,
                    first.triangles,
                    second.points,
                    second.triangles,
                    CLOSE_PAIR_DISTANCE_M,
                    exact=False,
                )
                if object_distance_m <= CLOSE_PAIR_DISTANCE_M
                else []
            )
            distance_m = min((item.distance_m for item in face_pairs), default=object_distance_m)
            minimum_surface_distance_m = (
                distance_m if minimum_surface_distance_m is None else min(minimum_surface_distance_m, distance_m)
            )
            pair = {
                "source_a": first.id,
                "source_b": second.id,
                "kind_a": first.kind,
                "kind_b": second.kind,
                "distance_m": distance_m,
                "face_a": face_pairs[0].face_a if face_pairs else -1,
                "face_b": face_pairs[0].face_b if face_pairs else -1,
                "close": bool(face_pairs),
            }
            if pair["close"]:
                for face_pair in face_pairs:
                    first_face = first.face_offset + face_pair.face_a
                    second_face = second.face_offset + face_pair.face_b
                    correction_order = (
                        close_pair_quadrature_order
                        if close_pair_quadrature_override is not None
                        else (8 if face_pair.distance_m <= 0.015 else 6 if face_pair.distance_m <= 0.03 else 4)
                    )
                    close_face_pairs.append([first_face, second_face, correction_order])
                    close_face_pairs.append([second_face, first_face, correction_order])
                pair["near_face_pair_count"] = len(face_pairs)
            else:
                pair["near_face_pair_count"] = 0
            proximity_pairs.append(pair)

    # A rigid half-space Green's function adds a positive image of every
    # cabinet boundary across the world Y=0 plane. Correct its near interactions
    # with the same tiered quadrature used for adjacent real cabinets. Exact
    # coincident/edge/vertex image pairs are omitted here because BEAT's image
    # Duffy cache owns those singular interactions. Use conservative face-AABB
    # distances for the correction tiers, matching the direct close-pair path;
    # exact scalar triangle distances are too costly for interactive staging.
    if status_callback is not None:
        status_callback("Building close-pair correction map")
    ground_image_face_pairs: list[list[int]] = []
    singular_tolerance_squared = GROUND_IMAGE_SINGULAR_TOLERANCE_M**2
    reflected_components = []
    for component in components:
        reflected = component.points.copy()
        reflected[:, 1] *= -1.0
        reflected_components.append(reflected)

    def non_singular_ground_pairs(
        test_component: DeployBoundaryComponent,
        trial_component: DeployBoundaryComponent,
        trial_points: np.ndarray,
    ) -> list[Any]:
        test_faces = test_component.points[test_component.triangles]
        trial_faces = trial_points[trial_component.triangles]
        filtered = []
        for face_pair in surface_face_pairs_within(
            test_component.points,
            test_component.triangles,
            trial_points,
            trial_component.triangles,
            CLOSE_PAIR_DISTANCE_M,
            exact=False,
        ):
            vertex_deltas = (
                test_faces[face_pair.face_a, :, np.newaxis, :] - trial_faces[face_pair.face_b, np.newaxis, :, :]
            )
            if np.any(np.sum(vertex_deltas * vertex_deltas, axis=2) <= singular_tolerance_squared):
                continue
            filtered.append(face_pair)
        return filtered

    ground_pair_cache = cache.ground_image_pairs if cache is not None else {}
    ground_pair_sets: list[tuple[int, int, list[Any]]] = []
    for component_index, component in enumerate(components):
        self_key = (
            component.fingerprint,
            hash(component.points.tobytes()),
            float(np.min(component.points[:, 1])),
            float(np.max(component.points[:, 1])),
            component.points.shape[0],
        )
        self_pairs = ground_pair_cache.get(self_key)
        if self_pairs is None:
            self_pairs = non_singular_ground_pairs(component, component, reflected_components[component_index])
            ground_pair_cache[self_key] = self_pairs
        ground_pair_sets.append((component_index, component_index, self_pairs))

    close_distance_squared = CLOSE_PAIR_DISTANCE_M**2
    for test_index, test_component in enumerate(components):
        test_minimum = np.min(test_component.points, axis=0)
        test_maximum = np.max(test_component.points, axis=0)
        for trial_index, trial_points in enumerate(reflected_components):
            if test_index == trial_index:
                continue
            trial_minimum = np.min(trial_points, axis=0)
            trial_maximum = np.max(trial_points, axis=0)
            separation = np.maximum(
                0.0,
                np.maximum(test_minimum - trial_maximum, trial_minimum - test_maximum),
            )
            if float(np.dot(separation, separation)) > close_distance_squared:
                continue
            ground_pair_sets.append(
                (
                    test_index,
                    trial_index,
                    non_singular_ground_pairs(test_component, components[trial_index], trial_points),
                )
            )

    for test_index, trial_index, face_pairs in ground_pair_sets:
        test_component = components[test_index]
        trial_component = components[trial_index]
        for face_pair in face_pairs:
            correction_order = (
                close_pair_quadrature_order
                if close_pair_quadrature_override is not None
                else (8 if face_pair.distance_m <= 0.015 else 6 if face_pair.distance_m <= 0.03 else 4)
            )
            ground_image_face_pairs.append(
                [
                    test_component.face_offset + face_pair.face_a,
                    trial_component.face_offset + face_pair.face_b,
                    correction_order,
                ]
            )
    ground_image_face_pairs.sort(key=lambda pair: (pair[0], pair[1]))

    q_neumann = np.concatenate([component.q_neumann for component in components])
    reference_pressure = np.concatenate([component.reference_pressure for component in components])
    reference_pressure_mask = np.concatenate(
        [np.full(component.points.shape[0], component.kind == "speaker", dtype=np.uint8) for component in components]
    )
    if not np.all(np.isfinite(q_neumann)) or not np.all(np.isfinite(reference_pressure)):
        raise ValueError("Fixed-source boundary traces contain non-finite values.")

    staged_mesh = work_path / "exterior.msh"
    _write_gmsh22_surface(staged_mesh, [(component.points, component.triangles) for component in components])
    if points_m.shape[0] == 0:
        raise ValueError("Deploy solve has no sampling points on or above the ground plane.")
    medium = manifest.get("medium", {})
    burton_miller_assembly = (
        str(
            payload.get(
                "burtonMillerAssembly",
                "direct_system" if backend == "cuda" else "operator_matrices",
            )
        )
        .strip()
        .lower()
    )
    if burton_miller_assembly not in {"direct_system", "operator_matrices"}:
        raise ValueError("Deploy burtonMillerAssembly must be 'direct_system' or 'operator_matrices'.")
    request: dict[str, Any] = {
        "schema": DEPLOY_SOLVE_SCHEMA,
        "schema_version": DEPLOY_SOLVE_SCHEMA_VERSION,
        "beat_engine_backend": backend,
        "burton_miller_assembly": burton_miller_assembly,
        "solution_key": solution_key,
        "include_complex_pressure": bool(payload.get("includeComplexPressure", False)),
        "frequency_hz": frequency_hz,
        "mesh_file": str(staged_mesh),
        "mesh_scale_factor": 1.0,
        "mesh_is_world_space": True,
        "source_transforms": [
            {
                "id": "combined-boundary",
                "position_m": [0.0, 0.0, 0.0],
                "pitch_deg": 0.0,
                "yaw_deg": 0.0,
                "roll_deg": 0.0,
            }
        ],
        "boundary_components": [
            {
                "id": component.id,
                "kind": component.kind,
                "vertex_offset": component.vertex_offset,
                "vertex_count": int(component.points.shape[0]),
                "face_offset": component.face_offset,
                "face_count": int(component.triangles.shape[0]),
            }
            for component in components
        ],
        "boundary_neumann": {
            "real": q_neumann.real.tolist(),
            "imag": q_neumann.imag.tolist(),
        },
        "reference_boundary_pressure": {
            "real": reference_pressure.real.tolist(),
            "imag": reference_pressure.imag.tolist(),
        },
        "reference_boundary_pressure_mask": reference_pressure_mask.tolist(),
        "density_kg_per_m3": float(medium.get("density_kg_per_m3", 1.21)),
        "sound_speed_m_per_s": float(medium.get("sound_speed_m_per_s", 343.0)),
        "quadrature_order": int(payload.get("quadratureOrder", 2)),
        "singular_order": int(payload.get("singularOrder", 3)),
        "close_pair_quadrature_order": close_pair_quadrature_order,
        "boundary": {
            "ground_plane": {
                "type": "rigid_half_space",
                "axis": "y",
                "offset_m": 0.0,
                "reflection_coefficient": 1.0,
            },
        },
        "proximity": {
            "surface_padding_m": SOURCE_SURFACE_PADDING_M,
            "close_pair_distance_m": CLOSE_PAIR_DISTANCE_M,
            "minimum_surface_distance_m": minimum_surface_distance_m,
            "pairs": proximity_pairs,
            "close_face_pairs": close_face_pairs,
            "ground_image_close_face_pairs": ground_image_face_pairs,
        },
        "provenance": {
            "package_path": str(package_path),
            "package_name": str(manifest.get("name", package_path.stem)),
            "frequency_index": frequency_index,
            "source_count": len(sources),
            "rigid_object_count": len(rigid_objects),
            "boundary_component_count": len(components),
            "source_ids": source_ids,
            "rigid_object_ids": rigid_ids,
            "package_node_count": int(points.shape[0]),
            "package_face_count": int(triangles.shape[0]),
            "node_count": int(sum(component.points.shape[0] for component in components)),
            "face_count": int(sum(component.triangles.shape[0] for component in components)),
            "excitation_index": 0,
            "excitation_indices": list(excitation_indices),
            "excitation_port_ids": [str(manifest["excitation_port_ids"][index]) for index in excitation_indices],
            "exterior_domain": "rigid_y0_half_space",
        },
    }
    if observation is None or backend != "cuda":
        request["observation_points_m"] = points_m.tolist()
        request["observation_shape"] = observation_shape
        request["observation_sample_indices"] = observation_sample_indices.tolist()
    if observation is not None:
        request["observation_plane"] = observation.wire()
    if status_callback is not None:
        status_callback("Serializing BEAT request")
    request_path = work_path / "request.json"
    request_path.write_text(json.dumps(request, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return request_path, request


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


def prepare_deploy_rom_request(
    payload: object,
    work_dir: str | Path,
    *,
    cache: DeploySolveCache | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Build a matrix-free Schur-eliminated Deploy request for a parity ROM."""

    if not isinstance(payload, dict):
        raise ValueError("Deploy Level 3 ROM request must be an object.")
    package_path = Path(str(payload.get("packagePath", ""))).expanduser().resolve()
    package = cache.load_package(package_path) if cache is not None else _load_deploy_package_data(package_path)
    model = package.coupled_model
    if not isinstance(model, dict) or model.get("representation") != "parity_petrov_galerkin_rom":
        raise ValueError("Deploy parity-ROM solve requires a parity Petrov-Galerkin package.")
    arrays = model.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("Deploy parity-ROM package did not load its reduced arrays.")

    request_path, request = prepare_deploy_solve_request(
        payload,
        work_dir,
        cache=cache,
        status_callback=status_callback,
    )
    requested_frequency = float(request["frequency_hz"])
    rom_frequencies = np.asarray(arrays["frequencies_hz"], dtype=np.float64)
    if rom_frequencies.size == 0 or not np.all(np.isfinite(rom_frequencies)):
        raise ValueError("Deploy parity-ROM package has no finite sweep frequencies.")
    frequency_index = int(np.argmin(np.abs(rom_frequencies - requested_frequency)))
    tolerance = max(1e-4, abs(requested_frequency) * 1e-6)
    if abs(float(rom_frequencies[frequency_index]) - requested_frequency) > tolerance:
        raise ValueError(f"Parity ROM does not contain {requested_frequency:g} Hz.")

    selected = {
        name: np.asarray(arrays[name][frequency_index], dtype=np.complex64)
        for name in (
            "k",
            "c",
            "d",
            "b",
            "e",
            "velocity",
            "current",
            "velocity_drive",
            "current_drive",
        )
    }
    rank = int(model.get("rank_per_sector", selected["k"].shape[-1]))
    symmetry_mode = str(model.get("symmetry_mode", "xy")).lower()
    expected_image_count = {"off": 1, "x": 2, "xy": 4}.get(symmetry_mode)
    if expected_image_count is None:
        raise ValueError(f"Parity ROM has unsupported symmetry mode {symmetry_mode!r}.")
    image_count = int(model.get("image_count", expected_image_count))
    sector_signs = model.get("sector_signs")
    if not isinstance(sector_signs, list) or len(sector_signs) != expected_image_count:
        raise ValueError("Parity ROM sector count does not match its symmetry mode.")
    sector_count = len(sector_signs)
    if image_count != expected_image_count:
        raise ValueError("Parity ROM image count does not match its symmetry mode.")
    if selected["k"].shape != (sector_count, rank, rank):
        raise ValueError("Parity ROM K array has an invalid shape.")
    node_orbits = model.get("node_orbits")
    face_orbits = model.get("face_orbits")
    if not isinstance(node_orbits, list) or not isinstance(face_orbits, list):
        raise ValueError("Parity ROM is missing boundary orbit maps.")
    if any(len(orbit) != image_count for orbit in (*node_orbits, *face_orbits)):
        raise ValueError("Parity ROM boundary orbit width does not match its symmetry mode.")
    if selected["c"].shape != (sector_count, rank, len(node_orbits)):
        raise ValueError("Parity ROM C array does not align with its node orbits.")
    if selected["d"].shape[:2] != (sector_count, len(face_orbits)):
        raise ValueError("Parity ROM D array does not align with its face orbits.")

    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Deploy parity-ROM solve requires at least one source.")
    sources = [DeploySourcePlacement.from_payload(raw) for raw in raw_sources]
    source_components = request["boundary_components"][: len(sources)]
    input_count = selected["b"].shape[-1]
    reference_voltage = float(payload.get("transducerReferenceVoltageV", 2.83))
    instances = []
    for source, component in zip(sources, source_components, strict=True):
        phase = 2.0 * math.pi * requested_frequency * source.delay_ms / 1000.0
        gain = (0.0 if source.muted else source.polarity * 10.0 ** (source.level_db / 20.0)) * np.exp(1j * phase)
        drive = np.full(input_count, reference_voltage * gain, dtype=np.complex64)
        instances.append(
            {
                "id": source.id,
                "node_offset": int(component["vertex_offset"]),
                "face_offset": int(component["face_offset"]),
                "input_real": drive.real.tolist(),
                "input_imag": drive.imag.tolist(),
            }
        )

    transducer_count = int(selected["velocity"].shape[-2])
    physical_system = package.manifest.get("physical_system", {})
    package_transducers = (
        [
            item
            for item in physical_system.get("components", [])
            if isinstance(item, dict) and item.get("kind") == "electrodynamic_transducer"
        ]
        if isinstance(physical_system, dict)
        else []
    )
    if len(package_transducers) != transducer_count:
        package_transducers = [
            {"id": f"transducer:{index}", "name": f"Transducer {index + 1}", "parameters": {}}
            for index in range(transducer_count)
        ]
    transducers = [
        {
            "id": f"{source.id}:{package_transducer['id']}",
            "name": f"{str(raw_source.get('name', source.id)).strip() or source.id} / {package_transducer['name']}",
            "source_id": source.id,
            "transducer_index": transducer_index,
            "physical_driver_orbit_count": int(
                package_transducer.get("parameters", {}).get("physical_driver_orbit_count", 1)
            ),
        }
        for raw_source, source in zip(raw_sources, sources, strict=True)
        for transducer_index, package_transducer in enumerate(package_transducers)
    ]
    speakers = [
        {
            "id": source.id,
            "name": str(raw_source.get("name", source.id)).strip() or source.id,
        }
        for raw_source, source in zip(raw_sources, sources, strict=True)
    ]

    binary_path = Path(work_dir).resolve() / "speaker-rom.bin"
    binary_arrays = _write_deploy_binary_arrays(binary_path, selected)
    face_count = sum(int(component["face_count"]) for component in request["boundary_components"])
    node_count = sum(int(component["vertex_count"]) for component in request["boundary_components"])
    zeros_faces = np.zeros(face_count, dtype=np.float32).tolist()
    zeros_nodes = np.zeros(node_count, dtype=np.float32).tolist()
    request.update(
        schema="boundary_lab_deploy_rom",
        schema_version=1,
        burton_miller_assembly="direct_system",
        boundary_neumann={"real": zeros_faces, "imag": zeros_faces},
        reference_boundary_pressure={"real": zeros_nodes, "imag": zeros_nodes},
        reference_boundary_pressure_mask=np.zeros(node_count, dtype=np.uint8).tolist(),
        rom={
            "format_version": 1,
            "representation": "parity_petrov_galerkin_rom",
            "symmetry_mode": symmetry_mode,
            "image_count": image_count,
            "rank_per_sector": rank,
            "sector_signs": sector_signs,
            "node_orbits": node_orbits,
            "face_orbits": face_orbits,
            "instances": instances,
            "binary_arrays": binary_arrays,
            "gmres_tolerance": float(payload.get("romGmresTolerance", 1e-4)),
            "gmres_max_iterations": int(payload.get("romGmresMaxIterations", 30)),
        },
        transducers=transducers,
        speakers=speakers,
    )
    request_path.write_text(json.dumps(request, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    if status_callback is not None:
        status_callback(f"Prepared rank-{rank} parity-ROM boundary feedback")
    return request_path, request


def prepare_deploy_rom_microphone_sweep_request(
    payload: object,
    work_dir: str | Path,
    *,
    cache: DeploySolveCache | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Build one geometry-cached microphone sweep for a parity speaker ROM."""

    if not isinstance(payload, dict):
        raise ValueError("Deploy Level 3 ROM microphone sweep request must be an object.")
    package_path = Path(str(payload.get("packagePath", ""))).expanduser().resolve()
    package = cache.load_package(package_path) if cache is not None else _load_deploy_package_data(package_path)
    model = package.coupled_model
    if not isinstance(model, dict) or model.get("representation") != "parity_petrov_galerkin_rom":
        raise ValueError("Deploy parity-ROM microphone sweep requires a parity Petrov-Galerkin package.")
    arrays = model.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("Deploy parity-ROM package did not load its reduced arrays.")

    rom_frequencies = np.asarray(arrays["frequencies_hz"], dtype=np.float64)
    raw_frequencies = payload.get("frequenciesHz")
    if raw_frequencies is None:
        package_frequencies = sorted({float(value) for value in package.frequencies})
    elif isinstance(raw_frequencies, list) and raw_frequencies:
        package_frequencies = sorted({float(value) for value in raw_frequencies})
    else:
        raise ValueError("Deploy Level 3 ROM frequenciesHz must be a non-empty array.")
    if any(not math.isfinite(value) or value <= 0.0 for value in package_frequencies):
        raise ValueError("Deploy Level 3 ROM frequencies must be finite and positive.")
    frequency_pairs: list[tuple[float, int]] = []
    for frequency_hz in package_frequencies:
        index = int(np.argmin(np.abs(rom_frequencies - frequency_hz)))
        tolerance = max(1e-4, abs(frequency_hz) * 1e-6)
        if abs(float(rom_frequencies[index]) - frequency_hz) <= tolerance:
            frequency_pairs.append((frequency_hz, index))
    if not frequency_pairs:
        raise ValueError("Parity ROM and speaker package have no common microphone-sweep frequencies.")

    first_payload = {
        **payload,
        "frequencyHz": frequency_pairs[0][0],
        "includeComplexPressure": True,
        "solutionKey": "coupled-rom-microphone-sweep-template",
    }
    request_path, request = prepare_deploy_rom_request(
        first_payload,
        work_dir,
        cache=cache,
        status_callback=status_callback,
    )
    base_rom = request["rom"]
    base_instances = list(base_rom["instances"])
    sources = [DeploySourcePlacement.from_payload(raw) for raw in payload.get("sources", [])]
    if len(sources) != len(base_instances):
        raise ValueError("Parity-ROM microphone sweep source count does not match the staged instances.")

    array_names = (
        "k",
        "c",
        "d",
        "b",
        "e",
        "velocity",
        "current",
        "velocity_drive",
        "current_drive",
    )
    sweep_entries: list[dict[str, Any]] = []
    reference_voltage = float(payload.get("transducerReferenceVoltageV", 2.83))
    if isinstance(cache, DeploySolveCache):
        staged, stage_cache_hit = cache.stage_rom_sweep_arrays(package, frequency_pairs, array_names)
        frequency_descriptors = staged.frequency_descriptors
        binary_bytes = staged.binary_bytes
        binary_bytes_written = 0 if stage_cache_hit else binary_bytes
    else:
        binary_values: dict[str, np.ndarray] = {}
        descriptor_names: list[dict[str, str]] = []
        for sweep_index, (_frequency_hz, array_index) in enumerate(frequency_pairs):
            names: dict[str, str] = {}
            for name in array_names:
                binary_name = f"{name}_{sweep_index}"
                binary_values[binary_name] = np.asarray(arrays[name][array_index], dtype=np.complex64)
                names[name] = binary_name
            descriptor_names.append(names)
        binary_path = Path(work_dir).resolve() / "speaker-rom-sweep.bin"
        all_descriptors = _write_deploy_binary_arrays(binary_path, binary_values)
        frequency_descriptors = tuple(
            {name: all_descriptors[binary_name] for name, binary_name in names.items()} for names in descriptor_names
        )
        stage_cache_hit = False
        binary_bytes = binary_path.stat().st_size
        binary_bytes_written = binary_bytes

    if status_callback is not None:
        status_callback(
            "Reusing staged Level 3 ROM sweep data" if stage_cache_hit else "Staging Level 3 ROM sweep data"
        )
    for sweep_index, (frequency_hz, array_index) in enumerate(frequency_pairs):
        input_count = int(np.asarray(arrays["b"][array_index]).shape[-1])
        instances: list[dict[str, Any]] = []
        for source, base_instance in zip(sources, base_instances, strict=True):
            phase = 2.0 * math.pi * frequency_hz * source.delay_ms / 1000.0
            gain = (0.0 if source.muted else source.polarity * 10.0 ** (source.level_db / 20.0)) * np.exp(1j * phase)
            drive = np.full(input_count, reference_voltage * gain, dtype=np.complex64)
            instances.append(
                {
                    "id": base_instance["id"],
                    "node_offset": base_instance["node_offset"],
                    "face_offset": base_instance["face_offset"],
                    "input_real": drive.real.tolist(),
                    "input_imag": drive.imag.tolist(),
                }
            )
        sweep_entries.append(
            {
                "binary_arrays": frequency_descriptors[sweep_index],
                "instances": instances,
            }
        )
    request["schema"] = DEPLOY_MICROPHONE_SWEEP_SCHEMA
    request["schema_version"] = 2
    request["geometry_key"] = "coupled-rom-microphone-sweep"
    request["frequencies_hz"] = [frequency for frequency, _index in frequency_pairs]
    request["rom_sweep"] = {
        **{key: copy.deepcopy(value) for key, value in base_rom.items() if key not in {"binary_arrays", "instances"}},
        "frequencies": sweep_entries,
    }
    request.pop("rom", None)
    request["provenance"]["frequency_count"] = len(frequency_pairs)
    request["provenance"]["rom_sweep_stage_cache_hit"] = int(stage_cache_hit)
    request["provenance"]["rom_sweep_stage_binary_bytes"] = binary_bytes
    request["provenance"]["rom_sweep_stage_binary_bytes_written"] = binary_bytes_written
    if status_callback is not None:
        status_callback(f"Serializing {len(frequency_pairs)}-frequency Level 3 ROM sweep")
    request_path.write_text(json.dumps(request, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return request_path, request


def prepare_deploy_microphone_sweep_request(
    payload: object,
    work_dir: str | Path,
    *,
    cache: DeploySolveCache | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Stage one geometry request containing every exported microphone frequency."""

    if not isinstance(payload, dict):
        raise ValueError("Deploy microphone sweep request must be an object.")
    package_path = Path(str(payload.get("packagePath", ""))).expanduser().resolve()
    package_data = cache.load_package(package_path) if cache is not None else _load_deploy_package_data(package_path)
    frequencies = sorted({float(value) for value in package_data.frequencies})
    if not frequencies:
        raise ValueError("Speaker package contains no frequencies for the microphone sweep.")
    raw_points = payload.get("observationPointsM")
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError("Deploy microphone sweep requires observationPointsM.")
    points_m = np.asarray(raw_points, dtype=np.float32)
    if points_m.ndim != 2 or points_m.shape[1] != 3 or not np.all(np.isfinite(points_m)):
        raise ValueError("Deploy microphone sweep observationPointsM must contain finite XYZ points.")
    if np.any(points_m[:, 1] < -GROUND_TOLERANCE_M):
        raise ValueError("Deploy microphone sweep observation points cannot be below the ground plane.")
    sources = [DeploySourcePlacement.from_payload(raw) for raw in payload.get("sources", [])]
    rigid_objects = [DeployRigidPlacement.from_payload(raw) for raw in payload.get("rigidObjects", [])]
    close_pair_override = payload.get("closePairQuadratureOrder")
    geometry_identity = {
        "package": package_data.fingerprint,
        "sources": [
            {
                "id": source.id,
                "position": [source.position_x_m, source.position_height_m, source.position_z_m],
                "rotation": [source.pitch_deg, source.yaw_deg, source.roll_deg],
            }
            for source in sources
        ],
        "rigid_objects": [
            {
                "id": rigid.id,
                "mesh": (
                    str(rigid.mesh_path),
                    int(rigid.mesh_path.stat().st_mtime_ns),
                    int(rigid.mesh_path.stat().st_size),
                ),
                "scale": rigid.scale_to_meters,
                "position": [rigid.position_x_m, rigid.position_height_m, rigid.position_z_m],
                "rotation": [rigid.pitch_deg, rigid.yaw_deg, rigid.roll_deg],
            }
            for rigid in rigid_objects
        ],
        "backend": str(payload.get("backend", "cuda")).strip().lower(),
        "burton_miller_assembly": payload.get("burtonMillerAssembly"),
        "quadrature_order": int(payload.get("quadratureOrder", 2)),
        "singular_order": int(payload.get("singularOrder", 3)),
        "close_pair_quadrature_order": int(
            CLOSE_PAIR_QUADRATURE_ORDER if close_pair_override is None else close_pair_override
        ),
    }
    geometry_key = hashlib.sha256(
        json.dumps(geometry_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    request_path = work_path / "request.json"
    cached_geometry = cache.sweep_geometries.get(geometry_key) if cache is not None else None
    if cached_geometry is None:
        first_payload = {
            **payload,
            "frequencyHz": frequencies[0],
            "includeComplexPressure": True,
            "solutionKey": "microphone-sweep-template",
        }
        request_path, request = prepare_deploy_solve_request(
            first_payload,
            work_dir,
            cache=cache,
            status_callback=status_callback,
        )
        if cache is not None:
            template = copy.deepcopy(request)
            for key in (
                "boundary_neumann",
                "reference_boundary_pressure",
                "observation_points_m",
                "observation_shape",
                "observation_sample_indices",
            ):
                template.pop(key, None)
            mesh_text = Path(str(request["mesh_file"])).read_text(encoding="utf-8")
            cache.sweep_geometries.clear()
            cache.sweep_geometries[geometry_key] = (template, mesh_text)
    else:
        if status_callback is not None:
            status_callback("Reusing prepared scene geometry")
        template, mesh_text = cached_geometry
        request = copy.deepcopy(template)
        staged_mesh = work_path / "exterior.msh"
        staged_mesh.write_text(mesh_text, encoding="utf-8")
        request["mesh_file"] = str(staged_mesh)
    request["observation_points_m"] = points_m.tolist()
    request["observation_shape"] = [1, int(points_m.shape[0])]
    request["observation_sample_indices"] = list(range(int(points_m.shape[0])))
    source_face_count = int(package_data.triangles.shape[0])
    source_vertex_count = int(package_data.points.shape[0])
    rigid_components = request["boundary_components"][len(sources) :]
    rigid_face_count = sum(int(component["face_count"]) for component in rigid_components)
    rigid_vertex_count = sum(int(component["vertex_count"]) for component in rigid_components)
    q_real_rows: list[list[float]] = []
    q_imag_rows: list[list[float]] = []
    pressure_real_rows: list[list[float]] = []
    pressure_imag_rows: list[list[float]] = []
    excitation_indices = _logical_excitation_indices(
        package_data.manifest,
        package_data.normal.shape[1],
    )
    if status_callback is not None:
        status_callback(f"Encoding {len(frequencies)} frequency traces")
    for frequency_hz in frequencies:
        frequency_index = int(np.argmin(np.abs(package_data.frequencies - frequency_hz)))
        q_parts: list[np.ndarray] = []
        pressure_parts: list[np.ndarray] = []
        logical_normal = _combined_excitation_trace(
            package_data.normal,
            frequency_index,
            excitation_indices,
        )
        logical_pressure = _combined_excitation_trace(
            package_data.pressure,
            frequency_index,
            excitation_indices,
        )
        for source in sources:
            phase = 2.0 * math.pi * frequency_hz * source.delay_ms / 1000.0
            gain = (0.0 if source.muted else source.polarity * 10.0 ** (source.level_db / 20.0)) * np.exp(1j * phase)
            q_parts.append(np.asarray(logical_normal * gain, dtype=np.complex64))
            pressure_parts.append(np.asarray(logical_pressure * gain, dtype=np.complex64))
        if rigid_face_count:
            q_parts.append(np.zeros(rigid_face_count, dtype=np.complex64))
        if rigid_vertex_count:
            pressure_parts.append(np.zeros(rigid_vertex_count, dtype=np.complex64))
        q = np.concatenate(q_parts) if q_parts else np.empty(0, dtype=np.complex64)
        reference = np.concatenate(pressure_parts) if pressure_parts else np.empty(0, dtype=np.complex64)
        if q.size != source_face_count * len(sources) + rigid_face_count:
            raise ValueError("Microphone sweep Neumann traces do not match the staged mesh.")
        if reference.size != source_vertex_count * len(sources) + rigid_vertex_count:
            raise ValueError("Microphone sweep reference traces do not match the staged mesh.")
        q_real_rows.append(q.real.tolist())
        q_imag_rows.append(q.imag.tolist())
        pressure_real_rows.append(reference.real.tolist())
        pressure_imag_rows.append(reference.imag.tolist())

    request["schema"] = DEPLOY_MICROPHONE_SWEEP_SCHEMA
    request["schema_version"] = 1
    request["geometry_key"] = geometry_key
    request["frequencies_hz"] = frequencies
    request["boundary_neumann_sweep"] = {"real": q_real_rows, "imag": q_imag_rows}
    request["reference_boundary_pressure_sweep"] = {
        "real": pressure_real_rows,
        "imag": pressure_imag_rows,
    }
    request.pop("boundary_neumann", None)
    request.pop("reference_boundary_pressure", None)
    request["provenance"]["frequency_count"] = len(frequencies)
    if status_callback is not None:
        status_callback("Serializing multi-frequency BEAT request")
    request_path.write_text(json.dumps(request, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return request_path, request


def prepare_deploy_field_request(payload: object, work_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    """Stage a plane-only request that reuses the worker's current boundary solution."""

    if not isinstance(payload, dict):
        raise ValueError("Deploy field request must be an object.")
    solution_key = str(payload.get("solutionKey", "")).strip()
    if not solution_key:
        raise ValueError("Deploy field reuse requires a boundary solution key.")
    observation = DeployObservationPlane.from_payload(payload.get("observation"))
    points_m, observation_sample_indices = observation.points()
    if points_m.shape[0] == 0:
        raise ValueError("Deploy observation plane has no sampling points on or above the ground plane.")
    backend = str(payload.get("backend", "cuda")).strip().lower()
    request: dict[str, Any] = {
        "schema": DEPLOY_FIELD_SCHEMA,
        "schema_version": 1,
        "beat_engine_backend": backend,
        "solution_key": solution_key,
        "include_complex_pressure": bool(payload.get("includeComplexPressure", False)),
    }
    if backend == "cuda":
        request["observation_plane"] = observation.wire()
    else:
        request["observation_points_m"] = points_m.tolist()
        request["observation_shape"] = [observation.rows, observation.columns]
        request["observation_sample_indices"] = observation_sample_indices.tolist()
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    request_path = work_path / "field-request.json"
    request_path.write_text(json.dumps(request, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return request_path, request
