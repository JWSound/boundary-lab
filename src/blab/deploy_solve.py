"""Qt-free preparation for Boundary Lab Deploy Level 2 solves."""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from blab.deploy_geometry import minimum_surface_distance, surface_face_pairs_within, transform_package_points
from blab.speaker_package import validate_speaker_package

DEPLOY_SOLVE_SCHEMA = "boundary_lab_deploy_solve"
DEPLOY_SOLVE_SCHEMA_VERSION = 2
DEPLOY_FIELD_SCHEMA = "boundary_lab_deploy_field"
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


@dataclass
class DeploySolveCache:
    packages: dict[tuple[str, int, int], DeployPackageData] = field(default_factory=dict)
    ground_image_pairs: dict[tuple[tuple[str, int, int], float, float, float], list[Any]] = field(
        default_factory=dict
    )

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
        return package


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
    with np.load(io.BytesIO(fixed_bytes), allow_pickle=False) as fixed:
        triangles = np.asarray(fixed["triangles"], dtype=np.int64)
        points = np.asarray(fixed["points_m"], dtype=np.float64)
        pressure = np.asarray(fixed["pressure_pa"])
        normal = np.asarray(fixed["normal_derivative_pa_per_m"])
    frequencies = np.asarray(manifest.get("frequencies_hz", ()), dtype=np.float64)
    return DeployPackageData(
        package_path,
        package_fingerprint,
        manifest,
        frequencies,
        triangles,
        points,
        pressure,
        normal,
        geometry_bytes,
    )


def prepare_deploy_solve_request(
    payload: object,
    work_dir: str | Path,
    *,
    cache: DeploySolveCache | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Validate a renderer request and stage fixed-source instances for BEAT."""

    if not isinstance(payload, dict):
        raise ValueError("Deploy solve request must be an object.")
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
        CLOSE_PAIR_QUADRATURE_ORDER
        if close_pair_quadrature_override is None
        else close_pair_quadrature_override
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
    observation = DeployObservationPlane.from_payload(payload.get("observation"))
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
    geometry_bytes = package_data.geometry_bytes
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

    transformed_sources = [
        transform_package_points(
            points,
            position_x_m=source.position_x_m,
            position_height_m=source.position_height_m,
            position_z_m=source.position_z_m,
            pitch_deg=source.pitch_deg,
            roll_deg=source.roll_deg,
            yaw_deg=source.yaw_deg,
        )
        for source in sources
    ]
    for source, transformed in zip(sources, transformed_sources, strict=True):
        minimum_y = float(np.min(transformed[:, 1]))
        if minimum_y < -GROUND_TOLERANCE_M:
            raise ValueError(
                f"Deploy source {source.id!r} extends {abs(minimum_y):.6f} m below the ground plane."
            )

    proximity_pairs: list[dict[str, Any]] = []
    close_face_pairs: list[list[int]] = []
    minimum_surface_distance_m: float | None = None
    for first_index in range(len(sources)):
        for second_index in range(first_index + 1, len(sources)):
            distance = minimum_surface_distance(
                transformed_sources[first_index],
                triangles,
                transformed_sources[second_index],
                triangles,
            )
            minimum_surface_distance_m = (
                distance.distance_m
                if minimum_surface_distance_m is None
                else min(minimum_surface_distance_m, distance.distance_m)
            )
            if distance.distance_m + GROUND_TOLERANCE_M < SOURCE_SURFACE_PADDING_M:
                raise ValueError(
                    f"Deploy sources {sources[first_index].id!r} and {sources[second_index].id!r} have "
                    f"{distance.distance_m * 1000.0:.3f} mm surface spacing; at least "
                    f"{SOURCE_SURFACE_PADDING_M * 1000.0:.1f} mm is required."
                )
            pair = {
                "source_a": sources[first_index].id,
                "source_b": sources[second_index].id,
                "distance_m": distance.distance_m,
                "face_a": distance.face_a,
                "face_b": distance.face_b,
                "close": distance.distance_m <= CLOSE_PAIR_DISTANCE_M,
            }
            if pair["close"]:
                face_pairs = surface_face_pairs_within(
                    transformed_sources[first_index],
                    triangles,
                    transformed_sources[second_index],
                    triangles,
                    CLOSE_PAIR_DISTANCE_M,
                    exact=False,
                )
                first_offset = first_index * triangles.shape[0]
                second_offset = second_index * triangles.shape[0]
                for face_pair in face_pairs:
                    first_face = first_offset + face_pair.face_a
                    second_face = second_offset + face_pair.face_b
                    correction_order = close_pair_quadrature_order if close_pair_quadrature_override is not None else (
                        8 if face_pair.distance_m <= 0.015 else 6 if face_pair.distance_m <= 0.03 else 4
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
    ground_image_face_pairs: list[list[int]] = []
    singular_tolerance_squared = GROUND_IMAGE_SINGULAR_TOLERANCE_M**2
    reflected_sources = []
    for transformed in transformed_sources:
        reflected = transformed.copy()
        reflected[:, 1] *= -1.0
        reflected_sources.append(reflected)

    def non_singular_ground_pairs(test_points: np.ndarray, trial_points: np.ndarray) -> list[Any]:
        test_faces = test_points[triangles]
        trial_faces = trial_points[triangles]
        filtered = []
        for face_pair in surface_face_pairs_within(
            test_points,
            triangles,
            trial_points,
            triangles,
            CLOSE_PAIR_DISTANCE_M,
            exact=False,
        ):
            vertex_deltas = (
                test_faces[face_pair.face_a, :, np.newaxis, :]
                - trial_faces[face_pair.face_b, np.newaxis, :, :]
            )
            if np.any(np.sum(vertex_deltas * vertex_deltas, axis=2) <= singular_tolerance_squared):
                continue
            filtered.append(face_pair)
        return filtered

    ground_pair_cache = cache.ground_image_pairs if cache is not None else {}
    ground_pair_sets: list[tuple[int, int, list[Any]]] = []
    for source_index, source in enumerate(sources):
        self_key = (
            package_data.fingerprint,
            source.position_height_m,
            source.pitch_deg,
            source.roll_deg,
        )
        self_pairs = ground_pair_cache.get(self_key)
        if self_pairs is None:
            canonical_points = transform_package_points(
                points,
                position_x_m=0.0,
                position_height_m=source.position_height_m,
                position_z_m=0.0,
                pitch_deg=source.pitch_deg,
                roll_deg=source.roll_deg,
                yaw_deg=0.0,
            )
            canonical_reflected = canonical_points.copy()
            canonical_reflected[:, 1] *= -1.0
            self_pairs = non_singular_ground_pairs(canonical_points, canonical_reflected)
            ground_pair_cache[self_key] = self_pairs
        ground_pair_sets.append((source_index, source_index, self_pairs))

    close_distance_squared = CLOSE_PAIR_DISTANCE_M**2
    for test_index, test_points in enumerate(transformed_sources):
        test_minimum = np.min(test_points, axis=0)
        test_maximum = np.max(test_points, axis=0)
        for trial_index, trial_points in enumerate(reflected_sources):
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
                (test_index, trial_index, non_singular_ground_pairs(test_points, trial_points))
            )

    face_count_per_source = triangles.shape[0]
    for test_source, trial_source, face_pairs in ground_pair_sets:
        for face_pair in face_pairs:
            correction_order = close_pair_quadrature_order if close_pair_quadrature_override is not None else (
                8 if face_pair.distance_m <= 0.015 else 6 if face_pair.distance_m <= 0.03 else 4
            )
            ground_image_face_pairs.append(
                [
                    test_source * face_count_per_source + face_pair.face_a,
                    trial_source * face_count_per_source + face_pair.face_b,
                    correction_order,
                ]
            )
    ground_image_face_pairs.sort(key=lambda pair: (pair[0], pair[1]))

    q_parts: list[np.ndarray] = []
    reference_parts: list[np.ndarray] = []
    for source in sources:
        phase = 2.0 * math.pi * frequency_hz * source.delay_ms / 1000.0
        gain = source.polarity * 10.0 ** (source.level_db / 20.0) * np.exp(1j * phase)
        q_parts.append(np.asarray(normal[frequency_index, 0] * gain, dtype=np.complex64))
        reference_parts.append(np.asarray(pressure[frequency_index, 0] * gain, dtype=np.complex64))
    q_neumann = np.concatenate(q_parts)
    reference_pressure = np.concatenate(reference_parts)
    if not np.all(np.isfinite(q_neumann)) or not np.all(np.isfinite(reference_pressure)):
        raise ValueError("Fixed-source boundary traces contain non-finite values.")

    staged_mesh = work_path / "exterior.msh"
    staged_mesh.write_bytes(geometry_bytes)
    points_m, observation_sample_indices = observation.points()
    if points_m.shape[0] == 0:
        raise ValueError("Deploy observation plane has no sampling points on or above the ground plane.")
    medium = manifest.get("medium", {})
    backend = str(payload.get("backend", "cuda")).strip().lower()
    burton_miller_assembly = str(
        payload.get(
            "burtonMillerAssembly",
            "direct_system" if backend == "cuda" else "operator_matrices",
        )
    ).strip().lower()
    if burton_miller_assembly not in {"direct_system", "operator_matrices"}:
        raise ValueError(
            "Deploy burtonMillerAssembly must be 'direct_system' or 'operator_matrices'."
        )
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
        "source_transforms": [
            {
                "id": source.id,
                "position_m": [source.position_x_m, source.position_height_m, source.position_z_m],
                "pitch_deg": source.pitch_deg,
                "yaw_deg": source.yaw_deg,
                "roll_deg": source.roll_deg,
            }
            for source in sources
        ],
        "boundary_neumann": {
            "real": q_neumann.real.tolist(),
            "imag": q_neumann.imag.tolist(),
        },
        "reference_boundary_pressure": {
            "real": reference_pressure.real.tolist(),
            "imag": reference_pressure.imag.tolist(),
        },
        "observation_points_m": points_m.tolist(),
        "observation_shape": [observation.rows, observation.columns],
        "observation_sample_indices": observation_sample_indices.tolist(),
        "observation_plane": observation.wire(),
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
            "source_ids": source_ids,
            "package_node_count": int(points.shape[0]),
            "package_face_count": int(triangles.shape[0]),
            "node_count": int(points.shape[0] * len(sources)),
            "face_count": int(triangles.shape[0] * len(sources)),
            "excitation_index": 0,
            "exterior_domain": "rigid_y0_half_space",
        },
    }
    request_path = work_path / "request.json"
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
