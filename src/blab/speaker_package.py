"""Versioned Boundary Lab speaker packages for array-simulation tools."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable

import numpy as np

from blab.config import normalize_symmetry
from blab.physical_model import AcousticRegionKind, PhysicalSolveKind
from blab.solve_results import (
    BEM_BOUNDARY_DOMAIN_ID,
    BEM_BOUNDARY_NEUMANN_ID,
    BEM_BOUNDARY_PRESSURE_ID,
    SPHERE_DOMAIN_ID,
    SPHERE_PRESSURE_ID,
    ResultDomain,
    SolvedSystem,
    SolvedSystemBuilder,
    SolveProvenance,
    bem_boundary_result_domain,
)
from blab.solvers.coupled_backend import PhysicalSystemProductionBackend, validate_solve_plan
from blab.symmetry import snap_points_to_symmetry_planes
from blab.system_contract import OutputRequest, compiled_system_to_dict
from blab.system_solve import SystemUiSolveRequest, canonicalize_observation_result

SPEAKER_PACKAGE_SCHEMA = "boundary-lab-speaker-package"
SPEAKER_PACKAGE_SCHEMA_VERSION = 1
SPEAKER_ROM_QUANTITIES = (
    ("speaker:rom:k", "speaker_rom_k", "k"),
    ("speaker:rom:c", "speaker_rom_c", "c"),
    ("speaker:rom:d", "speaker_rom_d", "d"),
    ("speaker:rom:b", "speaker_rom_b", "b"),
    ("speaker:rom:e", "speaker_rom_e", "e"),
    ("speaker:rom:velocity", "speaker_rom_velocity", "velocity"),
    ("speaker:rom:current", "speaker_rom_current", "current"),
    ("speaker:rom:velocity-drive", "speaker_rom_velocity_drive", "velocity_drive"),
    ("speaker:rom:current-drive", "speaker_rom_current_drive", "current_drive"),
)

# A proper right-handed rotation about X by -90 degrees. Boundary Lab +Z
# becomes package +Y; source +Y becomes package -Z.
SOURCE_TO_PACKAGE_ROTATION = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    dtype=np.float64,
)


class SpeakerPackageFidelity(IntEnum):
    PATTERN = 1
    FIXED_SOURCES = 2
    COUPLED = 3

    @classmethod
    def parse(cls, value: int | str | SpeakerPackageFidelity) -> SpeakerPackageFidelity:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_")
            names = {
                "1": cls.PATTERN,
                "pattern": cls.PATTERN,
                "pattern_superposition": cls.PATTERN,
                "2": cls.FIXED_SOURCES,
                "fixed": cls.FIXED_SOURCES,
                "fixed_sources": cls.FIXED_SOURCES,
                "3": cls.COUPLED,
                "coupled": cls.COUPLED,
                "dynamic": cls.COUPLED,
                "condensed": cls.COUPLED,
            }
            if normalized in names:
                return names[normalized]
        try:
            return cls(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError("Speaker package fidelity must be 'pattern', 'fixed', or 'coupled'.") from exc

    @property
    def cli_name(self) -> str:
        if self == self.PATTERN:
            return "pattern"
        return "fixed" if self == self.FIXED_SOURCES else "coupled"


class SpeakerPackageCoupledRepresentation(str, Enum):
    """On-disk representation of a Level-3 interior model."""

    EXACT_SYSTEM = "exact_system"
    PARITY_ROM = "parity_rom"

    @classmethod
    def parse(
        cls,
        value: str | SpeakerPackageCoupledRepresentation,
    ) -> SpeakerPackageCoupledRepresentation:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        aliases = {
            "exact": cls.EXACT_SYSTEM,
            "exact_system": cls.EXACT_SYSTEM,
            "sparse": cls.EXACT_SYSTEM,
            "rom": cls.PARITY_ROM,
            "parity_rom": cls.PARITY_ROM,
            "parity_petrov_galerkin": cls.PARITY_ROM,
            "petrov_galerkin": cls.PARITY_ROM,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError(
                "Coupled speaker representation must be 'parity-rom' or the developer-only 'exact-system'."
            ) from exc


@dataclass(frozen=True)
class SpeakerPackageConfig:
    output_path: Path
    name: str
    fidelity: SpeakerPackageFidelity = SpeakerPackageFidelity.PATTERN
    coupled_representation: SpeakerPackageCoupledRepresentation = SpeakerPackageCoupledRepresentation.PARITY_ROM

    def normalized(self) -> SpeakerPackageConfig:
        path = Path(self.output_path)
        if path.suffix.lower() != ".blabsp":
            path = path.with_suffix(".blabsp")
        name = str(self.name).strip()
        if not name:
            raise ValueError("Speaker package name must not be empty.")
        return SpeakerPackageConfig(
            path,
            name,
            SpeakerPackageFidelity.parse(self.fidelity),
            SpeakerPackageCoupledRepresentation.parse(self.coupled_representation),
        )


@dataclass(frozen=True)
class SpeakerPackageIssue:
    code: str
    message: str


@dataclass(frozen=True)
class SpeakerPackageExportResult:
    path: Path
    fidelity: SpeakerPackageFidelity
    frequency_count: int
    excitation_count: int


@dataclass(frozen=True)
class ExpandedBemBoundary:
    """Full-domain BEM geometry and trace provenance reconstructed from symmetry."""

    points_m: np.ndarray
    triangles: np.ndarray
    pressure_pa: np.ndarray
    normal_derivative_pa_per_m: np.ndarray
    source_node_index: np.ndarray
    source_face_index: np.ndarray
    face_symmetry_image_index: np.ndarray
    symmetry: str
    symmetry_images: tuple[dict[str, Any], ...]


def rotate_source_points(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Coordinates must have shape (point, 3).")
    return values @ SOURCE_TO_PACKAGE_ROTATION.T


def prepare_speaker_package_solve(
    prepared: SystemUiSolveRequest,
    *,
    fidelity: int | str | SpeakerPackageFidelity,
    coupled_representation: str | SpeakerPackageCoupledRepresentation = SpeakerPackageCoupledRepresentation.PARITY_ROM,
    sphere_point_count: int,
    sphere_radius_m: float,
    speaker_rom_rank: int = 32,
    speaker_rom_training_count: int = 96,
    speaker_rom_validation_count: int = 24,
) -> SystemUiSolveRequest:
    """Add the physical outputs required by a speaker package solve."""

    level = SpeakerPackageFidelity.parse(fidelity)
    representation = SpeakerPackageCoupledRepresentation.parse(coupled_representation)
    if sphere_point_count <= 0:
        raise ValueError("Speaker package sphere point count must be greater than zero.")
    if not np.isfinite(sphere_radius_m) or sphere_radius_m <= 0.0:
        raise ValueError("Speaker package sphere radius must be finite and greater than zero.")

    request = prepared.request
    outputs = list(request.outputs)
    domains = list(prepared.result_domains)
    domain_ids = {item.id for item in domains}
    sphere_present = any(
        str(domain.get("quantity_id", "")) == SPHERE_PRESSURE_ID
        for output in outputs
        if output.id == "ui:exterior-pressure"
        for domain in output.options.get("observation_domains", ())
    )

    if not sphere_present:
        points, sphere_metadata = _fibonacci_sphere_points(sphere_point_count, sphere_radius_m)
        compact_index = next((index for index, item in enumerate(outputs) if item.id == "ui:exterior-pressure"), None)
        if compact_index is None:
            compact_index = len(outputs)
            outputs.append(OutputRequest(id="ui:exterior-pressure", quantity="exterior_pressure", options={}))
        compact = outputs[compact_index]
        options = dict(compact.options)
        combined_points = list(options.get("points_m", ()))
        observation_domains = list(options.get("observation_domains", ()))
        offset = len(combined_points)
        combined_points.extend(points.tolist())
        observation_domains.append(
            {
                "id": SPHERE_DOMAIN_ID,
                "quantity_id": SPHERE_PRESSURE_ID,
                "offset": offset,
                "count": len(points),
            }
        )
        options.update({"points_m": combined_points, "observation_domains": observation_domains})
        outputs[compact_index] = replace(compact, options=options)
        if SPHERE_DOMAIN_ID not in domain_ids:
            domains.append(
                ResultDomain(
                    id=SPHERE_DOMAIN_ID,
                    kind="spherical_observation",
                    dimensions=("observation",),
                    coordinates={"points_m": points, **sphere_metadata},
                    metadata={"coordinate_frame": "boundary_lab_project"},
                )
            )

    if level >= SpeakerPackageFidelity.FIXED_SOURCES:
        symmetry = normalize_symmetry(request.solver_options.get("symmetry", "off"))
        existing_output_ids = {item.id for item in outputs}
        for identifier, quantity in (
            (BEM_BOUNDARY_PRESSURE_ID, "bem_boundary_pressure"),
            (BEM_BOUNDARY_NEUMANN_ID, "bem_boundary_neumann"),
        ):
            if identifier not in existing_output_ids:
                outputs.append(
                    OutputRequest(
                        id=identifier,
                        quantity=quantity,
                        target_ids=(BEM_BOUNDARY_DOMAIN_ID,),
                    )
                )
        if BEM_BOUNDARY_DOMAIN_ID not in domain_ids:
            domains.append(bem_boundary_result_domain(request.compiled_system, symmetry=symmetry))

    if level >= SpeakerPackageFidelity.COUPLED:
        if prepared.solve_kind != PhysicalSolveKind.COUPLED_BEM_FEM:
            raise ValueError("Level-3 speaker packages require a coupled FEM-BEM physical system.")
        symmetry = normalize_symmetry(request.solver_options.get("symmetry", "off"))
        if symmetry != "off":
            raise ValueError(
                "Level-3 speaker package preparation requires a temporary full-domain system with symmetry off."
            )
        if representation == SpeakerPackageCoupledRepresentation.PARITY_ROM:
            if speaker_rom_rank <= 0:
                raise ValueError("Speaker ROM rank must be positive.")
            if speaker_rom_training_count < speaker_rom_rank:
                raise ValueError("Speaker ROM training count must be at least its rank.")
            if speaker_rom_validation_count <= 0:
                raise ValueError("Speaker ROM validation count must be positive.")
            existing_output_ids = {item.id for item in outputs}
            for identifier, quantity, _archive_name in SPEAKER_ROM_QUANTITIES:
                if identifier not in existing_output_ids:
                    outputs.append(OutputRequest(id=identifier, quantity=quantity))

    solver_options = dict(request.solver_options)
    if level >= SpeakerPackageFidelity.COUPLED:
        solver_options.update(
            {
                "validation_diagnostics": False,
                "static_condensation": True,
                "symmetry": "off",
            }
        )
        if representation == SpeakerPackageCoupledRepresentation.PARITY_ROM:
            expansion = request.compiled_system.metadata.get("speaker_export_symmetry_expansion", {})
            source_symmetry = (
                expansion.get("source_symmetry", "off")
                if isinstance(expansion, dict)
                else "off"
            )
            solver_options["speaker_rom"] = {
                "rank_per_sector": int(speaker_rom_rank),
                "training_count_per_sector": int(speaker_rom_training_count),
                "validation_count_per_sector": int(speaker_rom_validation_count),
                "symmetry": normalize_symmetry(source_symmetry),
            }
    updated_request = replace(request, outputs=tuple(outputs), solver_options=solver_options)
    validate_solve_plan(updated_request)
    return replace(
        prepared,
        request=updated_request,
        result_domains=tuple(domains),
        backend_id=prepared.backend_id,
    )


def solve_speaker_package_system(
    prepared: SystemUiSolveRequest,
    *,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    julia_executable: str = "julia",
    julia_threads: str | int | None = None,
) -> SolvedSystem:
    """Run an export-prepared solve and return its complete canonical snapshot."""

    emit = event_callback or (lambda _event: None)
    request = prepared.request
    builder = SolvedSystemBuilder(
        frequencies_hz=request.frequencies_hz,
        excitation_ids=request.excitation_port_ids,
        provenance=SolveProvenance(
            backend_id=prepared.backend_id,
            solve_kind=prepared.solve_kind.value,
            solver_options=dict(request.solver_options),
        ),
        domains=prepared.result_domains,
        compiled_system=request.compiled_system,
    )
    backend = PhysicalSystemProductionBackend(
        bem_backend=prepared.backend_id.removeprefix("beat_"),
        julia_executable=julia_executable,
        julia_threads=julia_threads,
    )
    session = backend.create_system_session(request)
    emit({"event": "started", "frequency_count": len(request.frequencies_hz)})
    try:
        for raw_result in session.solve_stream():
            result = canonicalize_observation_result(prepared, raw_result)
            index = builder.add(result)
            emit(
                {
                    "event": "frequency_completed",
                    "index": index,
                    "freq_hz": float(result.freq_hz),
                    "solved_count": builder.solved_count,
                    "frequency_count": len(request.frequencies_hz),
                }
            )
    except Exception:
        session.stop()
        raise
    solved = builder.finalize(status="completed")
    if not solved.complete:
        raise RuntimeError("Speaker package solve did not complete every requested frequency.")
    return solved


def speaker_package_issues(
    solved: SolvedSystem | None,
    fidelity: int | str | SpeakerPackageFidelity,
    coupled_representation: str | SpeakerPackageCoupledRepresentation = SpeakerPackageCoupledRepresentation.PARITY_ROM,
) -> tuple[SpeakerPackageIssue, ...]:
    level = SpeakerPackageFidelity.parse(fidelity)
    representation = SpeakerPackageCoupledRepresentation.parse(coupled_representation)
    issues: list[SpeakerPackageIssue] = []
    if solved is None:
        return (SpeakerPackageIssue("missing_solved_system", "No solved system is available."),)
    if not solved.complete:
        issues.append(SpeakerPackageIssue("incomplete_frequency", "Every requested frequency must be complete."))

    sphere_domain = solved.domains.get(SPHERE_DOMAIN_ID)
    sphere_pressure = solved.quantities.get(SPHERE_PRESSURE_ID)
    if sphere_domain is None:
        issues.append(SpeakerPackageIssue("missing_sphere_domain", "Spherical sampling geometry was not retained."))
    if sphere_pressure is None:
        issues.append(SpeakerPackageIssue("missing_sphere_pressure", "Complex spherical pressure was not retained."))
    if sphere_domain is not None and sphere_pressure is not None:
        points = np.asarray(sphere_domain.coordinates.get("points_m"))
        values = np.asarray(sphere_pressure.values)
        expected = (solved.frequencies_hz.size, len(solved.excitation_ids), points.shape[0] if points.ndim == 2 else -1)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or sphere_pressure.dimensions != ("frequency", "excitation", "observation")
            or values.shape != expected
            or not np.iscomplexobj(values)
        ):
            issues.append(
                SpeakerPackageIssue("invalid_sphere_data", "Spherical pressure geometry or shape is invalid.")
            )
        elif not np.all(np.asarray(sphere_pressure.available_frequency_mask, dtype=bool)):
            issues.append(SpeakerPackageIssue("incomplete_sphere_pressure", "Spherical pressure is incomplete."))

    if level >= SpeakerPackageFidelity.FIXED_SOURCES:
        bem_domain = solved.domains.get(BEM_BOUNDARY_DOMAIN_ID)
        pressure = solved.quantities.get(BEM_BOUNDARY_PRESSURE_ID)
        neumann = solved.quantities.get(BEM_BOUNDARY_NEUMANN_ID)
        if bem_domain is None:
            issues.append(SpeakerPackageIssue("missing_bem_domain", "BEM boundary geometry was not retained."))
        if pressure is None:
            issues.append(SpeakerPackageIssue("missing_bem_pressure", "BEM boundary pressure was not retained."))
        if neumann is None:
            issues.append(SpeakerPackageIssue("missing_bem_neumann", "BEM normal derivative was not retained."))
        if bem_domain is not None and pressure is not None and neumann is not None:
            points = np.asarray(bem_domain.coordinates.get("points_m"))
            triangles = np.asarray(bem_domain.topology.get("triangles"))
            expected_pressure = (solved.frequencies_hz.size, len(solved.excitation_ids), points.shape[0])
            expected_neumann = (solved.frequencies_hz.size, len(solved.excitation_ids), triangles.shape[0])
            if (
                points.ndim != 2
                or points.shape[1] != 3
                or triangles.ndim != 2
                or triangles.shape[1] != 3
                or pressure.dimensions != ("frequency", "excitation", "bem_node")
                or neumann.dimensions != ("frequency", "excitation", "bem_face")
                or np.asarray(pressure.values).shape != expected_pressure
                or np.asarray(neumann.values).shape != expected_neumann
            ):
                issues.append(SpeakerPackageIssue("invalid_bem_data", "BEM geometry or trace shapes are invalid."))
            elif not np.all(np.asarray(pressure.available_frequency_mask, dtype=bool)) or not np.all(
                np.asarray(neumann.available_frequency_mask, dtype=bool)
            ):
                issues.append(SpeakerPackageIssue("incomplete_bem_traces", "BEM boundary traces are incomplete."))
    if level >= SpeakerPackageFidelity.COUPLED and representation == SpeakerPackageCoupledRepresentation.EXACT_SYSTEM:
        system = solved.compiled_system
        if system is None:
            issues.append(SpeakerPackageIssue("missing_compiled_system", "Exact Level-3 export requires a compiled system."))
        elif not any(region.kind == AcousticRegionKind.BOUNDED_AIR for region in system.regions):
            issues.append(
                SpeakerPackageIssue("missing_bounded_region", "Exact Level-3 export requires a bounded FEM region.")
            )
        elif any(not Path(mesh.file).is_file() for mesh in system.meshes):
            issues.append(
                SpeakerPackageIssue("missing_system_mesh", "Exact Level-3 export requires every compiled mesh file.")
            )
    if level >= SpeakerPackageFidelity.COUPLED and representation == SpeakerPackageCoupledRepresentation.PARITY_ROM:
        matrices = {
            identifier: solved.quantities.get(identifier)
            for identifier, _quantity, _archive_name in SPEAKER_ROM_QUANTITIES
        }
        for identifier, matrix in matrices.items():
            if matrix is None:
                issues.append(
                    SpeakerPackageIssue(
                        "missing_speaker_rom_matrix",
                        f"Parity speaker ROM quantity {identifier!r} was not retained.",
                    )
                )
        if all(matrix is not None for matrix in matrices.values()):
            values = {
                archive_name: np.asarray(matrices[identifier].values)
                for identifier, _quantity, archive_name in SPEAKER_ROM_QUANTITIES
            }
            metadata = matrices["speaker:rom:k"].metadata
            rank = int(metadata.get("rank_per_sector", 0))
            node_orbits = metadata.get("node_orbits", ())
            face_orbits = metadata.get("face_orbits", ())
            sector_signs = metadata.get("sector_signs", ())
            sector_names = metadata.get("sector_names", ())
            sector_count = len(sector_signs)
            symmetry_mode = str(metadata.get("symmetry_mode", "xy"))
            expected_image_count = {"off": 1, "x": 2, "xy": 4}.get(symmetry_mode, 0)
            image_count = int(metadata.get("image_count", expected_image_count))
            frequency_count = solved.frequencies_hz.size
            input_count = len(solved.excitation_ids)
            transducer_count = int(metadata.get("transducer_count", -1))
            expected_shapes = {
                "k": (frequency_count, sector_count, rank, rank),
                "c": (frequency_count, sector_count, rank, len(node_orbits)),
                "d": (frequency_count, sector_count, len(face_orbits), rank),
                "b": (frequency_count, sector_count, rank, input_count),
                "e": (frequency_count, sector_count, len(face_orbits), input_count),
                "velocity": (frequency_count, sector_count, transducer_count, rank),
                "current": (frequency_count, sector_count, transducer_count, rank),
                "velocity_drive": (frequency_count, sector_count, transducer_count, input_count),
                "current_drive": (frequency_count, sector_count, transducer_count, input_count),
            }
            if (
                rank <= 0
                or sector_count != expected_image_count
                or len(sector_names) != sector_count
                or image_count != expected_image_count
                or not node_orbits
                or not face_orbits
                or any(len(orbit) != image_count for orbit in (*node_orbits, *face_orbits))
                or any(len(signs) != 2 for signs in sector_signs)
                or any(values[name].shape != shape for name, shape in expected_shapes.items())
                or not all(np.iscomplexobj(array) for array in values.values())
            ):
                issues.append(
                    SpeakerPackageIssue(
                        "invalid_speaker_rom",
                        "Parity speaker ROM shapes, metadata, or complex dtypes are invalid.",
                    )
                )
            elif any(
                not np.all(np.asarray(matrices[identifier].available_frequency_mask, dtype=bool))
                for identifier, _quantity, _archive_name in SPEAKER_ROM_QUANTITIES
            ):
                issues.append(
                    SpeakerPackageIssue(
                        "incomplete_speaker_rom",
                        "Parity speaker ROM quantities are incomplete.",
                    )
                )
    return tuple(issues)


def export_speaker_package(
    solved: SolvedSystem,
    config: SpeakerPackageConfig,
) -> SpeakerPackageExportResult:
    normalized = config.normalized()
    issues = speaker_package_issues(solved, normalized.fidelity, normalized.coupled_representation)
    if issues:
        raise ValueError("Speaker package is not exportable: " + "; ".join(issue.message for issue in issues))

    destination = normalized.output_path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        _write_archive(temp_path, solved, normalized)
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    return SpeakerPackageExportResult(
        path=destination,
        fidelity=normalized.fidelity,
        frequency_count=int(solved.frequencies_hz.size),
        excitation_count=len(solved.excitation_ids),
    )


def validate_speaker_package(path: str | Path) -> dict[str, Any]:
    """Validate archive structure and checksums and return its manifest."""

    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = set(archive.namelist())
            if "manifest.json" not in members or "checksums.json" not in members:
                raise ValueError("Speaker package is missing manifest.json or checksums.json.")
            manifest = json.loads(archive.read("manifest.json"))
            checksums = json.loads(archive.read("checksums.json"))
            if manifest.get("schema") != SPEAKER_PACKAGE_SCHEMA:
                raise ValueError("Unsupported speaker package schema.")
            if int(manifest.get("schema_version", 0)) != SPEAKER_PACKAGE_SCHEMA_VERSION:
                raise ValueError("Unsupported speaker package schema version.")
            for member, expected in checksums.items():
                if member not in members:
                    raise ValueError(f"Speaker package is missing referenced member {member!r}.")
                actual = hashlib.sha256(archive.read(member)).hexdigest()
                if actual != expected:
                    raise ValueError(f"Speaker package member {member!r} failed its checksum.")
            return manifest
    except zipfile.BadZipFile as exc:
        raise ValueError("File is not a valid .blabsp archive.") from exc


def _write_archive(path: Path, solved: SolvedSystem, config: SpeakerPackageConfig) -> None:
    members, manifest = _archive_members(solved, config)
    manifest_bytes = _json_bytes(manifest)
    members["manifest.json"] = manifest_bytes
    checksums = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(members.items())}
    members["checksums.json"] = _json_bytes(checksums)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for name, data in sorted(members.items()):
            archive.writestr(name, data)


def _archive_members(solved: SolvedSystem, config: SpeakerPackageConfig) -> tuple[dict[str, bytes], dict[str, Any]]:
    sphere = solved.domains[SPHERE_DOMAIN_ID]
    sphere_pressure = solved.quantities[SPHERE_PRESSURE_ID]
    sphere_points = np.asarray(sphere.coordinates["points_m"], dtype=np.float64)
    radius = np.linalg.norm(sphere_points, axis=1)
    directions = np.divide(
        sphere_points,
        radius[:, None],
        out=np.zeros_like(sphere_points),
        where=radius[:, None] > 0.0,
    )
    members: dict[str, bytes] = {
        "data/patterns.npz": _npz_bytes(
            frequencies_hz=np.asarray(solved.frequencies_hz),
            points_m=rotate_source_points(sphere_points),
            directions_xyz=rotate_source_points(directions),
            radius_m=radius,
            pressure_pa=np.asarray(sphere_pressure.values),
        )
    }
    medium = _exterior_medium(solved)
    capabilities = ["complex_spherical_pattern"]
    files: dict[str, Any] = {
        "patterns": {
            "path": "data/patterns.npz",
            "pressure_space": "sampled_points",
            "pressure_unit": "Pa",
            "dimensions": ["frequency", "excitation", "direction"],
        }
    }
    if config.fidelity >= SpeakerPackageFidelity.FIXED_SOURCES:
        domain = solved.domains[BEM_BOUNDARY_DOMAIN_ID]
        pressure = solved.quantities[BEM_BOUNDARY_PRESSURE_ID]
        neumann = solved.quantities[BEM_BOUNDARY_NEUMANN_ID]
        symmetry = normalize_symmetry(
            domain.metadata.get("symmetry", solved.provenance.solver_options.get("symmetry", "off"))
        )
        expanded = expand_bem_boundary_symmetry(
            points_m=np.asarray(domain.coordinates["points_m"]),
            triangles=np.asarray(domain.topology["triangles"]),
            pressure_pa=np.asarray(pressure.values),
            normal_derivative_pa_per_m=np.asarray(neumann.values),
            symmetry=symmetry,
        )
        payload: dict[str, np.ndarray] = {
            "points_m": rotate_source_points(expanded.points_m),
            "triangles": expanded.triangles,
            "pressure_pa": expanded.pressure_pa,
            "normal_derivative_pa_per_m": expanded.normal_derivative_pa_per_m,
            "source_bem_node_index": expanded.source_node_index,
            "source_bem_face_index": expanded.source_face_index,
            "face_symmetry_image_index": expanded.face_symmetry_image_index,
        }
        if "mesh_index" in domain.coordinates:
            payload["node_mesh_index"] = np.take(
                np.asarray(domain.coordinates["mesh_index"]), expanded.source_node_index, axis=0
            )
        for source_key, target in (("mesh_index", "face_mesh_index"), ("source_physical_tag", "source_physical_tag")):
            if source_key in domain.topology:
                payload[target] = np.take(np.asarray(domain.topology[source_key]), expanded.source_face_index, axis=0)
        members["data/fixed-sources.npz"] = _npz_bytes(**payload)
        members["geometry/exterior.msh"] = _gmsh22_surface_bytes(payload["points_m"], payload["triangles"])
        capabilities.append("fixed_distributed_sources")
        files["fixed_sources"] = {
            "path": "data/fixed-sources.npz",
            "geometry_mesh": "geometry/exterior.msh",
            "field_representation": "double_layer_pressure_minus_single_layer_normal_derivative",
            "pressure_space": "P1",
            "normal_derivative_space": "DP0",
            "pressure_unit": "Pa",
            "normal_derivative_unit": "Pa/m",
            "geometry_expansion": "full",
            "source_symmetry": symmetry,
            "source_node_count": int(np.asarray(domain.coordinates["points_m"]).shape[0]),
            "source_face_count": int(np.asarray(domain.topology["triangles"]).shape[0]),
            "exported_node_count": int(expanded.points_m.shape[0]),
            "exported_face_count": int(expanded.triangles.shape[0]),
            "symmetry_images": list(expanded.symmetry_images),
            "index_maps": {
                "source_bem_node_index": "exported bem_node -> reduced-domain bem_node",
                "source_bem_face_index": "exported bem_face -> reduced-domain bem_face",
                "face_symmetry_image_index": "exported bem_face -> symmetry_images index",
            },
            "dimensions": {
                "pressure": ["frequency", "excitation", "bem_node"],
                "normal_derivative": ["frequency", "excitation", "bem_face"],
            },
        }
    if config.fidelity >= SpeakerPackageFidelity.COUPLED:
        if config.coupled_representation == SpeakerPackageCoupledRepresentation.EXACT_SYSTEM:
            exact_members, exact_descriptor = _exact_system_archive_members(solved)
            members.update(exact_members)
            capabilities.append("exact_frequency_parametric_interior")
            files["coupled_model"] = exact_descriptor
        elif config.coupled_representation == SpeakerPackageCoupledRepresentation.PARITY_ROM:
            quantity_by_archive_name = {
                archive_name: solved.quantities[identifier]
                for identifier, _quantity, archive_name in SPEAKER_ROM_QUANTITIES
            }
            rom_metadata = dict(quantity_by_archive_name["k"].metadata)
            rom_metadata.pop("matrix", None)
            members["data/coupled-rom.npz"] = _npz_bytes(
                frequencies_hz=np.asarray(solved.frequencies_hz, dtype=np.float64),
                **{
                    name: np.asarray(quantity.values, dtype=np.complex64)
                    for name, quantity in quantity_by_archive_name.items()
                },
            )
            capabilities.append("parity_petrov_galerkin_rom")
            rom_symmetry = str(rom_metadata.get("symmetry_mode", "xy"))
            default_image_count = {"off": 1, "x": 2, "xy": 4}.get(rom_symmetry, 4)
            files["coupled_model"] = {
                "path": "data/coupled-rom.npz",
                "representation": "parity_petrov_galerkin_rom",
                "format_version": 1,
                "symmetry_mode": rom_symmetry,
                "image_count": int(rom_metadata.get("image_count", default_image_count)),
                "rank_per_sector": int(rom_metadata["rank_per_sector"]),
                "sector_names": rom_metadata["sector_names"],
                "sector_signs": rom_metadata["sector_signs"],
                "node_orbits": rom_metadata["node_orbits"],
                "face_orbits": rom_metadata["face_orbits"],
                "equations": rom_metadata["equations"],
                "exterior_pressure_space": "P1",
                "exterior_normal_derivative_space": "DP0",
                "frequency_continuous_within_band": False,
                "input_ports": _coupled_input_ports(solved),
                "validation": rom_metadata["validation"],
                "matrix_dimensions": {
                    "k": ["frequency", "parity_sector", "reduced_row", "reduced_column"],
                    "c": ["frequency", "parity_sector", "reduced_row", "boundary_node_orbit"],
                    "d": ["frequency", "parity_sector", "boundary_face_orbit", "reduced_column"],
                    "b": ["frequency", "parity_sector", "reduced_row", "input_port"],
                    "e": ["frequency", "parity_sector", "boundary_face_orbit", "input_port"],
                    "velocity": ["frequency", "parity_sector", "transducer", "reduced_column"],
                    "current": ["frequency", "parity_sector", "transducer", "reduced_column"],
                    "velocity_drive": ["frequency", "parity_sector", "transducer", "input_port"],
                    "current_drive": ["frequency", "parity_sector", "transducer", "input_port"],
                },
                "metadata": rom_metadata,
            }
    manifest = {
        "schema": SPEAKER_PACKAGE_SCHEMA,
        "schema_version": SPEAKER_PACKAGE_SCHEMA_VERSION,
        "name": config.name,
        "fidelity_level": int(config.fidelity),
        "fidelity": config.fidelity.cli_name,
        "capabilities": capabilities,
        "frequencies_hz": np.asarray(solved.frequencies_hz, dtype=float).tolist(),
        "excitation_port_ids": list(solved.excitation_ids),
        "phasor_convention": solved.provenance.phasor_convention,
        "coordinate_system": {
            "handedness": "right",
            "unit": "m",
            "forward_axis": "+Y",
            "source_forward_axis": "+Z",
            "source_to_package_rotation": SOURCE_TO_PACKAGE_ROTATION.tolist(),
            "mapping": "(x, y, z) -> (x, z, -y)",
        },
        "medium": medium,
        "files": files,
        "physical_system": _physical_system_metadata(solved),
        "provenance": {
            "run_id": solved.run_id,
            "backend_id": solved.provenance.backend_id,
            "solve_kind": solved.provenance.solve_kind,
            "solver_options": _portable_solver_options(solved.provenance.solver_options),
            "started_at_utc": solved.provenance.started_at_utc,
            "finished_at_utc": solved.finished_at_utc,
        },
    }
    return members, manifest


def _exact_system_archive_members(solved: SolvedSystem) -> tuple[dict[str, bytes], dict[str, Any]]:
    system = solved.compiled_system
    if system is None:
        raise ValueError("Exact Level-3 export requires a compiled physical system.")
    compiled = compiled_system_to_dict(system)
    members: dict[str, bytes] = {}
    mesh_members: dict[str, str] = {}
    compiled_meshes = {str(item["id"]): item for item in compiled.get("meshes", ())}
    for index, mesh in enumerate(system.meshes):
        source = Path(mesh.file)
        if not source.is_file():
            raise ValueError(f"Exact Level-3 mesh {source!s} is unavailable.")
        suffix = source.suffix.lower() or ".msh"
        member = f"geometry/coupled/{index:03d}-{_safe_member_stem(mesh.name)}{suffix}"
        members[member] = source.read_bytes()
        mesh_members[mesh.id] = member
        compiled_meshes[mesh.id]["file"] = member

    descriptor_path = "data/coupled-exact-system.json"
    payload = {
        "format_version": 1,
        "representation": "exact_frequency_parametric_fem",
        "mesh_path_kind": "package_member",
        "compiled_system": compiled,
        "mesh_members": mesh_members,
        "solver_options": _portable_solver_options(solved.provenance.solver_options),
        "frequencies_hz": np.asarray(solved.frequencies_hz, dtype=float).tolist(),
        "frequency_band_hz": [float(np.min(solved.frequencies_hz)), float(np.max(solved.frequencies_hz))],
        "phasor_convention": solved.provenance.phasor_convention,
        "source_to_package_rotation": SOURCE_TO_PACKAGE_ROTATION.tolist(),
    }
    members[descriptor_path] = _json_bytes(payload)
    descriptor = {
        "path": descriptor_path,
        "representation": "exact_frequency_parametric_fem",
        "format_version": 2,
        "equations": ["K z + C p = B u", "q = D z + E u"],
        "exterior_pressure_space": "P1",
        "exterior_normal_derivative_space": "DP0",
        "runtime_assembly_required": True,
        "frequency_continuous_within_band": True,
        "mesh_members": mesh_members,
        "input_ports": _coupled_input_ports(solved),
    }
    return members, descriptor


def _coupled_input_ports(solved: SolvedSystem) -> list[dict[str, Any]]:
    return [
        {
            "id": port.id,
            "kind": _enum_value(port.kind),
            "unit": "V" if _enum_value(port.kind) == "voltage" else "m/s",
            "normalization": 1.0,
        }
        for port in (solved.compiled_system.excitation_ports if solved.compiled_system is not None else ())
        if port.id in solved.excitation_ids
    ]


def _safe_member_stem(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    return cleaned.strip("_") or "mesh"


def expand_bem_boundary_symmetry(
    *,
    points_m: np.ndarray,
    triangles: np.ndarray,
    pressure_pa: np.ndarray,
    normal_derivative_pa_per_m: np.ndarray,
    symmetry: object,
    tolerance_m: float | None = None,
) -> ExpandedBemBoundary:
    """Materialize X/XY mirror images while preserving P1/DP0 trace meaning.

    Pressure and outward-normal pressure derivative are scalar traces under the
    reflection. Odd-parity reflections reverse triangle winding so their
    geometric outward normals are reflected correctly. Nodes on a symmetry
    plane are shared by their images, and coincident images of the same source
    face are emitted only once.
    """

    mode = normalize_symmetry(symmetry)
    points = np.asarray(points_m, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    pressure = np.asarray(pressure_pa)
    neumann = np.asarray(normal_derivative_pa_per_m)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("BEM points must have shape (bem_node, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("BEM triangles must have shape (bem_face, 3).")
    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= points.shape[0]):
        raise ValueError("BEM triangles reference an invalid node index.")
    if pressure.ndim < 1 or pressure.shape[-1] != points.shape[0]:
        raise ValueError("BEM pressure must end with the bem_node dimension.")
    if neumann.ndim < 1 or neumann.shape[-1] != faces.shape[0]:
        raise ValueError("BEM normal derivative must end with the bem_face dimension.")
    if not np.all(np.isfinite(points)):
        raise ValueError("BEM points must be finite.")
    if tolerance_m is not None and (not np.isfinite(tolerance_m) or tolerance_m < 0.0):
        raise ValueError("Symmetry weld tolerance must be finite and non-negative.")

    identity = np.eye(3, dtype=np.float64)
    reflect_x = np.diag((-1.0, 1.0, 1.0))
    reflect_y = np.diag((1.0, -1.0, 1.0))
    images: list[tuple[str, np.ndarray]] = [("identity", identity)]
    if mode in {"x", "xy"}:
        images.append(("reflect_x", reflect_x))
    if mode == "xy":
        images.extend((("reflect_y", reflect_y), ("reflect_xy", reflect_x @ reflect_y)))

    snapped = snap_points_to_symmetry_planes(points, mode, tolerance_m=tolerance_m)

    expanded_points: list[np.ndarray] = []
    source_nodes: list[int] = []
    node_lookup: dict[tuple[int, float, float, float], int] = {}
    image_node_maps: list[np.ndarray] = []
    for _label, transform in images:
        transformed = snapped @ transform.T
        node_map = np.empty(points.shape[0], dtype=np.int64)
        for source_index, point in enumerate(transformed):
            key = (source_index, float(point[0]), float(point[1]), float(point[2]))
            target_index = node_lookup.get(key)
            if target_index is None:
                target_index = len(expanded_points)
                node_lookup[key] = target_index
                expanded_points.append(point.copy())
                source_nodes.append(source_index)
            node_map[source_index] = target_index
        image_node_maps.append(node_map)

    expanded_faces: list[np.ndarray] = []
    source_faces: list[int] = []
    face_images: list[int] = []
    face_lookup: set[tuple[int, int, int, int]] = set()
    for image_index, ((_label, transform), node_map) in enumerate(zip(images, image_node_maps, strict=True)):
        reflected = np.linalg.det(transform) < 0.0
        for source_index, face in enumerate(faces):
            mapped = node_map[face]
            if reflected:
                mapped = mapped[[0, 2, 1]]
            face_key = (source_index, *sorted(int(value) for value in mapped))
            if face_key in face_lookup:
                continue
            face_lookup.add(face_key)
            expanded_faces.append(mapped.copy())
            source_faces.append(source_index)
            face_images.append(image_index)

    node_sources = np.asarray(source_nodes, dtype=np.int64)
    face_sources = np.asarray(source_faces, dtype=np.int64)
    point_array = np.asarray(expanded_points, dtype=np.float64).reshape((-1, 3))
    face_array = np.asarray(expanded_faces, dtype=np.int64).reshape((-1, 3))
    image_metadata = tuple(
        {
            "index": index,
            "name": label,
            "source_frame_transform": transform.tolist(),
        }
        for index, (label, transform) in enumerate(images)
    )
    return ExpandedBemBoundary(
        points_m=point_array,
        triangles=face_array,
        pressure_pa=np.take(pressure, node_sources, axis=-1),
        normal_derivative_pa_per_m=np.take(neumann, face_sources, axis=-1),
        source_node_index=node_sources,
        source_face_index=face_sources,
        face_symmetry_image_index=np.asarray(face_images, dtype=np.int8),
        symmetry=mode,
        symmetry_images=image_metadata,
    )


def _exterior_medium(solved: SolvedSystem) -> dict[str, float]:
    if solved.compiled_system is None:
        raise ValueError("Speaker package export requires the compiled physical system.")
    exterior = [region for region in solved.compiled_system.regions if region.kind == AcousticRegionKind.UNBOUNDED_AIR]
    if len(exterior) != 1:
        raise ValueError("Speaker package export requires exactly one unbounded acoustic region.")
    return {
        "sound_speed_m_per_s": float(exterior[0].sound_speed_m_per_s),
        "density_kg_per_m3": float(exterior[0].density_kg_per_m3),
    }


def _portable_solver_options(options: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in options.items()
        if isinstance(value, (type(None), bool, int, float, str, list, dict))
    }


def _physical_system_metadata(solved: SolvedSystem) -> dict[str, Any]:
    system = solved.compiled_system
    if system is None:
        return {}
    return {
        "id": system.id,
        "name": system.name,
        "meshes": [{"id": item.id, "name": item.name, "purpose": _enum_value(item.purpose)} for item in system.meshes],
        "regions": [
            {
                "id": item.id,
                "name": item.name,
                "kind": _enum_value(item.kind),
                "sound_speed_m_per_s": float(item.sound_speed_m_per_s),
                "density_kg_per_m3": float(item.density_kg_per_m3),
            }
            for item in system.regions
        ],
        "boundaries": [
            {
                "id": item.id,
                "name": item.name,
                "region_id": item.region_id,
                "kind": _enum_value(item.kind),
                "mesh_id": item.group.mesh_id,
                "physical_tag": int(item.group.tag),
                "physical_name": item.group.name,
            }
            for item in system.boundaries
        ],
        "interfaces": [
            {
                "id": item.id,
                "name": item.name,
                "bounded_boundary_id": item.bounded_boundary_id,
                "unbounded_boundary_id": item.unbounded_boundary_id,
                "node_count": len(item.topology.fem_vertex_indices),
                "face_count": len(item.topology.fem_face_indices),
            }
            for item in system.interfaces
        ],
        "components": [
            {
                "id": item.id,
                "name": item.name,
                "kind": _enum_value(item.kind),
                "boundary_ids": list(item.boundary_ids),
                "parameters": item.parameters,
            }
            for item in system.components
        ],
        "excitation_ports": [
            {
                "id": item.id,
                "name": item.name,
                "component_id": item.component_id,
                "kind": _enum_value(item.kind),
            }
            for item in system.excitation_ports
        ],
        "metadata": system.metadata,
    }


def _enum_value(value: object) -> str:
    """Return a portable enum value while tolerating legacy string-backed models."""

    return str(value.value) if isinstance(value, Enum) else str(value)


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _gmsh22_surface_bytes(points: np.ndarray, triangles: np.ndarray) -> bytes:
    """Write a portable first-order surface mesh in the package coordinate frame."""

    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(points))]
    lines.extend(
        f"{index} {float(point[0]):.17g} {float(point[1]):.17g} {float(point[2]):.17g}"
        for index, point in enumerate(np.asarray(points), start=1)
    )
    lines.extend(("$EndNodes", "$Elements", str(len(triangles))))
    lines.extend(
        f"{index} 2 2 1 1 {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}"
        for index, face in enumerate(np.asarray(triangles), start=1)
    )
    lines.extend(("$EndElements", ""))
    return "\n".join(lines).encode("utf-8")


def _fibonacci_sphere_points(count: int, distance_m: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    indices = np.arange(count, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = golden_angle * indices
    x = radius_xy * np.cos(phi)
    y = radius_xy * np.sin(phi)
    points = distance_m * np.column_stack([x, y, z])
    return points, {
        "r_distance_m": np.full(count, distance_m, dtype=np.float32),
        "theta_polar_rad": np.arccos(np.clip(z, -1.0, 1.0)).astype(np.float32),
        "phi_azimuth_rad": np.arctan2(y, x).astype(np.float32),
    }


__all__ = [
    "ExpandedBemBoundary",
    "SOURCE_TO_PACKAGE_ROTATION",
    "SPEAKER_PACKAGE_SCHEMA_VERSION",
    "SpeakerPackageConfig",
    "SpeakerPackageCoupledRepresentation",
    "SpeakerPackageExportResult",
    "SpeakerPackageFidelity",
    "SPEAKER_ROM_QUANTITIES",
    "SpeakerPackageIssue",
    "export_speaker_package",
    "expand_bem_boundary_symmetry",
    "prepare_speaker_package_solve",
    "rotate_source_points",
    "solve_speaker_package_system",
    "speaker_package_issues",
    "validate_speaker_package",
]
