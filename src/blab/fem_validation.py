"""Qt-free validation metrics for tagged surfaces in headless FEM results."""

from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import meshio
import numpy as np

from blab.physical_model import AcousticRegionKind, BoundaryKind
from blab.system_contract import compiled_system_from_dict

FEM_PRESSURE_QUANTITY_ID = "acoustic:pressure:fem-nodes"
FEM_VALIDATION_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class FEMValidationGates:
    maximum_within_surface_phase_rms_deg: float = 10.0
    maximum_inter_surface_phase_rms_deg: float = 5.0
    maximum_inter_surface_phase_deg: float = 10.0
    minimum_plane_mode_fraction: float = 0.95
    minimum_points_per_wavelength_p95: float = 8.0
    minimum_points_per_wavelength_maximum_edge: float = 4.0


@dataclass(frozen=True)
class FEMConvergenceGates:
    maximum_surface_phase_rms_delta_deg: float = 1.0
    maximum_surface_phase_delta_deg: float = 2.0
    maximum_normalized_amplitude_rms_delta: float = 0.02
    maximum_plane_mode_fraction_delta: float = 0.01


@dataclass(frozen=True)
class _Surface:
    name: str
    region_id: str
    points_m: np.ndarray
    triangles: np.ndarray
    result_nodes: np.ndarray
    adjacent_result_tetrahedra: tuple[tuple[int, ...], ...]
    outward_normals: np.ndarray
    areas_m2: np.ndarray


def tetrahedral_shape_gradients(points_m: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    """Return physical P1 basis gradients with shape ``(tetrahedron, 4, xyz)``."""

    points = np.asarray(points_m, dtype=float)
    cells = np.asarray(tetrahedra, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("FEM points must have shape (node, 3).")
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("FEM tetrahedra must have shape (tetrahedron, 4).")
    vertices = points[cells]
    jacobians = np.stack(
        (vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0], vertices[:, 3] - vertices[:, 0]),
        axis=2,
    )
    determinants = np.linalg.det(jacobians)
    edge_scale = np.max(np.abs(jacobians), axis=(1, 2))
    if np.any(np.abs(determinants) <= np.finfo(float).eps * edge_scale**3):
        raise ValueError("FEM volume mesh contains a numerically degenerate tetrahedron.")
    reference = np.asarray(
        [[-1.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )
    gradients = np.linalg.solve(
        np.swapaxes(jacobians, 1, 2),
        np.broadcast_to(reference, (cells.shape[0], 3, 4)),
    )
    return np.swapaxes(gradients, 1, 2)


def quadratic_tetrahedral_shape_gradients(
    points_m: np.ndarray,
    tetrahedra10: np.ndarray,
    barycentric_coordinates: np.ndarray,
) -> np.ndarray:
    """Return affine P2 basis gradients at supplied tetrahedron barycentric points."""

    points = np.asarray(points_m, dtype=float)
    cells = np.asarray(tetrahedra10, dtype=np.int64)
    barycentric = np.asarray(barycentric_coordinates, dtype=float)
    if cells.ndim != 2 or cells.shape[1] != 10:
        raise ValueError("Quadratic FEM tetrahedra must have shape (tetrahedron, 10).")
    if barycentric.shape != (cells.shape[0], 4):
        raise ValueError("Quadratic FEM evaluation coordinates must have shape (tetrahedron, 4).")
    vertices = points[cells[:, :4]]
    jacobians = np.stack(
        (
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
        ),
        axis=2,
    )
    gradient_lambda = np.asarray(
        [[-1.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )
    reference = np.empty((cells.shape[0], 3, 10), dtype=float)
    for index in range(4):
        reference[:, :, index] = (4.0 * barycentric[:, index] - 1.0)[:, np.newaxis] * gradient_lambda[:, index]
    for offset, (left, right) in enumerate(((0, 1), (1, 2), (2, 0), (0, 3), (3, 2), (1, 3))):
        reference[:, :, 4 + offset] = 4.0 * (
            barycentric[:, left, np.newaxis] * gradient_lambda[:, right]
            + barycentric[:, right, np.newaxis] * gradient_lambda[:, left]
        )
    gradients = np.linalg.solve(np.swapaxes(jacobians, 1, 2), reference)
    return np.swapaxes(gradients, 1, 2)


def tetrahedron_edge_statistics_m(
    points_m: np.ndarray,
    tetrahedra: np.ndarray,
) -> dict[str, float]:
    """Return repeated tetrahedron-edge length statistics for resolution checks."""

    points = np.asarray(points_m, dtype=float)
    cells = np.asarray(tetrahedra, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("FEM points must have shape (node, 3).")
    if cells.ndim != 2 or cells.shape[1] not in (4, 10) or not len(cells):
        raise ValueError("FEM tetrahedra must have non-empty shape (tetrahedron, 4 or 10).")
    # Resolution is measured on the underlying straight-sided tetrahedron. The
    # six additional P2 nodes are interpolation nodes, not mesh edges.
    vertices = points[cells[:, :4]]
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    lengths = np.concatenate([np.linalg.norm(vertices[:, right] - vertices[:, left], axis=1) for left, right in pairs])
    return {
        "minimum": float(np.min(lengths)),
        "median": float(np.quantile(lengths, 0.5)),
        "p95": float(np.quantile(lengths, 0.95)),
        "maximum": float(np.max(lengths)),
    }


def frequency_mesh_resolution(
    edge_statistics_m: dict[str, float],
    frequency_hz: float,
    minimum_sound_speed_m_per_s: float,
    minimum_points_per_wavelength_p95: float,
    minimum_points_per_wavelength_maximum_edge: float,
) -> dict[str, Any]:
    """Describe P1 mesh resolution at one frequency using the slowest region."""

    if (
        min(
            frequency_hz,
            minimum_sound_speed_m_per_s,
            minimum_points_per_wavelength_p95,
            minimum_points_per_wavelength_maximum_edge,
        )
        <= 0.0
    ):
        raise ValueError("Frequency, sound speed, and resolution gate must be positive.")
    wavelength = minimum_sound_speed_m_per_s / frequency_hz
    p95_edge = float(edge_statistics_m["p95"])
    maximum_edge = float(edge_statistics_m["maximum"])
    p95_points = wavelength / p95_edge
    minimum_points = wavelength / maximum_edge
    return {
        "minimum_wavelength_m": wavelength,
        "points_per_wavelength_at_p95_edge": p95_points,
        "points_per_wavelength_at_maximum_edge": minimum_points,
        "minimum_points_per_wavelength_p95": minimum_points_per_wavelength_p95,
        "minimum_points_per_wavelength_maximum_edge": (minimum_points_per_wavelength_maximum_edge),
        "required_p95_edge_m": wavelength / minimum_points_per_wavelength_p95,
        "required_maximum_edge_m": (wavelength / minimum_points_per_wavelength_maximum_edge),
        "adequate": (
            p95_points >= minimum_points_per_wavelength_p95
            and minimum_points >= minimum_points_per_wavelength_maximum_edge
        ),
    }


def surface_pressure_metrics(
    points_m: np.ndarray,
    triangles: np.ndarray,
    pressure: np.ndarray,
) -> dict[str, Any]:
    """Calculate exact P1 mean/energy and quadrature phase metrics on a surface."""

    points = np.asarray(points_m, dtype=float)
    cells = np.asarray(triangles, dtype=np.int64)
    values = np.asarray(pressure, dtype=complex)
    vertices = points[cells]
    areas = 0.5 * np.linalg.norm(np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=1)
    area = float(np.sum(areas))
    if area <= 0.0:
        raise ValueError("Validation surface has zero area.")
    nodal = values[cells]
    if cells.shape[1] == 6:
        return _quadratic_surface_pressure_metrics(areas, nodal)
    if cells.shape[1] != 3:
        raise ValueError("Validation surface triangles must have three or six nodes.")
    mean = np.sum(areas * np.mean(nodal, axis=1)) / area
    mass = np.asarray([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
    energy = float(np.sum(areas * np.real(np.einsum("ti,ij,tj->t", np.conj(nodal), mass / 12.0, nodal))))
    plane_fraction = float(area * abs(mean) ** 2 / energy) if energy > 0.0 else 0.0

    barycentric = np.asarray(
        [[2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]]
    )
    quadrature = nodal @ barycentric.T
    weights = np.repeat(areas / 3.0, 3)
    flat = quadrature.reshape(-1)
    reference_pressure = mean if abs(mean) > np.finfo(float).eps else 1.0 + 0.0j
    residual_deg = np.degrees(np.angle(flat * np.conj(reference_pressure)))
    phase_rms_deg = float(np.sqrt(np.sum(weights * residual_deg**2) / area))
    phase_abs = np.abs(residual_deg)
    phase_p95_deg = _weighted_quantile(phase_abs, weights, 0.95)
    magnitude = np.abs(flat)
    magnitude_mean = float(np.sum(weights * magnitude) / area)
    magnitude_cv = (
        float(np.sqrt(np.sum(weights * (magnitude - magnitude_mean) ** 2) / area) / magnitude_mean)
        if magnitude_mean > 0.0
        else 0.0
    )
    coherence_denominator = float(np.sum(areas * np.mean(np.abs(nodal), axis=1)))
    coherence = float(area * abs(mean) / coherence_denominator) if coherence_denominator > 0.0 else 0.0
    return {
        "area_m2": area,
        "mean_pressure_pa": _complex_dict(mean),
        "plane_mode_fraction": min(max(plane_fraction, 0.0), 1.0),
        "pressure_coherence": min(max(coherence, 0.0), 1.0),
        "phase_rms_deg": phase_rms_deg,
        "phase_p95_deg": phase_p95_deg,
        "phase_max_deg": float(np.max(phase_abs)),
        "magnitude_coefficient_of_variation": magnitude_cv,
    }


def _quadratic_surface_pressure_metrics(
    areas: np.ndarray,
    nodal: np.ndarray,
) -> dict[str, Any]:
    first = (0.816847572980459, 0.091576213509771, 0.109951743655322)
    second = (0.108103018168070, 0.445948490915965, 0.223381589678011)
    barycentric = []
    quadrature_weights = []
    for single, repeated, weight in (first, second):
        for single_index in range(3):
            coordinates = np.full(3, repeated)
            coordinates[single_index] = single
            barycentric.append(coordinates)
            quadrature_weights.append(weight)
    barycentric = np.asarray(barycentric)
    basis = np.column_stack(
        (
            barycentric[:, 0] * (2.0 * barycentric[:, 0] - 1.0),
            barycentric[:, 1] * (2.0 * barycentric[:, 1] - 1.0),
            barycentric[:, 2] * (2.0 * barycentric[:, 2] - 1.0),
            4.0 * barycentric[:, 0] * barycentric[:, 1],
            4.0 * barycentric[:, 1] * barycentric[:, 2],
            4.0 * barycentric[:, 2] * barycentric[:, 0],
        )
    )
    quadrature = nodal @ basis.T
    weights = (areas[:, np.newaxis] * np.asarray(quadrature_weights)).reshape(-1)
    flat = quadrature.reshape(-1)
    area = float(np.sum(areas))
    mean = np.sum(weights * flat) / area
    energy = float(np.sum(weights * np.abs(flat) ** 2))
    plane_fraction = float(area * abs(mean) ** 2 / energy) if energy > 0.0 else 0.0
    reference_pressure = mean if abs(mean) > np.finfo(float).eps else 1.0 + 0.0j
    residual_deg = np.degrees(np.angle(flat * np.conj(reference_pressure)))
    phase_abs = np.abs(residual_deg)
    magnitude = np.abs(flat)
    magnitude_mean = float(np.sum(weights * magnitude) / area)
    magnitude_cv = (
        float(np.sqrt(np.sum(weights * (magnitude - magnitude_mean) ** 2) / area) / magnitude_mean)
        if magnitude_mean > 0.0
        else 0.0
    )
    coherence_denominator = float(np.sum(weights * magnitude))
    coherence = float(area * abs(mean) / coherence_denominator) if coherence_denominator > 0.0 else 0.0
    return {
        "area_m2": area,
        "mean_pressure_pa": _complex_dict(mean),
        "plane_mode_fraction": min(max(plane_fraction, 0.0), 1.0),
        "pressure_coherence": min(max(coherence, 0.0), 1.0),
        "phase_rms_deg": float(np.sqrt(np.sum(weights * residual_deg**2) / area)),
        "phase_p95_deg": _weighted_quantile(phase_abs, weights, 0.95),
        "phase_max_deg": float(np.max(phase_abs)),
        "magnitude_coefficient_of_variation": magnitude_cv,
    }


def evaluate_fem_run(
    run_dir: str | Path,
    *,
    surface_patterns: tuple[str, ...] = ("exit_*",),
    split_surface_entities: bool = False,
    gates: FEMValidationGates | None = None,
) -> dict[str, Any]:
    """Evaluate tagged FEM surfaces in a completed Boundary Lab headless run."""

    root = Path(run_dir).resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema") != "boundary-lab-headless-result":
        raise ValueError(f"Not a Boundary Lab headless result: {root}")
    if manifest.get("status") != "complete":
        raise ValueError(f"FEM result is not complete (status={manifest.get('status')!r}).")
    _verify_source_meshes(manifest)
    system = compiled_system_from_dict(_read_json(root / str(manifest["compiled_system_file"])))
    domain_meta = _read_json(root / str(manifest["domains_metadata_file"]))
    with np.load(root / str(manifest["domains_file"])) as arrays:
        fem_domain = next(item for item in domain_meta["domains"] if item["kind"] == "fem_volume")
        result_points = np.asarray(arrays[fem_domain["coordinates"]["points_m"]], dtype=float)
        tetrahedron_key = fem_domain["topology"].get(
            "tetrahedra10",
            fem_domain["topology"]["tetrahedra"],
        )
        result_tetrahedra = np.asarray(arrays[tetrahedron_key], dtype=np.int64)
    surfaces = _reconstruct_surfaces(
        system,
        fem_domain,
        result_points,
        result_tetrahedra,
        surface_patterns,
        split_surface_entities,
    )
    if not surfaces:
        raise ValueError(f"No FEM boundary surfaces match {surface_patterns!r}.")

    configured_gates = gates or FEMValidationGates()
    edge_statistics = tetrahedron_edge_statistics_m(result_points, result_tetrahedra)
    bounded_sound_speeds = [
        float(region.sound_speed_m_per_s) for region in system.regions if region.kind == AcousticRegionKind.BOUNDED_AIR
    ]
    if not bounded_sound_speeds:
        raise ValueError("FEM result does not contain a bounded acoustic region.")
    minimum_sound_speed = min(bounded_sound_speeds)
    frequency_results: list[dict[str, Any]] = []
    excitation_ids = tuple(str(value) for value in manifest["excitation_port_ids"])
    for result_entry in manifest["results"]:
        if result_entry is None:
            continue
        metadata = _read_json(root / str(result_entry["metadata_file"]))
        pressure_meta = next(
            (item for item in metadata["quantities"] if item["id"] == FEM_PRESSURE_QUANTITY_ID),
            None,
        )
        if pressure_meta is None:
            raise ValueError(f"Frequency {metadata['freq_hz']:g} Hz does not contain {FEM_PRESSURE_QUANTITY_ID}.")
        with np.load(root / str(result_entry["arrays_file"])) as arrays:
            pressure_by_excitation = np.asarray(arrays[pressure_meta["key"]], dtype=complex)
        if pressure_by_excitation.ndim == 1:
            pressure_by_excitation = pressure_by_excitation[np.newaxis, :]
        if pressure_by_excitation.shape != (len(excitation_ids), len(result_points)):
            raise ValueError(
                f"FEM pressure shape {pressure_by_excitation.shape} does not match "
                f"({len(excitation_ids)}, {len(result_points)})."
            )
        for excitation_index, excitation_id in enumerate(excitation_ids):
            frequency_results.append(
                _evaluate_frequency(
                    float(metadata["freq_hz"]),
                    excitation_id,
                    pressure_by_excitation[excitation_index],
                    surfaces,
                    result_points,
                    result_tetrahedra,
                    system,
                    configured_gates,
                    frequency_mesh_resolution(
                        edge_statistics,
                        float(metadata["freq_hz"]),
                        minimum_sound_speed,
                        configured_gates.minimum_points_per_wavelength_p95,
                        configured_gates.minimum_points_per_wavelength_maximum_edge,
                    ),
                )
            )

    by_excitation: dict[str, list[dict[str, Any]]] = {item: [] for item in excitation_ids}
    for item in frequency_results:
        by_excitation[item["excitation_port_id"]].append(item)
    ceilings = {
        excitation_id: _sampled_ceiling(sorted(items, key=lambda item: item["frequency_hz"]))
        for excitation_id, items in by_excitation.items()
    }
    return {
        "schema": "boundary-lab-fem-validation",
        "schema_version": FEM_VALIDATION_SCHEMA_VERSION,
        "source_run": str(root),
        "source_manifest_sha256": _sha256(root / "manifest.json"),
        "project_sha256": manifest.get("project_sha256"),
        "phasor_convention": manifest.get("phasor_convention"),
        "surface_patterns": list(surface_patterns),
        "split_surface_entities": split_surface_entities,
        "gates": asdict(configured_gates),
        "tetrahedron_edge_statistics_m": edge_statistics,
        "surface_count": len(surfaces),
        "frequencies_hz": sorted({item["frequency_hz"] for item in frequency_results}),
        "sampled_coherence_ceiling_hz_by_excitation": ceilings,
        "results": sorted(frequency_results, key=lambda item: (item["excitation_port_id"], item["frequency_hz"])),
    }


def write_fem_validation_report(path: str | Path, report: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def compare_fem_validation_reports(
    coarse_report: str | Path,
    fine_report: str | Path,
    *,
    gates: FEMConvergenceGates | None = None,
) -> dict[str, Any]:
    """Compare normalized tagged-surface fields from two validation reports."""

    coarse_path = Path(coarse_report).resolve()
    fine_path = Path(fine_report).resolve()
    coarse = _read_json(coarse_path)
    fine = _read_json(fine_path)
    for label, report in (("coarse", coarse), ("fine", fine)):
        if report.get("schema") != "boundary-lab-fem-validation":
            raise ValueError(f"{label.capitalize()} input is not a FEM validation report.")
    configured = gates or FEMConvergenceGates()
    coarse_results = {
        (float(item["frequency_hz"]), str(item["excitation_port_id"])): item for item in coarse["results"]
    }
    fine_results = {(float(item["frequency_hz"]), str(item["excitation_port_id"])): item for item in fine["results"]}
    common_results = set(coarse_results) & set(fine_results)
    if not common_results:
        raise ValueError("Convergence reports have no common frequency/excitation pairs.")

    comparisons = []
    for frequency_hz, excitation_id in sorted(common_results):
        coarse_result = coarse_results[(frequency_hz, excitation_id)]
        fine_result = fine_results[(frequency_hz, excitation_id)]
        coarse_surfaces = {item["name"]: item for item in coarse_result["surfaces"]}
        fine_surfaces = {item["name"]: item for item in fine_result["surfaces"]}
        if set(coarse_surfaces) != set(fine_surfaces):
            raise ValueError(f"Surface names differ at {frequency_hz:g} Hz for {excitation_id!r}.")
        names = sorted(coarse_surfaces)
        areas = np.asarray([fine_surfaces[name]["area_m2"] for name in names], dtype=float)
        coarse_pressure = np.asarray([_complex_from_dict(coarse_surfaces[name]["mean_pressure_pa"]) for name in names])
        fine_pressure = np.asarray([_complex_from_dict(fine_surfaces[name]["mean_pressure_pa"]) for name in names])
        coarse_reference = np.sum(areas * coarse_pressure)
        fine_reference = np.sum(areas * fine_pressure)
        if min(abs(coarse_reference), abs(fine_reference)) <= np.finfo(float).eps:
            raise ValueError(f"Mean surface pressure is zero at {frequency_hz:g} Hz.")
        phase_delta = np.degrees(
            np.angle(fine_pressure * np.conj(fine_reference) * np.conj(coarse_pressure * np.conj(coarse_reference)))
        )
        area_sum = float(np.sum(areas))
        phase_rms = float(np.sqrt(np.sum(areas * phase_delta**2) / area_sum))
        coarse_magnitude = np.abs(coarse_pressure)
        fine_magnitude = np.abs(fine_pressure)
        coarse_magnitude /= np.sum(areas * coarse_magnitude) / area_sum
        fine_magnitude /= np.sum(areas * fine_magnitude) / area_sum
        amplitude_delta = fine_magnitude - coarse_magnitude
        amplitude_rms = float(np.sqrt(np.sum(areas * amplitude_delta**2) / area_sum))
        plane_delta = max(
            abs(float(fine_surfaces[name]["plane_mode_fraction"]) - float(coarse_surfaces[name]["plane_mode_fraction"]))
            for name in names
        )
        checks = {
            "surface_phase_rms": phase_rms <= configured.maximum_surface_phase_rms_delta_deg,
            "surface_phase_max": float(np.max(np.abs(phase_delta))) <= configured.maximum_surface_phase_delta_deg,
            "normalized_amplitude_rms": amplitude_rms <= configured.maximum_normalized_amplitude_rms_delta,
            "plane_mode_fraction": plane_delta <= configured.maximum_plane_mode_fraction_delta,
        }
        comparisons.append(
            {
                "frequency_hz": frequency_hz,
                "excitation_port_id": excitation_id,
                "passed": all(checks.values()),
                "checks": checks,
                "surface_phase_rms_delta_deg": phase_rms,
                "surface_phase_max_delta_deg": float(np.max(np.abs(phase_delta))),
                "normalized_amplitude_rms_delta": amplitude_rms,
                "maximum_plane_mode_fraction_delta": plane_delta,
                "surface_phase_delta_deg": {name: float(value) for name, value in zip(names, phase_delta, strict=True)},
            }
        )
    return {
        "schema": "boundary-lab-fem-convergence",
        "schema_version": 1,
        "coarse_report": str(coarse_path),
        "fine_report": str(fine_path),
        "gates": asdict(configured),
        "coarse_only_frequency_excitations": [list(item) for item in sorted(set(coarse_results) - common_results)],
        "fine_only_frequency_excitations": [list(item) for item in sorted(set(fine_results) - common_results)],
        "passed": all(item["passed"] for item in comparisons),
        "comparisons": comparisons,
    }


def _evaluate_frequency(
    frequency_hz: float,
    excitation_id: str,
    pressure: np.ndarray,
    surfaces: list[_Surface],
    points: np.ndarray,
    tetrahedra: np.ndarray,
    system,
    gates: FEMValidationGates,
    mesh_resolution: dict[str, Any],
) -> dict[str, Any]:
    region_by_id = {region.id: region for region in system.regions}
    surface_results: list[dict[str, Any]] = []
    for surface in surfaces:
        metrics = surface_pressure_metrics(surface.points_m, surface.triangles, pressure[surface.result_nodes])
        region = region_by_id[surface.region_id]
        velocity_values: list[complex] = []
        velocity_jump_values: list[float] = []
        for area, normal, adjacent, surface_triangle in zip(
            surface.areas_m2,
            surface.outward_normals,
            surface.adjacent_result_tetrahedra,
            surface.triangles,
            strict=True,
        ):
            adjacent_indices = np.asarray(adjacent, dtype=np.int64)
            adjacent_tetrahedra = tetrahedra[adjacent_indices]
            if tetrahedra.shape[1] == 10:
                face_nodes = surface.result_nodes[surface_triangle[:3]]
                barycentric = np.zeros((len(adjacent_tetrahedra), 4), dtype=float)
                for tet_index, tet in enumerate(adjacent_tetrahedra):
                    for face_node in face_nodes:
                        positions = np.flatnonzero(tet[:4] == face_node)
                        if len(positions) != 1:
                            raise ValueError("Quadratic validation face does not match its adjacent tetrahedron.")
                        barycentric[tet_index, positions[0]] = 1.0 / 3.0
                gradients = quadratic_tetrahedral_shape_gradients(
                    points,
                    adjacent_tetrahedra,
                    barycentric,
                )
            else:
                gradients = tetrahedral_shape_gradients(points, adjacent_tetrahedra)
            tet_pressure = pressure[adjacent_tetrahedra]
            pressure_gradient = np.einsum("ti,tij->tj", tet_pressure, gradients)
            normal_velocity = (
                pressure_gradient @ normal / (1j * 2.0 * np.pi * frequency_hz * float(region.density_kg_per_m3))
            )
            velocity_values.append(complex(np.mean(normal_velocity)))
            if len(normal_velocity) > 1:
                velocity_jump_values.append(float(abs(normal_velocity[0] - normal_velocity[1])))
            else:
                velocity_jump_values.append(0.0)
        area = np.asarray(surface.areas_m2)
        mean_velocity = np.sum(area * np.asarray(velocity_values)) / np.sum(area)
        mean_pressure = _complex_from_dict(metrics["mean_pressure_pa"])
        characteristic_impedance = float(region.density_kg_per_m3 * region.sound_speed_m_per_s)
        forward = 0.5 * (mean_pressure + characteristic_impedance * mean_velocity)
        backward = 0.5 * (mean_pressure - characteristic_impedance * mean_velocity)
        reflection = backward / forward if abs(forward) > np.finfo(float).eps else complex(np.nan, np.nan)
        metrics.update(
            {
                "name": surface.name,
                "region_id": surface.region_id,
                "mean_outward_normal_velocity_m_per_s": _complex_dict(mean_velocity),
                "mean_specific_impedance_pa_s_per_m": _complex_dict(
                    mean_pressure / mean_velocity
                    if abs(mean_velocity) > np.finfo(float).eps
                    else complex(np.nan, np.nan)
                ),
                "forward_pressure_pa": _complex_dict(forward),
                "backward_pressure_pa": _complex_dict(backward),
                "reflection_coefficient": _complex_dict(reflection),
                "return_loss_db": float(-20.0 * np.log10(abs(reflection))) if abs(reflection) > 0.0 else float("inf"),
                "normal_velocity_gradient_jump_max_m_per_s": max(velocity_jump_values, default=0.0),
            }
        )
        surface_results.append(metrics)

    areas = np.asarray([item["area_m2"] for item in surface_results])
    means = np.asarray([_complex_from_dict(item["mean_pressure_pa"]) for item in surface_results])
    reference = np.sum(areas * means)
    residuals = np.degrees(np.angle(means * np.conj(reference))) if abs(reference) > 0.0 else np.zeros(len(means))
    inter_rms = float(np.sqrt(np.sum(areas * residuals**2) / np.sum(areas)))
    aggregate = {
        "worst_within_surface_phase_rms_deg": max(item["phase_rms_deg"] for item in surface_results),
        "minimum_plane_mode_fraction": min(item["plane_mode_fraction"] for item in surface_results),
        "inter_surface_phase_rms_deg": inter_rms,
        "inter_surface_phase_max_deg": float(np.max(np.abs(residuals))),
        "inter_surface_phase_peak_to_peak_deg": float(np.ptp(residuals)),
        "surface_mean_phase_residual_deg": {
            item["name"]: float(value) for item, value in zip(surface_results, residuals, strict=True)
        },
    }
    checks = {
        "within_surface_phase": aggregate["worst_within_surface_phase_rms_deg"]
        <= gates.maximum_within_surface_phase_rms_deg,
        "inter_surface_phase_rms": aggregate["inter_surface_phase_rms_deg"]
        <= gates.maximum_inter_surface_phase_rms_deg,
        "inter_surface_phase_max": aggregate["inter_surface_phase_max_deg"] <= gates.maximum_inter_surface_phase_deg,
        "plane_mode_fraction": aggregate["minimum_plane_mode_fraction"] >= gates.minimum_plane_mode_fraction,
        "mesh_resolution": bool(mesh_resolution["adequate"]),
    }
    return {
        "frequency_hz": frequency_hz,
        "excitation_port_id": excitation_id,
        "passed": all(checks.values()),
        "checks": checks,
        "mesh_resolution": mesh_resolution,
        "aggregate": aggregate,
        "surfaces": surface_results,
    }


def _reconstruct_surfaces(
    system,
    domain,
    result_points,
    result_tetrahedra,
    patterns,
    split_surface_entities: bool,
) -> list[_Surface]:
    meshes = {item.id: item for item in system.meshes}
    region_ids = list(domain["metadata"]["region_ids"])
    node_offsets = list(domain["metadata"]["node_offsets"])
    tetra_offsets = list(domain["metadata"]["tetra_offsets"])
    element_order = int(domain["metadata"].get("element_order", 1))
    if element_order == 2:
        volume_cell_types = {"tetra10"}
        surface_cell_types = {"triangle6"}
        volume_width = 10
        surface_width = 6
    elif element_order == 1:
        volume_cell_types = {"tetra", "tetra4"}
        surface_cell_types = {"triangle", "triangle3"}
        volume_width = 4
        surface_width = 3
    else:
        raise ValueError(f"Unsupported FEM element order {element_order}.")
    bounded = [region for region in system.regions if region.kind == AcousticRegionKind.BOUNDED_AIR]
    surfaces: list[_Surface] = []
    for region in bounded:
        region_position = region_ids.index(region.id)
        mesh_resource = meshes[region.mesh_ids[0]]
        mesh = meshio.read(Path(mesh_resource.file))
        selected_tags = {int(item.tag) for item in region.volume_groups if item.mesh_id == mesh_resource.id}
        selected_tets = _selected_cells(
            mesh,
            volume_cell_types,
            selected_tags,
            volume_width,
        )
        active_vertices = np.unique(selected_tets.reshape(-1))
        source_to_result = np.full(len(mesh.points), -1, dtype=np.int64)
        source_to_result[active_vertices] = np.arange(active_vertices.size) + int(node_offsets[region_position])
        face_adjacency: dict[tuple[int, int, int], list[int]] = {}
        for local_tet, tet in enumerate(selected_tets):
            for face in (
                (tet[0], tet[1], tet[2]),
                (tet[0], tet[1], tet[3]),
                (tet[0], tet[2], tet[3]),
                (tet[1], tet[2], tet[3]),
            ):
                face_adjacency.setdefault(tuple(sorted(int(value) for value in face)), []).append(
                    int(tetra_offsets[region_position]) + local_tet
                )
        region_boundaries = [item for item in system.boundaries if item.region_id == region.id]
        for boundary in region_boundaries:
            name = boundary.group.name or boundary.name
            if boundary.kind != BoundaryKind.PLANE_WAVE_TUBE_TERMINATION or not any(
                fnmatch.fnmatchcase(name, pattern) for pattern in patterns
            ):
                continue
            transformed_points = np.asarray(mesh.points, dtype=float) * float(mesh_resource.scale_to_m) + np.asarray(
                mesh_resource.translation_m, dtype=float
            )
            source_triangles, geometrical_tags = _selected_cells_with_entities(
                mesh,
                surface_cell_types,
                {int(boundary.group.tag)},
                surface_width,
            )
            groups: list[tuple[str, np.ndarray]] = [(name, source_triangles)]
            if split_surface_entities:
                entities = []
                for entity_tag in np.unique(geometrical_tags):
                    triangles = source_triangles[geometrical_tags == entity_tag]
                    center = np.mean(transformed_points[triangles].reshape(-1, 3), axis=0)
                    entities.append((int(entity_tag), triangles, center))
                entities = _ordered_split_surface_entities(entities)
                groups = [
                    (
                        f"{name}_s{index // 2}{index % 2}" if len(entities) == 4 else f"{name}_entity_{entity_tag}",
                        triangles,
                    )
                    for index, (entity_tag, triangles, _center) in enumerate(entities)
                ]

            for surface_name, triangles in groups:
                mapped = source_to_result[triangles]
                if np.any(mapped < 0):
                    raise ValueError(f"Surface {surface_name!r} contains nodes outside bounded region {region.id!r}.")
                triangle_points = transformed_points[triangles]
                crosses = np.cross(
                    triangle_points[:, 1] - triangle_points[:, 0],
                    triangle_points[:, 2] - triangle_points[:, 0],
                )
                areas = 0.5 * np.linalg.norm(crosses, axis=1)
                normals = crosses / (2.0 * areas[:, np.newaxis])
                adjacent: list[tuple[int, ...]] = []
                for index, source_triangle in enumerate(triangles):
                    tetra_indices = tuple(
                        face_adjacency.get(
                            tuple(sorted(int(value) for value in source_triangle[:3])),
                            (),
                        )
                    )
                    if not tetra_indices:
                        raise ValueError(f"Surface triangle on {surface_name!r} has no adjacent selected tetrahedron.")
                    adjacent.append(tetra_indices)
                    if len(tetra_indices) == 1:
                        face_center = np.mean(result_points[mapped[index, :3]], axis=0)
                        tet_center = np.mean(
                            result_points[result_tetrahedra[tetra_indices[0], :4]],
                            axis=0,
                        )
                        if np.dot(normals[index], face_center - tet_center) < 0.0:
                            normals[index] *= -1.0
                local_points, inverse = np.unique(mapped.reshape(-1), return_inverse=True)
                surfaces.append(
                    _Surface(
                        name=surface_name,
                        region_id=region.id,
                        points_m=result_points[local_points],
                        triangles=inverse.reshape(-1, surface_width),
                        result_nodes=local_points,
                        adjacent_result_tetrahedra=tuple(adjacent),
                        outward_normals=normals,
                        areas_m2=areas,
                    )
                )
    return sorted(surfaces, key=lambda item: item.name)


def _ordered_split_surface_entities(
    entities: list[tuple[int, np.ndarray, np.ndarray]],
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Order a 2x2 aperture grid by row then column despite centroid noise."""

    if len(entities) != 4:
        return sorted(entities, key=lambda item: (float(item[2][1]), float(item[2][0])))
    by_y = sorted(entities, key=lambda item: float(item[2][1]))
    lower_row = sorted(by_y[:2], key=lambda item: float(item[2][0]))
    upper_row = sorted(by_y[2:], key=lambda item: float(item[2][0]))
    return lower_row + upper_row


def _selected_cells(mesh: meshio.Mesh, cell_types: set[str], selected_tags: set[int], width: int) -> np.ndarray:
    physical = mesh.cell_data.get("gmsh:physical")
    if physical is None or len(physical) != len(mesh.cells):
        raise ValueError("FEM mesh cells do not contain aligned gmsh:physical tags.")
    selected = []
    for block, raw_tags in zip(mesh.cells, physical, strict=True):
        if block.type not in cell_types:
            continue
        cells = np.asarray(block.data, dtype=np.int64)[:, :width]
        tags = np.asarray(raw_tags, dtype=np.int64)
        mask = np.isin(tags, tuple(selected_tags))
        if np.any(mask):
            selected.append(cells[mask])
    return np.vstack(selected) if selected else np.empty((0, width), dtype=np.int64)


def _selected_cells_with_entities(
    mesh: meshio.Mesh,
    cell_types: set[str],
    selected_tags: set[int],
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    physical = mesh.cell_data.get("gmsh:physical")
    geometrical = mesh.cell_data.get("gmsh:geometrical")
    if physical is None or geometrical is None:
        raise ValueError("FEM mesh cells require Gmsh physical and geometrical tags.")
    selected_cells = []
    selected_entities = []
    for block, raw_physical, raw_geometrical in zip(
        mesh.cells,
        physical,
        geometrical,
        strict=True,
    ):
        if block.type not in cell_types:
            continue
        cells = np.asarray(block.data, dtype=np.int64)[:, :width]
        tags = np.asarray(raw_physical, dtype=np.int64)
        mask = np.isin(tags, tuple(selected_tags))
        if np.any(mask):
            selected_cells.append(cells[mask])
            selected_entities.append(np.asarray(raw_geometrical, dtype=np.int64)[mask])
    if not selected_cells:
        return np.empty((0, width), dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.vstack(selected_cells), np.concatenate(selected_entities)


def _sampled_ceiling(results: list[dict[str, Any]]) -> float | None:
    ceiling = None
    for item in results:
        if not item["passed"]:
            break
        ceiling = float(item["frequency_hz"])
    return ceiling


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = np.asarray(values)[order]
    cumulative = np.cumsum(np.asarray(weights)[order])
    return float(sorted_values[np.searchsorted(cumulative, quantile * cumulative[-1], side="left")])


def _complex_dict(value: complex) -> dict[str, float]:
    number = complex(value)
    return {
        "real": float(number.real),
        "imag": float(number.imag),
        "magnitude": float(abs(number)),
        "phase_deg": float(np.degrees(np.angle(number))),
    }


def _complex_from_dict(value: dict[str, float]) -> complex:
    return complex(float(value["real"]), float(value["imag"]))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_source_meshes(manifest: dict[str, Any]) -> None:
    meshes = manifest.get("meshes")
    if meshes is None:
        raise ValueError(
            "This legacy headless result does not record source mesh hashes; "
            "tagged-surface reconstruction cannot be verified safely. Re-run the solve."
        )
    for mesh in meshes:
        path = Path(str(mesh["file"])).resolve()
        if not path.is_file():
            raise ValueError(f"Source FEM mesh is unavailable for tagged-surface reconstruction: {path}")
        expected = str(mesh.get("sha256", ""))
        actual = _sha256(path)
        if expected and actual != expected:
            raise ValueError(
                "Source FEM mesh changed after this solve; tagged-surface reconstruction "
                f"would be unsafe ({path}, expected {expected}, found {actual})."
            )


__all__ = [
    "FEMConvergenceGates",
    "FEMValidationGates",
    "compare_fem_validation_reports",
    "evaluate_fem_run",
    "frequency_mesh_resolution",
    "surface_pressure_metrics",
    "tetrahedron_edge_statistics_m",
    "tetrahedral_shape_gradients",
    "write_fem_validation_report",
]
