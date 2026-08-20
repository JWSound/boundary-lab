"""Headless project loading, physical-system solving, and result persistence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from blab.live import build_log_frequencies
from blab.observation_planes import observation_planes_from_payload
from blab.physical_model import PhysicalSolveKind, PhysicalSystem, physical_system_from_dict
from blab.solve_results import (
    BEM_BOUNDARY_DOMAIN_ID,
    BEM_BOUNDARY_NEUMANN_ID,
    BEM_BOUNDARY_PRESSURE_ID,
    FEM_NODAL_PRESSURE_ID,
    FEM_VOLUME_DOMAIN_ID,
    HORIZONTAL_POLAR_DOMAIN_ID,
    SPHERE_DOMAIN_ID,
    VERTICAL_POLAR_DOMAIN_ID,
    ResultDomain,
    bem_boundary_result_domain,
    fem_volume_result_domain,
)
from blab.solvers.beat_engine_backend import DEFAULT_BEAT_ENGINE_CUDA_PROJECT
from blab.solvers.coupled_backend import PhysicalSystemProductionBackend, validate_system_capabilities
from blab.solvers.registry import normalize_backend_id
from blab.system_contract import (
    OutputRequest,
    SystemFrequencyResult,
    compiled_system_to_dict,
    validate_system_solve_request,
)
from blab.system_solve import (
    SystemUiSolveRequest,
    canonicalize_observation_result,
    prepare_system_ui_solve,
)
from blab.ui.project_io import PROJECT_SCHEMA_VERSION, read_project_file
from blab.ui.project_state import ProjectPreferencesState

HEADLESS_REQUEST_VERSION = 1
HEADLESS_RESULT_VERSION = 2
SUPPORTED_RETAIN_VALUES = {
    "bem_boundary_pressure",
    "bem_boundary_neumann",
    "bem_boundary_traces",
    "fem_nodal_pressure",
}
HEADLESS_BACKEND_AUTO = "beat_auto"
HEADLESS_BACKEND_IDS = (
    HEADLESS_BACKEND_AUTO,
    "beat_cpu",
    "beat_cuda",
    "beat_rocm",
)


@dataclass(frozen=True)
class HeadlessProject:
    path: Path
    payload: dict[str, Any]
    physical_system: PhysicalSystem
    preferences: ProjectPreferencesState
    symmetry: str
    component_channel_by_id: dict[str, str]


@dataclass(frozen=True)
class PointProbe:
    id: str
    points_m: np.ndarray


@dataclass(frozen=True)
class HeadlessSolveSpec:
    frequencies_hz: tuple[float, ...] | None = None
    excitation_port_ids: tuple[str, ...] | None = None
    probes: tuple[PointProbe, ...] = ()
    retain: tuple[str, ...] = ()
    solver_options: dict[str, Any] | None = None
    include_project_observations: bool = True
    raw: dict[str, Any] | None = None


def load_headless_project(path: str | Path) -> HeadlessProject:
    """Load, migrate, and resolve a project without importing Qt."""

    project_path = Path(path).resolve()
    payload = read_project_file(project_path)
    raw_system = payload.get("physical_system")
    if not isinstance(raw_system, dict):
        raise ValueError("Headless solving requires a project with a physical_system object.")
    system = physical_system_from_dict(raw_system)
    preferences = ProjectPreferencesState.from_payload(payload.get("project_preferences"))
    if preferences is None:
        preferences = ProjectPreferencesState()
    symmetry = str(payload.get("symmetry", "off")).strip().lower()
    if symmetry not in {"off", "x", "xy"}:
        symmetry = "off"
    component_channels = payload.get("component_channel_by_id", {})
    if not isinstance(component_channels, dict):
        component_channels = {}
    return HeadlessProject(
        path=project_path,
        payload=payload,
        physical_system=system,
        preferences=preferences,
        symmetry=symmetry,
        component_channel_by_id={str(key): str(value) for key, value in component_channels.items()},
    )


def resolve_headless_backend(
    backend_id: str = HEADLESS_BACKEND_AUTO,
    *,
    julia_executable: str = "julia",
    cuda_probe_timeout_s: float = 30.0,
) -> str:
    """Resolve the headless BEAT backend, preferring functional CUDA in auto mode."""

    requested = str(backend_id or HEADLESS_BACKEND_AUTO).strip().lower()
    if requested in {"auto", "beat"}:
        requested = HEADLESS_BACKEND_AUTO
    elif requested != HEADLESS_BACKEND_AUTO:
        requested = normalize_backend_id(requested)
    if requested not in HEADLESS_BACKEND_IDS:
        raise ValueError(f"Unknown headless backend {backend_id!r}; expected " + ", ".join(HEADLESS_BACKEND_IDS))
    if requested != HEADLESS_BACKEND_AUTO:
        return requested
    if not _command_available("nvidia-smi") or not _command_available(julia_executable):
        return "beat_cpu"
    command = [
        julia_executable,
        "--startup-file=no",
        f"--project={DEFAULT_BEAT_ENGINE_CUDA_PROJECT}",
        "-e",
        "import CUDA; exit(CUDA.functional() ? 0 : 1)",
    ]
    try:
        probe = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=max(float(cuda_probe_timeout_s), 0.1),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "beat_cpu"
    return "beat_cuda" if probe.returncode == 0 else "beat_cpu"


def load_headless_solve_spec(path: str | Path | None) -> HeadlessSolveSpec:
    if path is None:
        return HeadlessSolveSpec(raw={"schema_version": HEADLESS_REQUEST_VERSION})
    request_path = Path(path)
    raw = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Headless solve request must contain a JSON object.")
    version = int(raw.get("schema_version", 0))
    if version != HEADLESS_REQUEST_VERSION:
        raise ValueError(
            f"Unsupported headless request schema_version {version}; expected {HEADLESS_REQUEST_VERSION}."
        )

    frequencies = _request_frequencies(raw)
    excitation_ids = raw.get("excitation_port_ids")
    if excitation_ids is not None:
        if not isinstance(excitation_ids, list) or not excitation_ids:
            raise ValueError("excitation_port_ids must be a non-empty array when provided.")
        excitation_ids = tuple(str(value) for value in excitation_ids)
        if len(excitation_ids) != len(set(excitation_ids)):
            raise ValueError("excitation_port_ids must be unique.")

    probes = _request_probes(raw.get("probes", []))
    retain_raw = raw.get("retain", [])
    if not isinstance(retain_raw, list):
        raise ValueError("retain must be an array.")
    retain = tuple(dict.fromkeys(str(value) for value in retain_raw))
    unsupported = sorted(set(retain) - SUPPORTED_RETAIN_VALUES)
    if unsupported:
        raise ValueError("Unsupported retained quantities: " + ", ".join(unsupported))
    solver_options = raw.get("solver_options", {})
    if not isinstance(solver_options, dict):
        raise ValueError("solver_options must be an object.")
    return HeadlessSolveSpec(
        frequencies_hz=frequencies,
        excitation_port_ids=excitation_ids,
        probes=probes,
        retain=retain,
        solver_options=dict(solver_options),
        include_project_observations=bool(raw.get("include_project_observations", True)),
        raw=raw,
    )


def prepare_headless_solve(
    project: HeadlessProject,
    spec: HeadlessSolveSpec,
    *,
    backend_id: str,
) -> SystemUiSolveRequest:
    """Build and validate a canonical system request from a project and overlay."""

    preferences = project.preferences
    sphere_angle_deg = min(max(float(preferences.balloon_angle_precision_deg), 0.5), 15.0)
    spherical_count = max(int(round(41253.0 / sphere_angle_deg**2)), 1)
    prepared = prepare_system_ui_solve(
        project.physical_system,
        freq_min_hz=preferences.freq_min_hz,
        freq_max_hz=preferences.freq_max_hz,
        freq_count=preferences.freq_count,
        observation_distance_m=preferences.polar_observation_distance_m,
        polar_angle_step_deg=preferences.polar_angle_step_deg,
        spherical_sampling_enabled=preferences.spherical_sampling_enabled,
        spherical_sampling_points=spherical_count,
        component_channel_by_id=project.component_channel_by_id,
        backend_id=backend_id,
        symmetry_mode=project.symmetry,
        observation_planes=observation_planes_from_payload(project.payload.get("observation_planes")),
    )
    if prepared.solve_kind == PhysicalSolveKind.INTERIOR_FEM and spec.probes:
        raise ValueError(
            "Interior FEM point probes are not yet supported; retain fem_nodal_pressure "
            "or use an Interior observation plane."
        )
    if prepared.solve_kind == PhysicalSolveKind.INTERIOR_FEM and set(spec.retain) & {
        "bem_boundary_pressure",
        "bem_boundary_neumann",
        "bem_boundary_traces",
    }:
        raise ValueError("Interior FEM solves cannot retain BEM boundary quantities.")
    request = prepared.request
    frequencies = spec.frequencies_hz
    if frequencies is None:
        frequencies = tuple(
            float(value)
            for value in build_log_frequencies(
                preferences.freq_min_hz,
                preferences.freq_max_hz,
                preferences.freq_count,
            )
        )

    outputs = list(request.outputs)
    domains = list(prepared.result_domains)
    if not spec.include_project_observations:
        outputs, domains = _without_project_observations(outputs, domains)
    outputs, domains = _with_probes(outputs, domains, spec.probes)
    outputs, domains = _with_retained_fields(request.compiled_system, outputs, domains, spec.retain, project.symmetry)

    excitation_ids = spec.excitation_port_ids or request.excitation_port_ids
    available_ids = {port.id for port in request.compiled_system.excitation_ports}
    unknown_ids = [value for value in excitation_ids if value not in available_ids]
    if unknown_ids:
        raise ValueError("Unknown excitation port ids: " + ", ".join(unknown_ids))
    options = dict(request.solver_options)
    option_overrides = dict(spec.solver_options or {})
    requested_symmetry = option_overrides.get("symmetry")
    if requested_symmetry is not None and str(requested_symmetry).strip().lower() != project.symmetry:
        raise ValueError("solver_options.symmetry must match the project's compiled symmetry mode.")
    options.update(option_overrides)
    if options.get("validation_diagnostics", False) and "static_condensation" not in option_overrides:
        options["static_condensation"] = False
    options["symmetry"] = project.symmetry
    request = replace(
        request,
        frequencies_hz=frequencies,
        excitation_port_ids=tuple(excitation_ids),
        outputs=tuple(outputs),
        solver_options=options,
    )
    validate_system_solve_request(request)
    validate_system_capabilities(request)
    return replace(prepared, request=request, result_domains=tuple(domains))


def validation_summary(
    project: HeadlessProject,
    prepared: SystemUiSolveRequest,
    *,
    backend_id: str,
) -> dict[str, Any]:
    request = prepared.request
    mesh_entries = _mesh_manifest_entries(request.compiled_system)
    return {
        "valid": True,
        "project": str(project.path),
        "project_schema_version": PROJECT_SCHEMA_VERSION,
        "system_id": request.compiled_system.id,
        "system_name": request.compiled_system.name,
        "solve_kind": prepared.solve_kind.value,
        "backend_id": backend_id,
        "symmetry": project.symmetry,
        "frequency_count": len(request.frequencies_hz),
        "frequency_min_hz": min(request.frequencies_hz),
        "frequency_max_hz": max(request.frequencies_hz),
        "excitation_port_ids": list(request.excitation_port_ids),
        "output_ids": [output.id for output in request.outputs],
        "meshes": mesh_entries,
        "assumptions": [
            {"status": item.status.value, "statement": item.statement}
            for item in request.compiled_system.assumptions
        ],
    }


def default_result_path(project_path: str | Path) -> Path:
    path = Path(project_path).resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = path.name.removesuffix(".blab.json").removesuffix(".json")
    return path.parent / "runs" / "headless" / f"{name}-{stamp}"


class HeadlessResultWriter:
    """Write independently readable, crash-tolerant per-frequency result files."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        project: HeadlessProject,
        prepared: SystemUiSolveRequest,
        backend_id: str,
        public_request: dict[str, Any],
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        if self.output_dir.exists():
            raise FileExistsError(f"Result directory already exists: {self.output_dir}")
        self.output_dir.mkdir(parents=True)
        self.frequency_dir = self.output_dir / "frequencies"
        self.frequency_dir.mkdir()
        shutil.copyfile(project.path, self.output_dir / "project.snapshot.blab.json")
        _write_json(self.output_dir / "request.json", public_request)
        _write_json(
            self.output_dir / "compiled-system.json",
            compiled_system_to_dict(prepared.request.compiled_system),
        )
        self._write_domains(prepared.result_domains)
        self.frequencies = np.asarray(prepared.request.frequencies_hz, dtype=float)
        self.completion = np.zeros(self.frequencies.size, dtype=bool)
        self.results: list[dict[str, Any] | None] = [None] * self.frequencies.size
        self.manifest: dict[str, Any] = {
            "schema": "boundary-lab-headless-result",
            "schema_version": HEADLESS_RESULT_VERSION,
            "status": "running",
            "started_at_utc": datetime.now(UTC).isoformat(),
            "project_file": "project.snapshot.blab.json",
            "project_sha256": _sha256(project.path),
            "compiled_system_file": "compiled-system.json",
            "request_file": "request.json",
            "domains_file": "domains.npz",
            "domains_metadata_file": "domains.json",
            "backend_id": backend_id,
            "solve_kind": prepared.solve_kind.value,
            "phasor_convention": "exp(-i omega t)",
            "frequencies_hz": self.frequencies.tolist(),
            "excitation_port_ids": list(prepared.request.excitation_port_ids),
            "solver_options": _json_safe(prepared.request.solver_options),
            "meshes": _mesh_manifest_entries(prepared.request.compiled_system),
            "completion_mask": self.completion.tolist(),
            "results": self.results,
        }
        self._flush_manifest()

    def write_result(self, result: SystemFrequencyResult) -> int:
        distances = np.abs(self.frequencies - float(result.freq_hz))
        index = int(np.argmin(distances))
        tolerance = max(1e-5, abs(float(result.freq_hz)) * 2e-6)
        if distances[index] > tolerance:
            raise ValueError(f"Solver returned unrequested frequency {result.freq_hz:g} Hz.")
        arrays: dict[str, np.ndarray] = {}
        quantities = []
        for quantity_index, quantity in enumerate(result.quantities):
            key = f"q{quantity_index:04d}"
            arrays[key] = np.asarray(quantity.values)
            quantities.append(
                {
                    "key": key,
                    "id": quantity.id,
                    "quantity": quantity.quantity,
                    "unit": quantity.unit,
                    "target_id": quantity.target_id,
                    "axes": list(quantity.axes),
                    "shape": list(arrays[key].shape),
                    "dtype": str(arrays[key].dtype),
                    "metadata": _json_safe(quantity.metadata),
                }
            )
        stem = f"{index:06d}"
        npz_path = self.frequency_dir / f"{stem}.npz"
        tmp_path = self.frequency_dir / f".{stem}.npz.tmp"
        with tmp_path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(tmp_path, npz_path)
        metadata = {
            "freq_hz": float(result.freq_hz),
            "excitation_port_ids": list(result.excitation_port_ids),
            "diagnostics": _json_safe(result.diagnostics),
            "arrays_file": npz_path.name,
            "quantities": quantities,
        }
        metadata_path = self.frequency_dir / f"{stem}.json"
        _write_json(metadata_path, metadata)
        self.completion[index] = True
        self.results[index] = {
            "freq_hz": float(result.freq_hz),
            "metadata_file": f"frequencies/{metadata_path.name}",
            "arrays_file": f"frequencies/{npz_path.name}",
        }
        self._flush_manifest()
        return index

    def finish(self, *, status: str, error: str | None = None) -> None:
        self.manifest["status"] = status
        self.manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
        if error:
            self.manifest["error"] = str(error)
        self._flush_manifest()

    def _write_domains(self, domains: Iterable[ResultDomain]) -> None:
        arrays: dict[str, np.ndarray] = {}
        metadata = []
        for domain_index, domain in enumerate(domains):
            coordinates = {}
            topology = {}
            for name, values in domain.coordinates.items():
                key = f"d{domain_index:04d}_c_{len(coordinates):04d}"
                arrays[key] = np.asarray(values)
                coordinates[name] = key
            for name, values in domain.topology.items():
                key = f"d{domain_index:04d}_t_{len(topology):04d}"
                arrays[key] = np.asarray(values)
                topology[name] = key
            metadata.append(
                {
                    "id": domain.id,
                    "kind": domain.kind,
                    "dimensions": list(domain.dimensions),
                    "coordinates": coordinates,
                    "topology": topology,
                    "metadata": _json_safe(domain.metadata),
                }
            )
        with (self.output_dir / "domains.npz").open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        _write_json(self.output_dir / "domains.json", {"domains": metadata})

    def _flush_manifest(self) -> None:
        self.manifest["completion_mask"] = self.completion.tolist()
        self.manifest["results"] = self.results
        _write_json(self.output_dir / "manifest.json", self.manifest)


def run_headless_solve(
    project: HeadlessProject,
    prepared: SystemUiSolveRequest,
    *,
    output_dir: str | Path,
    backend_id: str,
    public_request: dict[str, Any],
    julia_executable: str = "julia",
    julia_threads: str | int | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    emit = event_callback or (lambda _event: None)
    writer = HeadlessResultWriter(
        output_dir,
        project=project,
        prepared=prepared,
        backend_id=backend_id,
        public_request=public_request,
    )
    solved_count = 0
    emit(
        {
            "event": "started",
            "output": str(writer.output_dir),
            "backend_id": backend_id,
            "frequency_count": len(writer.frequencies),
        }
    )
    session = None
    try:
        backend = PhysicalSystemProductionBackend(
            bem_backend=backend_id.removeprefix("beat_"),
            julia_executable=julia_executable,
            julia_threads=julia_threads,
        )
        request = replace(
            prepared.request,
            status_callback=lambda message: emit({"event": "status", "message": str(message)}),
        )
        session = backend.create_system_session(request)
        for raw_result in session.solve_stream():
            result = canonicalize_observation_result(prepared, raw_result)
            index = writer.write_result(result)
            solved_count += 1
            emit(
                {
                    "event": "frequency_completed",
                    "index": index,
                    "freq_hz": float(result.freq_hz),
                    "solved_count": solved_count,
                    "frequency_count": len(writer.frequencies),
                }
            )
    except KeyboardInterrupt:
        if session is not None:
            session.stop()
        writer.finish(status="interrupted")
        emit({"event": "interrupted", "solved_count": solved_count})
        raise
    except Exception as exc:
        if session is not None:
            session.stop()
        writer.finish(status="failed", error=str(exc))
        emit({"event": "failed", "message": str(exc), "solved_count": solved_count})
        raise
    writer.finish(status="complete")
    summary = {
        "event": "completed",
        "output": str(writer.output_dir),
        "backend_id": backend_id,
        "solved_count": solved_count,
        "frequency_count": len(writer.frequencies),
    }
    emit(summary)
    return summary


def _request_frequencies(raw: dict[str, Any]) -> tuple[float, ...] | None:
    explicit = raw.get("frequencies_hz")
    sweep = raw.get("frequency_sweep")
    if explicit is not None and sweep is not None:
        raise ValueError("Specify frequencies_hz or frequency_sweep, not both.")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("frequencies_hz must be a non-empty array.")
        return _validated_frequencies(explicit)
    if sweep is None:
        return None
    if not isinstance(sweep, dict):
        raise ValueError("frequency_sweep must be an object.")
    first = float(sweep["min_hz"])
    second = float(sweep["max_hz"])
    minimum = min(first, second)
    maximum = max(first, second)
    count = int(sweep["count"])
    spacing = str(sweep.get("spacing", "log")).strip().lower()
    if count <= 0 or minimum <= 0.0 or maximum <= 0.0:
        raise ValueError("frequency_sweep values must be greater than zero.")
    if spacing == "log":
        values = np.geomspace(minimum, maximum, count)
    elif spacing == "linear":
        values = np.linspace(minimum, maximum, count)
    else:
        raise ValueError("frequency_sweep spacing must be 'log' or 'linear'.")
    return _validated_frequencies(values)


def _validated_frequencies(values: Iterable[Any]) -> tuple[float, ...]:
    frequencies = np.asarray(tuple(float(value) for value in values), dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("Frequencies must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("Frequencies must be finite and greater than zero.")
    if np.unique(frequencies).size != frequencies.size:
        raise ValueError("Frequencies must be unique.")
    return tuple(float(value) for value in frequencies)


def _request_probes(raw: Any) -> tuple[PointProbe, ...]:
    if not isinstance(raw, list):
        raise ValueError("probes must be an array.")
    probes = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each probe must be an object.")
        identifier = str(item.get("id", "")).strip()
        if not identifier or identifier in seen:
            raise ValueError("Probe ids must be non-empty and unique.")
        if str(item.get("coordinate_frame", "project")) != "project":
            raise ValueError(f"Probe '{identifier}' uses unsupported coordinate_frame; expected 'project'.")
        points = np.asarray(item.get("points_m"), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError(f"Probe '{identifier}' points_m must have shape (point, 3).")
        if not np.all(np.isfinite(points)):
            raise ValueError(f"Probe '{identifier}' contains non-finite coordinates.")
        seen.add(identifier)
        probes.append(PointProbe(identifier, points))
    return tuple(probes)


def _without_project_observations(
    outputs: list[OutputRequest],
    domains: list[ResultDomain],
) -> tuple[list[OutputRequest], list[ResultDomain]]:
    removed_ids = {HORIZONTAL_POLAR_DOMAIN_ID, VERTICAL_POLAR_DOMAIN_ID, SPHERE_DOMAIN_ID}
    retained_outputs = [output for output in outputs if output.id != "ui:exterior-pressure"]
    return retained_outputs, [domain for domain in domains if domain.id not in removed_ids]


def _with_probes(
    outputs: list[OutputRequest],
    domains: list[ResultDomain],
    probes: tuple[PointProbe, ...],
) -> tuple[list[OutputRequest], list[ResultDomain]]:
    if not probes:
        return outputs, domains
    compact_index = next((index for index, output in enumerate(outputs) if output.id == "ui:exterior-pressure"), None)
    if compact_index is None:
        compact_index = len(outputs)
        outputs.append(OutputRequest(id="ui:exterior-pressure", quantity="exterior_pressure", options={}))
    compact = outputs[compact_index]
    options = dict(compact.options)
    points = list(options.get("points_m", []))
    observation_domains = list(options.get("observation_domains", []))
    for probe in probes:
        domain_id = f"observation:probe:{probe.id}"
        if any(domain.id == domain_id for domain in domains):
            raise ValueError(f"Probe domain id collides with an existing result domain: {domain_id}")
        offset = len(points)
        points.extend(probe.points_m.tolist())
        observation_domains.append(
            {
                "id": domain_id,
                "quantity_id": f"acoustic:pressure:probe:{probe.id}",
                "offset": offset,
                "count": len(probe.points_m),
            }
        )
        domains.append(
            ResultDomain(
                id=domain_id,
                kind="point_probe",
                dimensions=("observation",),
                coordinates={"points_m": probe.points_m},
                metadata={"probe_id": probe.id, "coordinate_frame": "project"},
            )
        )
    options.update({"points_m": points, "observation_domains": observation_domains})
    outputs[compact_index] = replace(compact, options=options)
    return outputs, domains


def _with_retained_fields(
    compiled_system,
    outputs: list[OutputRequest],
    domains: list[ResultDomain],
    retain: tuple[str, ...],
    symmetry: str,
) -> tuple[list[OutputRequest], list[ResultDomain]]:
    output_ids = {output.id for output in outputs}
    domain_ids = {domain.id for domain in domains}
    retain_set = set(retain)
    if "bem_boundary_traces" in retain_set:
        retain_set.update({"bem_boundary_pressure", "bem_boundary_neumann"})
    if "bem_boundary_pressure" in retain_set and BEM_BOUNDARY_PRESSURE_ID not in output_ids:
        outputs.append(
            OutputRequest(
                id=BEM_BOUNDARY_PRESSURE_ID,
                quantity="bem_boundary_pressure",
                target_ids=(BEM_BOUNDARY_DOMAIN_ID,),
            )
        )
    if "bem_boundary_neumann" in retain_set and BEM_BOUNDARY_NEUMANN_ID not in output_ids:
        outputs.append(
            OutputRequest(
                id=BEM_BOUNDARY_NEUMANN_ID,
                quantity="bem_boundary_neumann",
                target_ids=(BEM_BOUNDARY_DOMAIN_ID,),
            )
        )
    if retain_set.intersection({"bem_boundary_pressure", "bem_boundary_neumann"}) and BEM_BOUNDARY_DOMAIN_ID not in domain_ids:
        domains.append(bem_boundary_result_domain(compiled_system, symmetry=symmetry))
    if "fem_nodal_pressure" in retain_set and FEM_NODAL_PRESSURE_ID not in output_ids:
        outputs.append(
            OutputRequest(
                id=FEM_NODAL_PRESSURE_ID,
                quantity="fem_nodal_pressure",
                target_ids=(FEM_VOLUME_DOMAIN_ID,),
            )
        )
        if FEM_VOLUME_DOMAIN_ID not in domain_ids:
            domains.append(fem_volume_result_domain(compiled_system, symmetry=symmetry))
    return outputs, domains


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mesh_manifest_entries(system) -> list[dict[str, Any]]:
    entries = []
    for mesh in system.meshes:
        path = Path(mesh.file).resolve()
        entries.append(
            {
                "id": mesh.id,
                "name": mesh.name,
                "purpose": mesh.purpose.value,
                "file": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return entries


def _command_available(command: str) -> bool:
    candidate = str(command).strip()
    if not candidate:
        return False
    path = Path(candidate)
    if path.is_absolute() or path.parent != Path("."):
        return path.is_file()
    return shutil.which(candidate) is not None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


__all__ = [
    "HEADLESS_REQUEST_VERSION",
    "HEADLESS_BACKEND_AUTO",
    "HEADLESS_BACKEND_IDS",
    "HeadlessProject",
    "HeadlessResultWriter",
    "HeadlessSolveSpec",
    "PointProbe",
    "default_result_path",
    "load_headless_project",
    "load_headless_solve_spec",
    "prepare_headless_solve",
    "resolve_headless_backend",
    "run_headless_solve",
    "validation_summary",
]
