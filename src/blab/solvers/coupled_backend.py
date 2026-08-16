"""Julia backend adapters for directly coupled FEM-BEM systems."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterator

from blab.acoustic_materials import (
    REGION_BULK_LOSS_FACTOR_KEY,
    WALL_IMPEDANCE_KEY,
    region_bulk_loss_factor,
    wall_impedance_parameters,
)
from blab.physical_model import (
    AcousticRegionKind,
    BoundaryKind,
    ComponentKind,
    ExcitationPortKind,
)
from blab.solvers.beat_engine_backend import (
    DEFAULT_BEAT_ENGINE_CUDA_PROJECT,
    BeatEngineWorkerProcess,
    _get_julia_worker,
)
from blab.system_contract import (
    SystemFrequencyResult,
    SystemSolveMetadata,
    SystemSolveRequest,
    system_frequency_result_from_dict,
    system_solve_request_to_dict,
)

DEFAULT_COUPLED_SOLVER_SCRIPT = Path(__file__).with_name("julia_local") / "coupled_solver.jl"
DEFAULT_COUPLED_CPU_PROJECT = DEFAULT_COUPLED_SOLVER_SCRIPT.parent
COUPLED_BEM_BACKENDS = {"cpu", "cuda"}
COUPLED_BOUNDARY_KINDS = {
    BoundaryKind.RIGID,
    BoundaryKind.MOVING,
    BoundaryKind.INTERFACE,
}
DEFAULT_TRANSDUCER_REFERENCE_VOLTAGE_V = 2.83
ELECTRODYNAMIC_REQUIRED_PARAMETERS = {
    "re_ohm",
    "le_h",
    "bl_n_per_a",
    "mmd_kg",
    "cms_m_per_n",
    "rms_n_s_per_m",
    "motion_axis",
}
ELECTRODYNAMIC_OPTIONAL_PARAMETERS = {
    "motion_profile",
    "boundary_motion_signs",
    "boundary_motion_weights",
    "symmetry_role",
    "surface_completion_factor",
    "physical_driver_orbit_count",
    "fractional_symmetry_axes",
}


class CoupledSession:
    def __init__(
        self,
        request: SystemSolveRequest,
        *,
        julia_executable: str = "julia",
        solver_script: str | Path = DEFAULT_COUPLED_SOLVER_SCRIPT,
        julia_project: str | Path | None = DEFAULT_COUPLED_CPU_PROJECT,
        julia_threads: str | int = "4",
        persistent_worker: bool = True,
    ):
        self.request = request
        self.julia_executable = julia_executable.strip() or "julia"
        self.solver_script = Path(solver_script).resolve()
        self.julia_project = None if julia_project is None else Path(julia_project).resolve()
        self.julia_threads = julia_threads
        self.persistent_worker = persistent_worker
        self._process: subprocess.Popen[str] | None = None
        self._worker: BeatEngineWorkerProcess | None = None
        self._stop = False
        self._cancel_path: Path | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._metadata = SystemSolveMetadata(
            system_id=request.compiled_system.id,
            assumptions=request.compiled_system.assumptions,
            excitation_port_ids=request.excitation_port_ids,
            available_quantity_ids=tuple(output.id for output in request.outputs),
        )

    @property
    def metadata(self) -> SystemSolveMetadata:
        return self._metadata

    def solve_stream(
        self,
        *,
        stop_requested: Callable[[], bool] | None = None,
    ) -> Iterator[SystemFrequencyResult]:
        if self._stop:
            return
        if self.persistent_worker:
            yield from self._solve_stream_persistent(stop_requested=stop_requested)
            return

        command = [self.julia_executable]
        if self.julia_project is not None:
            command.append(f"--project={self.julia_project}")
        command.append(str(self.solver_script))
        environment = os.environ.copy()
        environment.setdefault("JULIA_NUM_THREADS", str(self.julia_threads))
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=environment,
        )
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._stderr_thread = threading.Thread(
            target=self._capture_stderr,
            args=(self._process.stderr,),
            daemon=True,
        )
        self._stderr_thread.start()
        try:
            self._process.stdin.write(json.dumps(system_solve_request_to_dict(self.request)))
            self._process.stdin.close()
            for line in self._process.stdout:
                if self._stop or (stop_requested is not None and stop_requested()):
                    self.stop()
                    return
                text = line.strip()
                if not text:
                    continue
                yield system_frequency_result_from_dict(json.loads(text))
            return_code = self._process.wait()
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=2.0)
            if return_code != 0 and not self._stop:
                detail = "\n".join(self._stderr_lines).strip()
                raise RuntimeError(detail or f"Coupled Julia solver exited with status {return_code}.")
        finally:
            self._close_process()

    def _solve_stream_persistent(
        self,
        *,
        stop_requested: Callable[[], bool] | None,
    ) -> Iterator[SystemFrequencyResult]:
        self._worker = _get_julia_worker(
            julia_executable=self.julia_executable,
            solver_script=self.solver_script,
            julia_threads=self.julia_threads,
            julia_project=self.julia_project,
        )
        callback = self.request.status_callback
        with tempfile.TemporaryDirectory(prefix="blab-coupled-") as temp_dir:
            request_path = Path(temp_dir) / "request.json"
            self._cancel_path = Path(temp_dir) / "cancel"
            payload = system_solve_request_to_dict(self.request)
            payload["cancel_path"] = str(self._cancel_path)
            request_path.write_text(
                json.dumps(payload, separators=(",", ":")),
                encoding="utf-8",
            )
            try:
                if self._stop:
                    return
                events = self._worker.submit(request_path, status_callback=callback)
                for event in events:
                    if not self._stop and stop_requested is not None and stop_requested():
                        self.stop()
                    event_type = str(event.get("type", ""))
                    if event_type == "result":
                        if not self._stop:
                            yield system_frequency_result_from_dict(event["result"])
                    elif event_type == "status" and callback is not None:
                        callback(str(event.get("message", "")))
                    elif event_type == "failed":
                        raise RuntimeError(str(event.get("error", "Coupled Julia solver failed.")))
                    elif event_type in {"completed", "cancelled"}:
                        return
            except RuntimeError:
                if not self._stop:
                    raise
            finally:
                self._cancel_path = None

    def stop(self) -> None:
        self._stop = True
        worker = self._worker
        if worker is not None:
            cancel_path = self._cancel_path
            if cancel_path is not None:
                try:
                    cancel_path.write_text("cancel", encoding="utf-8")
                except OSError:
                    worker.terminate()
                    self._worker = None
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def _capture_stderr(self, stream) -> None:
        for line in stream:
            text = line.rstrip()
            if text:
                self._stderr_lines.append(text)
                callback = self.request.status_callback
                if callback is not None:
                    callback(text)

    def _close_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


class _CoupledBackend:
    def __init__(
        self,
        *,
        julia_executable: str = "julia",
        solver_script: str | Path = DEFAULT_COUPLED_SOLVER_SCRIPT,
        julia_project: str | Path | None = DEFAULT_COUPLED_CPU_PROJECT,
        julia_threads: str | int = "4",
        persistent_worker: bool = True,
        precision: str = "float64",
        bem_backend: str = "cpu",
    ):
        normalized_precision = str(precision).strip().lower()
        if normalized_precision not in {"float32", "float64"}:
            raise ValueError("Coupled precision must be float32 or float64.")
        normalized_bem_backend = str(bem_backend).strip().lower()
        if normalized_bem_backend not in COUPLED_BEM_BACKENDS:
            raise ValueError("Coupled BEM backend must be cpu or cuda.")
        self.julia_executable = julia_executable
        self.solver_script = Path(solver_script)
        self.julia_project = None if julia_project is None else Path(julia_project)
        self.julia_threads = julia_threads
        self.persistent_worker = persistent_worker
        self.precision = normalized_precision
        self.bem_backend = normalized_bem_backend

    def create_system_session(self, request: SystemSolveRequest) -> CoupledSession:
        solver_options = dict(request.solver_options)
        solver_options["precision"] = self.precision
        solver_options["bem_backend"] = self.bem_backend
        solver_options["transducer_reference_voltage_v"] = DEFAULT_TRANSDUCER_REFERENCE_VOLTAGE_V
        typed_request = replace(request, solver_options=solver_options)
        validate_system_capabilities(typed_request)
        return CoupledSession(
            typed_request,
            julia_executable=self.julia_executable,
            solver_script=self.solver_script,
            julia_project=self.julia_project,
            julia_threads=self.julia_threads,
            persistent_worker=self.persistent_worker,
        )


def validate_coupled_capabilities(request: SystemSolveRequest) -> None:
    """Reject physical-model features that the current coupled backend cannot solve."""

    system = request.compiled_system
    if request.solver_options.get("static_condensation", False) and request.solver_options.get(
        "validation_diagnostics", False
    ):
        raise ValueError("FEM static condensation cannot be combined with full-matrix validation diagnostics.")
    requested_symmetry = str(request.solver_options.get("symmetry", "off")).strip().lower()
    symmetry_mode = "off" if requested_symmetry in {"", "none"} else requested_symmetry
    symmetry_factors = {"off": 1, "x": 2, "xy": 4}
    if symmetry_mode not in symmetry_factors:
        raise ValueError(f"Unsupported coupled symmetry mode {requested_symmetry!r}; expected off, x, or xy.")
    bounded_regions = [region for region in system.regions if region.kind == AcousticRegionKind.BOUNDED_AIR]
    unbounded_regions = [region for region in system.regions if region.kind == AcousticRegionKind.UNBOUNDED_AIR]
    if not bounded_regions:
        raise ValueError("Coupled solver requires at least one bounded acoustic region.")
    if len(unbounded_regions) != 1:
        raise ValueError("Coupled solver requires exactly one unbounded acoustic region.")
    if any(len(region.mesh_ids) != 1 for region in bounded_regions):
        raise ValueError("Each coupled bounded acoustic region must currently reference exactly one FEM mesh.")
    reference_medium = bounded_regions[0]
    mismatched_media = [
        region.id
        for region in system.regions
        if not math.isclose(
            region.sound_speed_m_per_s,
            reference_medium.sound_speed_m_per_s,
            rel_tol=1e-9,
            abs_tol=0.0,
        )
        or not math.isclose(
            region.density_kg_per_m3,
            reference_medium.density_kg_per_m3,
            rel_tol=1e-9,
            abs_tol=0.0,
        )
    ]
    if mismatched_media:
        raise ValueError(
            "Coupled solver currently requires one shared sound speed and density across all regions; "
            "mismatched: " + ", ".join(mismatched_media)
        )
    unsupported_boundaries = [
        boundary.id for boundary in system.boundaries if boundary.kind not in COUPLED_BOUNDARY_KINDS
    ]
    if unsupported_boundaries:
        raise ValueError(
            "Coupled solver does not support the boundary assignments used by: " + ", ".join(unsupported_boundaries)
        )
    region_by_id = {region.id: region for region in system.regions}
    parameterized_boundaries = [
        boundary.id
        for boundary in system.boundaries
        if boundary.parameters and set(boundary.parameters) != {WALL_IMPEDANCE_KEY}
    ]
    if parameterized_boundaries:
        raise ValueError(
            "Coupled solver does not support the boundary parameters used by: " + ", ".join(parameterized_boundaries)
        )
    for boundary in system.boundaries:
        treatment = wall_impedance_parameters(boundary.parameters)
        if treatment is None:
            continue
        region = region_by_id[boundary.region_id]
        if boundary.kind != BoundaryKind.RIGID or region.kind != AcousticRegionKind.BOUNDED_AIR:
            raise ValueError(f"Wall impedance boundary '{boundary.id}' must be rigid and belong to bounded air.")
    for region in system.regions:
        unknown_loss_keys = set(region.loss_model) - {REGION_BULK_LOSS_FACTOR_KEY}
        if unknown_loss_keys:
            raise ValueError(
                f"Coupled solver does not support acoustic loss parameters on '{region.id}': "
                + ", ".join(sorted(unknown_loss_keys))
            )
        loss_factor = region_bulk_loss_factor(region.loss_model)
        if region.kind != AcousticRegionKind.BOUNDED_AIR and loss_factor != 0.0:
            raise ValueError(f"FEM bulk loss on '{region.id}' requires a bounded acoustic region.")
    supported_component_kinds = {
        ComponentKind.IDEAL_VELOCITY_SOURCE,
        ComponentKind.ELECTRODYNAMIC_TRANSDUCER,
    }
    unsupported_components = [
        component.id for component in system.components if component.kind not in supported_component_kinds
    ]
    if unsupported_components:
        raise ValueError(
            "Coupled solver supports prescribed-velocity and linear electrodynamic components; unsupported: "
            + ", ".join(unsupported_components)
        )
    for component in system.components:
        _validate_boundary_motion_weights(component)
        if component.kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER:
            _validate_electrodynamic_component(
                component,
                symmetry_factor=symmetry_factors[symmetry_mode],
                active_symmetry_axes=(() if symmetry_mode == "off" else ("x",) if symmetry_mode == "x" else ("x", "y")),
            )
            continue
        unsupported_parameters = set(component.parameters) - {
            "motion_profile",
            "boundary_motion_weights",
        }
        if unsupported_parameters:
            raise ValueError(
                f"Coupled solver does not support component parameters on '{component.id}': "
                + ", ".join(sorted(unsupported_parameters))
            )
        motion_profile = component.parameters.get("motion_profile", "uniform")
        if motion_profile != "uniform":
            raise ValueError(
                f"Coupled solver supports only uniform prescribed motion; component "
                f"'{component.id}' requests {motion_profile!r}."
            )
    components_by_id = {component.id: component for component in system.components}
    unsupported_ports = []
    for port in system.excitation_ports:
        component = components_by_id[port.component_id]
        expected = (
            ExcitationPortKind.VOLTAGE
            if component.kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER
            else ExcitationPortKind.NORMAL_VELOCITY
        )
        if port.kind != expected:
            unsupported_ports.append(port.id)
    if unsupported_ports:
        raise ValueError(
            "Coupled solver received excitation ports incompatible with their components: "
            + ", ".join(unsupported_ports)
        )


def validate_system_capabilities(request: SystemSolveRequest) -> None:
    """Dispatch validation according to the acoustic-region topology."""

    bounded_regions = [
        region for region in request.compiled_system.regions if region.kind == AcousticRegionKind.BOUNDED_AIR
    ]
    if bounded_regions:
        validate_coupled_capabilities(request)
        return
    validate_exterior_capabilities(request)


def validate_exterior_capabilities(request: SystemSolveRequest) -> None:
    """Reject physical-model features unsupported by the BEAT BEM-only branch."""

    system = request.compiled_system
    unbounded_regions = [region for region in system.regions if region.kind == AcousticRegionKind.UNBOUNDED_AIR]
    if len(unbounded_regions) != 1:
        raise ValueError("Exterior solver requires exactly one unbounded acoustic region.")
    if system.interfaces:
        raise ValueError("Exterior-only systems cannot contain FEM-BEM interfaces.")
    requested_symmetry = str(request.solver_options.get("symmetry", "off")).strip().lower()
    if requested_symmetry not in {"off", "x", "xy"}:
        raise ValueError(f"Unsupported exterior symmetry mode {requested_symmetry!r}; expected off, x, or xy.")
    unsupported_boundaries = [
        boundary.id for boundary in system.boundaries if boundary.kind not in {BoundaryKind.RIGID, BoundaryKind.MOVING}
    ]
    if unsupported_boundaries:
        raise ValueError(
            "Exterior solver supports rigid and prescribed-moving boundaries only: " + ", ".join(unsupported_boundaries)
        )
    parameterized_boundaries = [boundary.id for boundary in system.boundaries if boundary.parameters]
    if parameterized_boundaries:
        raise ValueError(
            "Exterior solver does not support boundary parameters on: " + ", ".join(parameterized_boundaries)
        )
    unsupported_components = [
        component.id for component in system.components if component.kind != ComponentKind.IDEAL_VELOCITY_SOURCE
    ]
    if unsupported_components:
        raise ValueError(
            "Exterior solver supports prescribed-velocity components only: " + ", ".join(unsupported_components)
        )
    for component in system.components:
        _validate_boundary_motion_weights(component)
        unsupported_parameters = set(component.parameters) - {"motion_profile", "boundary_motion_weights"}
        if unsupported_parameters:
            raise ValueError(
                f"Exterior solver does not support component parameters on '{component.id}': "
                + ", ".join(sorted(unsupported_parameters))
            )
        if component.parameters.get("motion_profile", "uniform") != "uniform":
            raise ValueError(f"Exterior component '{component.id}' must use uniform prescribed motion.")
    components_by_id = {component.id: component for component in system.components}
    incompatible_ports = [
        port.id
        for port in system.excitation_ports
        if port.kind != ExcitationPortKind.NORMAL_VELOCITY
        or components_by_id[port.component_id].kind != ComponentKind.IDEAL_VELOCITY_SOURCE
    ]
    if incompatible_ports:
        raise ValueError("Exterior solver received incompatible excitation ports: " + ", ".join(incompatible_ports))
    lossy_regions = [region.id for region in system.regions if region.loss_model]
    if lossy_regions:
        raise ValueError("Exterior solver does not support region loss models on: " + ", ".join(lossy_regions))


def _validate_boundary_motion_weights(component) -> None:
    raw_weights = component.parameters.get("boundary_motion_weights", {})
    if not isinstance(raw_weights, dict):
        raise ValueError(f"Component '{component.id}' boundary_motion_weights must be an object.")
    unknown = sorted(set(raw_weights) - set(component.boundary_ids))
    if unknown:
        raise ValueError(
            f"Component '{component.id}' has motion weights for unrelated boundaries: " + ", ".join(unknown)
        )
    for boundary_id, value in raw_weights.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(
                f"Component '{component.id}' motion weight for '{boundary_id}' must be finite and greater than zero."
            )


def _validate_electrodynamic_component(
    component,
    *,
    symmetry_factor: int,
    active_symmetry_axes: tuple[str, ...],
) -> None:
    parameters = component.parameters
    parameter_names = set(parameters)
    missing = sorted(ELECTRODYNAMIC_REQUIRED_PARAMETERS - parameter_names)
    if missing:
        raise ValueError(
            f"Electrodynamic component '{component.id}' is missing required parameters: " + ", ".join(missing)
        )
    unsupported = sorted(parameter_names - ELECTRODYNAMIC_REQUIRED_PARAMETERS - ELECTRODYNAMIC_OPTIONAL_PARAMETERS)
    if unsupported:
        raise ValueError(
            f"Electrodynamic component '{component.id}' has unsupported parameters: " + ", ".join(unsupported)
        )

    positive = ("re_ohm", "bl_n_per_a", "mmd_kg", "cms_m_per_n")
    nonnegative = ("le_h", "rms_n_s_per_m")
    for name in (*positive, *nonnegative):
        value = parameters[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Electrodynamic component '{component.id}' parameter '{name}' must be a finite number.")
        if name in positive and float(value) <= 0.0:
            raise ValueError(f"Electrodynamic component '{component.id}' parameter '{name}' must be greater than zero.")
        if name in nonnegative and float(value) < 0.0:
            raise ValueError(f"Electrodynamic component '{component.id}' parameter '{name}' must not be negative.")

    raw_axis = parameters["motion_axis"]
    if (
        not isinstance(raw_axis, (list, tuple))
        or len(raw_axis) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in raw_axis
        )
        or math.sqrt(sum(float(value) ** 2 for value in raw_axis)) <= 0.0
    ):
        raise ValueError(
            f"Electrodynamic component '{component.id}' motion_axis must contain three finite "
            "values with nonzero length."
        )

    motion_profile = parameters.get("motion_profile", "rigid_translation")
    if motion_profile != "rigid_translation":
        raise ValueError(
            f"Coupled solver supports only a rigid-translation piston motion profile; component "
            f"'{component.id}' requests {motion_profile!r}."
        )

    symmetry_role = parameters.get("symmetry_role", "complete_representative")
    if symmetry_role not in {"complete_representative", "fractional_driver"}:
        raise ValueError(
            f"Electrodynamic component '{component.id}' symmetry_role must be "
            "'complete_representative' or 'fractional_driver'."
        )
    completion_factor = parameters.get("surface_completion_factor", 1)
    orbit_count = parameters.get("physical_driver_orbit_count", 1)
    for name, value in (
        ("surface_completion_factor", completion_factor),
        ("physical_driver_orbit_count", orbit_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value not in {1, 2, 4}:
            raise ValueError(f"Electrodynamic component '{component.id}' {name} must be one of 1, 2, or 4.")
    expected_role = "fractional_driver" if completion_factor > 1 else "complete_representative"
    if symmetry_role != expected_role:
        raise ValueError(
            f"Electrodynamic component '{component.id}' symmetry_role {symmetry_role!r} "
            f"is inconsistent with surface_completion_factor={completion_factor}."
        )
    raw_fractional_axes = parameters.get("fractional_symmetry_axes", [])
    if (
        not isinstance(raw_fractional_axes, (list, tuple))
        or any(axis not in {"x", "y"} for axis in raw_fractional_axes)
        or len(raw_fractional_axes) != len(set(raw_fractional_axes))
    ):
        raise ValueError(
            f"Electrodynamic component '{component.id}' fractional_symmetry_axes must contain "
            "unique axis names chosen from 'x' and 'y'."
        )
    fractional_axes = tuple(str(axis) for axis in raw_fractional_axes)
    if any(axis not in active_symmetry_axes for axis in fractional_axes):
        raise ValueError(
            f"Electrodynamic component '{component.id}' fractional_symmetry_axes must be active "
            "in the selected global symmetry mode."
        )
    if completion_factor != 2 ** len(fractional_axes):
        raise ValueError(
            f"Electrodynamic component '{component.id}' surface_completion_factor must equal "
            "2 raised to the number of fractional_symmetry_axes."
        )
    axis_norm = math.sqrt(sum(float(value) ** 2 for value in raw_axis))
    normalized_axis = tuple(float(value) / axis_norm for value in raw_axis)
    axis_indices = {"x": 0, "y": 1}
    incompatible_axes = [axis for axis in fractional_axes if abs(normalized_axis[axis_indices[axis]]) > 1e-8]
    if incompatible_axes:
        raise ValueError(
            f"Electrodynamic component '{component.id}' motion_axis must lie in each symmetry "
            "plane that cuts the same physical driver; incompatible: " + ", ".join(incompatible_axes)
        )
    if completion_factor * orbit_count != symmetry_factor:
        raise ValueError(
            f"Electrodynamic component '{component.id}' represents "
            f"{completion_factor * orbit_count} symmetry images, but the selected symmetry "
            f"requires {symmetry_factor}."
        )

    raw_signs = parameters.get("boundary_motion_signs", {})
    if not isinstance(raw_signs, dict):
        raise ValueError(f"Electrodynamic component '{component.id}' boundary_motion_signs must be an object.")
    if not all(isinstance(boundary_id, str) for boundary_id in raw_signs):
        raise ValueError(f"Electrodynamic component '{component.id}' boundary_motion_signs keys must be boundary IDs.")
    component_boundaries = set(component.boundary_ids)
    unknown_boundaries = sorted(set(raw_signs) - component_boundaries)
    if unknown_boundaries:
        raise ValueError(
            f"Electrodynamic component '{component.id}' has motion signs for unrelated boundaries: "
            + ", ".join(unknown_boundaries)
        )
    for boundary_id, sign in raw_signs.items():
        if isinstance(sign, bool) or not isinstance(sign, (int, float)) or float(sign) not in {-1.0, 1.0}:
            raise ValueError(
                f"Electrodynamic component '{component.id}' motion sign for '{boundary_id}' must be -1 or +1."
            )


class CoupledReferenceBackend(_CoupledBackend):
    """Backend-only double-precision CPU correctness reference."""

    backend_id = "coupled_reference"
    label = "Coupled FEM-BEM Reference (Julia CPU FP64)"

    def __init__(
        self,
        *,
        julia_executable: str = "julia",
        solver_script: str | Path = DEFAULT_COUPLED_SOLVER_SCRIPT,
        julia_project: str | Path | None = DEFAULT_COUPLED_CPU_PROJECT,
        julia_threads: str | int = "4",
        persistent_worker: bool = True,
    ):
        super().__init__(
            julia_executable=julia_executable,
            solver_script=solver_script,
            julia_project=julia_project,
            julia_threads=julia_threads,
            persistent_worker=persistent_worker,
            precision="float64",
            bem_backend="cpu",
        )


class PhysicalSystemProductionBackend(_CoupledBackend):
    """FP32 physical-system backend for exterior and coupled BEAT solves."""

    backend_id = "coupled_production"
    label = "Coupled FEM-BEM (FP32)"

    def __init__(
        self,
        *,
        bem_backend: str,
        julia_executable: str = "julia",
        solver_script: str | Path = DEFAULT_COUPLED_SOLVER_SCRIPT,
        julia_threads: str | int | None = None,
        persistent_worker: bool = True,
    ):
        normalized_bem_backend = str(bem_backend).strip().lower()
        resolved_threads = (4 if normalized_bem_backend == "cuda" else 8) if julia_threads is None else julia_threads
        julia_project = (
            DEFAULT_BEAT_ENGINE_CUDA_PROJECT if normalized_bem_backend == "cuda" else DEFAULT_COUPLED_CPU_PROJECT
        )
        super().__init__(
            julia_executable=julia_executable,
            solver_script=solver_script,
            julia_project=julia_project,
            julia_threads=resolved_threads,
            persistent_worker=persistent_worker,
            precision="float32",
            bem_backend=normalized_bem_backend,
        )


CoupledProductionBackend = PhysicalSystemProductionBackend
