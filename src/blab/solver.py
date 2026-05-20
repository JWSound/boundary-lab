"""
Simulates loudspeaker radiation using the Boundary Element Method (BEM)
via the Bempp-cl library.

- Supports one or more configured surface meshes
- Applies prescribed radiator drive conditions on tagged source surfaces
- Outputs normalized horizontal and vertical polar SPL around the device
- Outputs per-radiator real and imaginary acoustic impedance data
"""

import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
import bempp_cl.api
import meshio
import numpy as np
from scipy import signal
import warnings
from blab.config import (
    CrossoverConfig,
    MeshConfig,
    RadiatorConfig,
    SimulationConfig,
    load_external_config,
)
from blab.defaults import EXAMPLE_CLEAN_MESH_PATH
from pyopencl import CompilerWarning
warnings.filterwarnings("ignore", category=CompilerWarning)

bempp_cl.api.BOUNDARY_OPERATOR_DEVICE_TYPE = "cpu"
bempp_cl.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "cpu"
bempp_cl.api.DEFAULT_PRECISION = "single"


@dataclass
class RadiatorGeometry:
    config: RadiatorConfig
    driver_dofs: np.ndarray
    element_areas: np.ndarray
    p1_dofs: np.ndarray


# Global instance for easy configuration editing
CONFIG = SimulationConfig(mesh_file=str(EXAMPLE_CLEAN_MESH_PATH))


def _build_arg_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Run a BEM frequency sweep on a loudspeaker mesh.")
    parser.add_argument(
        "mesh_file",
        nargs="?",
        default=CONFIG.mesh_file,
        help="Path to the input mesh file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional TOML config file describing mesh and radiator drive settings",
    )
    parser.add_argument(
        "--output-npz",
        default=CONFIG.output_npz,
        help="Path for the solver output NPZ file",
    )
    parser.add_argument("--output-npz-base-path", dest="output_npz", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--freq-min",
        type=float,
        default=CONFIG.freq_min,
        help="Minimum frequency in Hz",
    )
    parser.add_argument(
        "--freq-max",
        type=float,
        default=CONFIG.freq_max,
        help="Maximum frequency in Hz",
    )
    parser.add_argument(
        "--freq-count",
        type=int,
        default=CONFIG.freq_count,
        help="Number of frequency points in the sweep",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=CONFIG.step_size,
        help="Angular step for polar evaluation in degrees",
    )
    parser.add_argument("--polar-angle-step-deg", dest="step_size", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--min-angle",
        type=float,
        default=CONFIG.min_angle,
        help="Minimum polar angle in degrees",
    )
    parser.add_argument("--polar-angle-min-deg", dest="min_angle", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-angle",
        type=float,
        default=CONFIG.max_angle,
        help="Maximum polar angle in degrees",
    )
    parser.add_argument("--polar-angle-max-deg", dest="max_angle", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--axial-offset",
        type=float,
        default=CONFIG.axial_offset,
        help="Shift the polar evaluation origin along +Z in meters",
    )
    parser.add_argument("--observation-axial-offset-m", dest="axial_offset", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--workers",
        type=int,
        default=CONFIG.workers,
        help="Number of worker processes to use for the frequency sweep",
    )
    parser.add_argument(
        "--gmres-tol",
        type=float,
        default=CONFIG.gmres_tolerance,
        help="GMRES convergence tolerance",
    )
    parser.add_argument(
        "--spherical-sampling",
        action="store_true",
        default=CONFIG.spherical_sampling_enabled,
        help="Evaluate a Fibonacci sphere of observation points for balloon plotting",
    )
    parser.add_argument(
        "--spherical-sampling-points",
        type=int,
        default=CONFIG.spherical_sampling_points,
        help="Number of Fibonacci sphere observation points",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> SimulationConfig:
    meshes, radiators = load_external_config(args.config)
    return SimulationConfig(
        mesh_file=args.mesh_file,
        sound_speed=CONFIG.sound_speed,
        rho=CONFIG.rho,
        distance=CONFIG.distance,
        axial_offset=args.axial_offset,
        step_size=args.step_size,
        min_angle=args.min_angle,
        max_angle=args.max_angle,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        freq_count=args.freq_count,
        tag_throat=CONFIG.tag_throat,
        meshes=meshes,
        radiators=radiators,
        scale_factor=CONFIG.scale_factor,
        use_burton_miller=CONFIG.use_burton_miller,
        gmres_tolerance=args.gmres_tol,
        workers=args.workers,
        spherical_sampling_enabled=args.spherical_sampling,
        spherical_sampling_points=args.spherical_sampling_points,
        output_npz=args.output_npz,
    )


def _split_frequencies_evenly(frequencies: np.ndarray, worker_count: int) -> List[np.ndarray]:
    if worker_count <= 1 or len(frequencies) == 0:
        return [frequencies]

    return [chunk for chunk in np.array_split(frequencies, worker_count) if len(chunk) > 0]


def build_fibonacci_sphere_observation_points(
    point_count: int,
    distance_m: float,
    axial_offset_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate approximately equal-area spherical observation points."""
    n_points = int(point_count)
    if n_points <= 0:
        raise ValueError("spherical_sampling_points must be positive.")
    if distance_m <= 0:
        raise ValueError("distance_m must be positive.")

    indices = np.arange(n_points, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - (2.0 * indices + 1.0) / n_points
    xy_radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    phi = indices * golden_angle
    x = xy_radius * np.cos(phi)
    y = xy_radius * np.sin(phi)

    points = float(distance_m) * np.vstack([x, y, z])
    points[2, :] += float(axial_offset_m)
    theta_polar = np.arccos(np.clip(z, -1.0, 1.0))
    phi_azimuth = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    r_distance = np.full(n_points, float(distance_m), dtype=np.float32)
    return points, theta_polar.astype(np.float32), phi_azimuth.astype(np.float32), r_distance


def _solve_frequency_chunk(config: SimulationConfig, frequencies: Sequence[float]):
    solver = HornBEMSolver(config)
    return solver.solve_frequencies(np.asarray(frequencies, dtype=float), show_progress=False)

# ==========================================
# Solver Class
# ==========================================
class HornBEMSolver:
    def __init__(self, config: SimulationConfig):
        self.cfg = config

        self.grid, self.physical_tags, self.element_mesh_ids, self.mesh_names = self._load_meshes()
        
        # Setup Spaces
        # P1: Continuous linear elements (for Pressure)
        # DP0: Discontinuous constant elements (for Velocity/Flux)
        self.p1_space = bempp_cl.api.function_space(self.grid, "P", 1)
        self.dp0_space = bempp_cl.api.function_space(self.grid, "DP", 0)
        
        # Pre-compute Geometry info
        self._setup_driver_geometry()
        self._setup_polar_evaluation_points()
        self._setup_spherical_evaluation_points()
        
        # Pre-compute Identity Operator (Frequency Independent)
        self.lhs_identity = bempp_cl.api.operators.boundary.sparse.identity(
            self.p1_space, self.p1_space, self.p1_space
        )
        self.rhs_identity = bempp_cl.api.operators.boundary.sparse.identity(
            self.dp0_space, self.p1_space, self.p1_space
        )

        self.radiator_names = tuple(r.config.name for r in self.radiator_geometries)

    def _resolved_mesh_configs(self) -> tuple[MeshConfig, ...]:
        if self.cfg.meshes:
            return self.cfg.meshes

        return (
            MeshConfig(
                name="mesh",
                file=self.cfg.mesh_file,
                scale_factor=self.cfg.scale_factor,
            ),
        )

    def _load_meshes(self) -> Tuple[bempp_cl.api.Grid, np.ndarray, np.ndarray, tuple[str, ...]]:
        vertices_parts = []
        element_parts = []
        physical_tag_parts = []
        element_mesh_id_parts = []
        mesh_names = []
        vertex_offset = 0

        for mesh_id, mesh_cfg in enumerate(self._resolved_mesh_configs()):
            print(f"Loading mesh '{mesh_cfg.name}': {mesh_cfg.file}...")
            vertices, elements, physical_tags = self._read_mesh_geometry(mesh_cfg)
            vertices_parts.append(vertices)
            element_parts.append(elements + vertex_offset)
            physical_tag_parts.append(physical_tags)
            element_mesh_id_parts.append(np.full(elements.shape[0], mesh_id, dtype=np.int32))
            mesh_names.append(mesh_cfg.name)
            vertex_offset += vertices.shape[0]
            print(
                f"Mesh '{mesh_cfg.name}' loaded with {vertices.shape[0]} vertices "
                f"and {elements.shape[0]} triangles."
            )

        vertices_all = np.vstack(vertices_parts)
        elements_all = np.vstack(element_parts)
        physical_tags_all = np.concatenate(physical_tag_parts)
        element_mesh_ids = np.concatenate(element_mesh_id_parts)
        grid = bempp_cl.api.Grid(vertices_all.T, elements_all.T)
        return grid, physical_tags_all, element_mesh_ids, tuple(mesh_names)

    def _read_mesh_geometry(self, mesh_cfg: MeshConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mesh_data = meshio.read(mesh_cfg.file)
        scale_factor = self.cfg.scale_factor if mesh_cfg.scale_factor is None else mesh_cfg.scale_factor
        vertices = mesh_data.points * scale_factor
        vertices = vertices + np.asarray(mesh_cfg.translation_m, dtype=float)

        # Handle meshio cell key variations
        if 'triangle' in mesh_data.cells_dict:
            elements = mesh_data.cells_dict['triangle']
            tri_key = 'triangle'
        elif 'triangle3' in mesh_data.cells_dict:
            elements = mesh_data.cells_dict['triangle3']
            tri_key = 'triangle3'
        else:
            raise ValueError("No triangular elements found in mesh.")

        physical_tags = None
        for key in mesh_data.cell_data_dict:
            if 'gmsh:physical' in key and tri_key in mesh_data.cell_data_dict[key]:
                physical_tags = mesh_data.cell_data_dict[key][tri_key]
                break
        
        if physical_tags is None:
            raise ValueError("No physical tags found in mesh.")

        return vertices, elements, np.asarray(physical_tags)

    def _setup_driver_geometry(self):
        radiator_configs = self.cfg.radiators or (
            RadiatorConfig(name="throat", tag=self.cfg.tag_throat, mesh=self.mesh_names[0]),
        )
        self._validate_radiator_configs(radiator_configs)

        self.radiator_geometries = []
        driven_mask = np.zeros(self.dp0_space.global_dof_count, dtype=bool)

        for radiator in radiator_configs:
            mesh_id = self._resolve_radiator_mesh_id(radiator)
            # In DP0, DOFs map 1:1 to elements.
            driver_dofs = np.asarray(
                [
                    i for i in range(self.dp0_space.global_dof_count)
                    if self.physical_tags[i] == radiator.tag and self.element_mesh_ids[i] == mesh_id
                ],
                dtype=int,
            )

            if driver_dofs.size == 0:
                raise ValueError(
                    f"No elements found for radiator '{radiator.name}' tag={radiator.tag}. "
                    "Check mesh physical tags."
                )

            self.radiator_geometries.append(
                RadiatorGeometry(
                    config=radiator,
                    driver_dofs=driver_dofs,
                    element_areas=self.grid.volumes[driver_dofs],
                    p1_dofs=self.p1_space.local2global[driver_dofs],
                )
            )
            driven_mask[driver_dofs] = True

        self.enclosure_dofs = [
            i for i in range(self.dp0_space.global_dof_count)
            if not driven_mask[i]
        ]

        for radiator in self.radiator_geometries:
            cfg = radiator.config
            print(
                f"Radiator '{cfg.name}' mesh '{self._radiator_mesh_name(cfg)}' tag {cfg.tag}: "
                f"{len(radiator.driver_dofs)} elements, "
                f"level {cfg.level_db:g} dB, polarity {cfg.polarity}, delay {cfg.delay_ms:g} ms."
            )
        print(f"Rigid/unassigned surface: {len(self.enclosure_dofs)} elements.")

    def _validate_radiator_configs(self, radiators: Sequence[RadiatorConfig]) -> None:
        names = set()
        mesh_name_set = set(self.mesh_names)
        for radiator in radiators:
            if radiator.name in names:
                raise ValueError(f"Duplicate radiator name: {radiator.name}")
            names.add(radiator.name)
            if radiator.mesh is not None and radiator.mesh not in mesh_name_set:
                raise ValueError(
                    f"Radiator '{radiator.name}' references unknown mesh '{radiator.mesh}'."
                )
            if radiator.mesh is None and len(self.mesh_names) > 1:
                raise ValueError(
                    f"Radiator '{radiator.name}' must specify 'mesh' when multiple meshes are configured."
                )
            if radiator.polarity not in (-1, 1):
                raise ValueError(f"Radiator '{radiator.name}' polarity must be +1 or -1.")

            for crossover in self._radiator_crossovers(radiator):
                self._validate_crossover_config(radiator.name, crossover)

    @staticmethod
    def _radiator_crossovers(radiator: RadiatorConfig) -> tuple[CrossoverConfig, ...]:
        crossovers = tuple(
            crossover
            for crossover in (radiator.hpf, radiator.lpf)
            if crossover.type.lower() != "none"
        )
        if crossovers:
            return crossovers
        return (radiator.crossover,)

    @staticmethod
    def _validate_crossover_config(radiator_name: str, crossover: CrossoverConfig) -> None:
        crossover_type = crossover.type.lower()
        if crossover_type not in ("none", "lowpass", "highpass"):
            raise ValueError(
                f"Radiator '{radiator_name}' crossover type must be none, lowpass, or highpass."
            )
        if crossover_type == "none":
            return
        if crossover.frequency_hz is None or crossover.frequency_hz <= 0:
            raise ValueError(f"Radiator '{radiator_name}' crossover frequency_hz must be > 0.")
        if crossover.filter not in ("butterworth", "linkwitz_riley"):
            raise ValueError(
                f"Radiator '{radiator_name}' crossover filter must be butterworth or linkwitz_riley."
            )
        if crossover.order not in (1, 2, 4):
            raise ValueError(f"Radiator '{radiator_name}' crossover order must be 1, 2, or 4.")
        if crossover.filter == "linkwitz_riley" and crossover.order not in (2, 4):
            raise ValueError(f"Radiator '{radiator_name}' Linkwitz-Riley order must be 2 or 4.")

    def _resolve_radiator_mesh_id(self, radiator: RadiatorConfig) -> int:
        mesh_name = self._radiator_mesh_name(radiator)
        return self.mesh_names.index(mesh_name)

    def _radiator_mesh_name(self, radiator: RadiatorConfig) -> str:
        return radiator.mesh or self.mesh_names[0]

    def _create_velocity(self, freq: float) -> tuple[bempp_cl.api.GridFunction, np.ndarray]:
        coeffs = np.zeros(self.dp0_space.global_dof_count, dtype=np.complex128)
        drives = np.empty(len(self.radiator_geometries), dtype=np.complex128)

        for i, radiator in enumerate(self.radiator_geometries):
            drive = self._radiator_drive(radiator.config, freq)
            drives[i] = drive
            coeffs[radiator.driver_dofs] = drive

        return bempp_cl.api.GridFunction(self.dp0_space, coefficients=coeffs), drives

    def _radiator_drive(self, radiator: RadiatorConfig, freq: float) -> complex:
        omega = 2.0 * np.pi * freq
        level = 10.0 ** (radiator.level_db / 20.0)
        delay = np.exp(-1j * omega * (radiator.delay_ms / 1000.0))
        crossover = 1.0 + 0.0j
        for crossover_config in HornBEMSolver._radiator_crossovers(radiator):
            crossover *= HornBEMSolver._crossover_response(self, crossover_config, freq)
        return complex(level * radiator.polarity * delay * crossover)

    def _crossover_response(self, crossover: CrossoverConfig, freq: float) -> complex:
        crossover_type = crossover.type.lower()
        if crossover_type == "none":
            return 1.0 + 0.0j

        filter_name = crossover.filter.lower()
        if filter_name == "linkwitz_riley":
            section_order = crossover.order // 2
            section = HornBEMSolver._butterworth_response(crossover_type, section_order, crossover.frequency_hz, freq)
            return section * section

        return HornBEMSolver._butterworth_response(crossover_type, crossover.order, crossover.frequency_hz, freq)

    @staticmethod
    def _butterworth_response(crossover_type: str, order: int, cutoff_hz: float, freq: float) -> complex:
        b, a = signal.butter(
            order,
            2.0 * np.pi * cutoff_hz,
            btype="lowpass" if crossover_type == "lowpass" else "highpass",
            analog=True,
        )
        _, h = signal.freqs(b, a, worN=[2.0 * np.pi * freq])
        return complex(h[0])

    def _create_unit_velocity(self):
        # Create a normal velocity boundary condition with magnitude 1.0 on every radiator.
        coeffs = np.zeros(self.dp0_space.global_dof_count, dtype=np.complex128)
        for radiator in self.radiator_geometries:
            coeffs[radiator.driver_dofs] = 1.0
        return bempp_cl.api.GridFunction(self.dp0_space, coefficients=coeffs)

    def _setup_polar_evaluation_points(self):
        #Generate horizontal and vertical polar evaluation points.
        step = float(self.cfg.step_size)
        if step <= 0:
            raise ValueError("step_size must be positive.")

        angle_min = float(self.cfg.min_angle)
        angle_max = float(self.cfg.max_angle)
        if angle_min < -180.0 or angle_max > 180.0:
            raise ValueError("polar angle range must stay within [-180, 180] degrees.")
        if angle_max < angle_min:
            raise ValueError("max_angle must be >= min_angle.")
        if not (angle_min <= 0.0 <= angle_max):
            raise ValueError("polar angle range must include 0 degrees for on-axis normalization.")

        self.polar_angles_deg = np.arange(angle_min, angle_max + 0.5 * step, step, dtype=np.float32)
        self.polar_angles_deg = np.clip(self.polar_angles_deg, angle_min, angle_max)
        angles_rad = np.deg2rad(self.polar_angles_deg.astype(float))

        x_h = np.sin(angles_rad)
        y_h = np.zeros_like(x_h)
        z_h = np.cos(angles_rad)

        x_v = np.zeros_like(angles_rad)
        y_v = np.sin(angles_rad)
        z_v = np.cos(angles_rad)

        r_dist = float(self.cfg.distance)
        axial_offset_m = float(self.cfg.axial_offset)
        axial_shift = np.array([[0.0], [0.0], [axial_offset_m]], dtype=float)

        self.horizontal_eval_points = r_dist * np.vstack([x_h, y_h, z_h]) + axial_shift
        self.vertical_eval_points = r_dist * np.vstack([x_v, y_v, z_v]) + axial_shift
        self.on_axis_idx = int(np.argmin(np.abs(self.polar_angles_deg)))

    def _setup_spherical_evaluation_points(self) -> None:
        self.sphere_eval_points = None
        self.sphere_theta_polar_rad = None
        self.sphere_phi_azimuth_rad = None
        self.sphere_r_distance_m = None
        if not self.cfg.spherical_sampling_enabled:
            return

        (
            self.sphere_eval_points,
            self.sphere_theta_polar_rad,
            self.sphere_phi_azimuth_rad,
            self.sphere_r_distance_m,
        ) = build_fibonacci_sphere_observation_points(
            self.cfg.spherical_sampling_points,
            float(self.cfg.distance),
            float(self.cfg.axial_offset),
        )

    @property
    def spherical_sampling_enabled(self) -> bool:
        return self.sphere_eval_points is not None

    @property
    def sphere_metadata(self) -> dict[str, np.ndarray] | None:
        if not self.spherical_sampling_enabled:
            return None
        return {
            "r_distance_m": self.sphere_r_distance_m,
            "theta_polar_rad": self.sphere_theta_polar_rad,
            "phi_azimuth_rad": self.sphere_phi_azimuth_rad,
        }

    def solve_sweep(self) -> Tuple[list, np.ndarray]:
        frequencies = np.logspace(
            np.log10(self.cfg.freq_min),
            np.log10(self.cfg.freq_max),
            self.cfg.freq_count
        )

        worker_count = self._resolve_worker_count(len(frequencies))
        print(
            f"Starting solver: {len(frequencies)} frequencies "
            f"using {worker_count} worker{'s' if worker_count != 1 else ''}."
        )

        if worker_count == 1:
            return self.solve_frequencies(frequencies, show_progress=True)

        return self._solve_sweep_parallel(frequencies, worker_count)

    def solve_frequencies(self, frequencies: Sequence[float], show_progress: bool = True) -> Tuple[list, np.ndarray]:
        frequencies = np.asarray(frequencies, dtype=float)

        results_polar = []
        results_imp = []
        for i, freq in enumerate(frequencies):
            res_h, res_v, res_z, raw_h, raw_v, sphere_spl = self._solve_single_frequency(freq)
            results_polar.append((freq, res_h, res_v, raw_h, raw_v, sphere_spl))
            results_imp.append(res_z)
            if show_progress:
                print(f"[{i+1}/{len(frequencies)}] {freq:.1f} Hz")

        imp_matrix = np.stack(results_imp, axis=1).astype(np.float32, copy=False)
        return results_polar, imp_matrix

    def solve_frequency(self, frequency_hz: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Solve one frequency using this already-initialized solver instance."""
        horizontal_spl, vertical_spl, impedance, raw_horizontal_spl, raw_vertical_spl, sphere_spl = self._solve_single_frequency(
            float(frequency_hz)
        )
        return (
            float(frequency_hz),
            horizontal_spl,
            vertical_spl,
            impedance,
            raw_horizontal_spl,
            raw_vertical_spl,
            sphere_spl,
        )

    def solve_frequencies_stream(
        self,
        frequencies: Sequence[float],
        *,
        stop_requested=None,
        show_progress: bool = False,
    ):
        """Yield one completed frequency at a time while keeping solver setup warm.

        ``stop_requested`` may be a callable returning True. Cancellation is checked
        between frequencies; an in-flight BEM solve is allowed to finish cleanly.
        """
        frequencies = np.asarray(frequencies, dtype=float)

        for i, freq in enumerate(frequencies):
            if stop_requested is not None and stop_requested():
                break
            result = self.solve_frequency(float(freq))
            if show_progress:
                print(f"[{i+1}/{len(frequencies)}] {freq:.1f} Hz")
            yield result

    def _resolve_worker_count(self, frequency_count: int) -> int:
        if self.cfg.workers < 1:
            raise ValueError("workers must be >= 1.")

        return min(self.cfg.workers, max(1, frequency_count), os.cpu_count() or 1)

    def _solve_sweep_parallel(self, frequencies: np.ndarray, worker_count: int) -> Tuple[list, np.ndarray]:
        chunks = _split_frequencies_evenly(frequencies, worker_count)
        ctx = mp.get_context("spawn")
        chunk_results = {}
        completed = 0

        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as executor:
            futures = {
                executor.submit(_solve_frequency_chunk, self.cfg, chunk.tolist()): index
                for index, chunk in enumerate(chunks)
            }

            for future in as_completed(futures):
                index = futures[future]
                polar_chunk, imp_chunk = future.result()
                chunk_results[index] = (polar_chunk, imp_chunk)
                completed += len(polar_chunk)
                print(f"[{completed}/{len(frequencies)}] completed worker chunk {index + 1}/{len(chunks)}")

        polar_results = []
        imp_results = []
        for index in range(len(chunks)):
            polar_chunk, imp_chunk = chunk_results[index]
            polar_results.extend(polar_chunk)
            imp_results.append(imp_chunk)

        imp_matrix = np.concatenate(imp_results, axis=1).astype(np.float32, copy=False)
        return polar_results, imp_matrix

    def _solve_single_frequency(self, freq):
        omega = 2 * np.pi * freq
        k = omega / self.cfg.sound_speed
        
        # 1. Update Boundary Conditions
        velocity_fun, radiator_drives = self._create_velocity(freq)
        neumann_fun = 1j * self.cfg.rho * omega * velocity_fun

        # 2. Assemble Operators
        dlp = bempp_cl.api.operators.boundary.helmholtz.double_layer(
            self.p1_space, self.p1_space, self.p1_space, k
        )
        slp = bempp_cl.api.operators.boundary.helmholtz.single_layer(
            self.dp0_space, self.p1_space, self.p1_space, k
        )

        # 3. Formulate LHS and RHS
        if self.cfg.use_burton_miller:
            hyp = bempp_cl.api.operators.boundary.helmholtz.hypersingular(
                self.p1_space, self.p1_space, self.p1_space, k
            )
            adlp = bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer(
                self.dp0_space, self.p1_space, self.p1_space, k
            )
            # Exterior Neumann, Burton-Miller (BEMPP sign conventions)
            # Note that BEMPP negates the hypersingular operator
            coupling = 1j / k
            lhs = 0.5 * self.lhs_identity - dlp - coupling * -hyp
            rhs = (-slp - coupling * (adlp + 0.5 * self.rhs_identity)) * neumann_fun
        else:
            # Exterior Neumann (classical)
            lhs = dlp - 0.5 * self.lhs_identity
            rhs = slp * neumann_fun

        # 4. Solve System
        dirichlet_fun, info = bempp_cl.api.linalg.gmres(lhs, rhs, tol=self.cfg.gmres_tolerance)
        if info != 0:
            print(f"  Warning: Solver did not converge at {freq:.1f}Hz")

        # 5. Post-Processing
        z_data = self._calculate_impedance(dirichlet_fun, radiator_drives)
        horizontal_spl = self._evaluate_field(self.horizontal_eval_points, k, dirichlet_fun, neumann_fun, omega)
        vertical_spl = self._evaluate_field(self.vertical_eval_points, k, dirichlet_fun, neumann_fun, omega)
        horizontal_spl_norm, vertical_spl_norm = self._normalize_polar_to_on_axis(horizontal_spl, vertical_spl)
        sphere_spl_norm = None
        if self.spherical_sampling_enabled:
            sphere_spl = self._evaluate_field(self.sphere_eval_points, k, dirichlet_fun, neumann_fun, omega)
            sphere_spl_norm = sphere_spl - horizontal_spl[self.on_axis_idx]
        
        return horizontal_spl_norm, vertical_spl_norm, z_data, horizontal_spl, vertical_spl, sphere_spl_norm

    def _calculate_impedance(self, dirichlet_fun, radiator_drives):
        z_data = np.empty((len(self.radiator_geometries), 2), dtype=np.float32)

        for i, radiator in enumerate(self.radiator_geometries):
            # Pressure at local P1 dofs for each radiator element.
            # Do not index with raw mesh vertex ids: P1 global dof numbering may differ.
            p_at_vertices = dirichlet_fun.coefficients[radiator.p1_dofs]
            p_avg = np.mean(p_at_vertices, axis=1)

            # Force = Integral(p dS) ~ sum(p_avg * area).
            total_force = np.sum(p_avg * radiator.element_areas) * 10
            drive = radiator_drives[i]
            if np.isclose(np.abs(drive), 0.0):
                z_data[i, :] = np.nan
                continue

            z_complex = total_force / drive
            z_data[i, 0] = np.real(z_complex) / 2
            z_data[i, 1] = -np.imag(z_complex) / 2

        return z_data

    def _evaluate_field(self, points, k, dirichlet_fun, neumann_fun, omega):
        slp_pot = bempp_cl.api.operators.potential.helmholtz.single_layer(
            self.dp0_space, points, k, device_interface="opencl"
        )
        dlp_pot = bempp_cl.api.operators.potential.helmholtz.double_layer(
            self.p1_space, points, k, device_interface="opencl"
        )

        p_field = (dlp_pot * dirichlet_fun - slp_pot * neumann_fun).ravel()
        
        # Convert to SPL
        # Ref pressure = 20e-6 Pa
        return 20 * np.log10(np.abs(p_field) / 20e-6)

    def _normalize_polar_to_on_axis(self, horizontal_spl, vertical_spl):
        on_axis_ref = horizontal_spl[self.on_axis_idx]
        return horizontal_spl - on_axis_ref, vertical_spl - on_axis_ref

    def save_outputs(self, polar_results, imp_matrix):
        output_path = Path(self.cfg.output_npz)
        if output_path.suffix.lower() != ".npz":
            output_path = Path(f"{output_path}.npz")

        freqs = np.array([freq for freq, *_ in polar_results], dtype=np.float32)
        horizontal_spl = np.vstack([item[1] for item in polar_results]).astype(np.float32, copy=False)
        vertical_spl = np.vstack([item[2] for item in polar_results]).astype(np.float32, copy=False)
        horizontal_raw_spl = np.vstack([item[3] for item in polar_results]).astype(np.float32, copy=False)
        vertical_raw_spl = np.vstack([item[4] for item in polar_results]).astype(np.float32, copy=False)
        z_real = imp_matrix[:, :, 0].astype(np.float32, copy=False)
        z_imag = imp_matrix[:, :, 1].astype(np.float32, copy=False)
        bundle = {
            "freq_hz": freqs,
            "polar_angle_deg": self.polar_angles_deg.astype(np.float32, copy=False),
            "horizontal_spl_db": horizontal_raw_spl,
            "vertical_spl_db": vertical_raw_spl,
            "horizontal_spl_norm_db": horizontal_spl,
            "vertical_spl_norm_db": vertical_spl,
            "impedance_freq_hz": freqs,
            "impedance_radiator_names": np.asarray(self.radiator_names),
            "impedance_real": z_real,
            "impedance_imag": z_imag,
            "observation_axial_offset_m": np.float32(self.cfg.axial_offset),
        }
        if self.spherical_sampling_enabled and all(len(item) > 5 and item[5] is not None for item in polar_results):
            bundle.update(
                sphere_r_distance_m=self.sphere_r_distance_m.astype(np.float32, copy=False),
                sphere_theta_polar_rad=self.sphere_theta_polar_rad.astype(np.float32, copy=False),
                sphere_phi_azimuth_rad=self.sphere_phi_azimuth_rad.astype(np.float32, copy=False),
                sphere_spl_norm_db=np.vstack([item[5] for item in polar_results]).astype(np.float32, copy=False),
            )

        np.savez_compressed(output_path, **bundle)
        print(f"Saved {output_path}")


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> None:
    mp.freeze_support()
    args = _build_arg_parser(prog=prog).parse_args(argv)
    config = _config_from_args(args)
    t_start = time.time()
    solver = HornBEMSolver(config)
    polar_results, imp_matrix = solver.solve_sweep()

    # Save Results
    solver.save_outputs(polar_results, imp_matrix)
    
    print(f"Total Analysis Time: {time.time() - t_start:.2f}s")
    print("Analysis Complete.")


if __name__ == "__main__":
    main()
