"""
Simulates radiation of a freestanding horn loudspeaker
using the Boundary Element Method (BEM) via the Bempp-cl library.
Provides the normalized SPL on a spherical surface around the horn
using a Fibonacci sphere distribution of sampling points.
Provides the real + imaginary acoustic impedance as well.
Only the throat surface tag is treated as driven; all other mesh surface
tags are treated as rigid boundaries.
Frequency sweeps are solved serially inside this module.
"""

import os
import time
from dataclasses import dataclass
from typing import Tuple
import bempp_cl.api
import meshio
import numpy as np
import warnings
from pyopencl import CompilerWarning
warnings.filterwarnings("ignore", category=CompilerWarning)


# ==========================================
# Configuration
# ==========================================
@dataclass
class SimulationConfig:
    mesh_file: str
    sound_speed: float = 343.0      # m/s
    rho: float = 1.21               # kg/m^3
    distance: float = 2.0           # meters
    eval_point_count: int = 6000    # fibonacci sphere points. 6000 points gives an angular resolution of 2.5° ±0.04°
    freq_min: float = 200.0
    freq_max: float = 20000.0
    freq_count: int = 32
    tag_throat: int = 1             # Physical tag index for the driven compression-driver/throat surface.
    scale_factor: float = 0.001     # Mesh scaling
    use_burton_miller: bool = True  # Enable Burton-Miller formulation

    # BEMPP Device Configuration
    bempp_cl.api.BOUNDARY_OPERATOR_DEVICE_TYPE = "cpu"
    bempp_cl.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "cpu"
    bempp_cl.api.DEFAULT_PRECISION = 'single'

# Global instance for easy configuration editing
CONFIG = SimulationConfig(
    mesh_file=os.path.join("horn_outputs", "meshes", "02_Sym_Linear_Conical.msh")
)

# ==========================================
# Solver Class
# ==========================================
class HornBEMSolver:
    def __init__(self, config: SimulationConfig):
        self.cfg = config

        print(f"Loading mesh: {self.cfg.mesh_file}...")
        self.grid, self.physical_tags = self._load_mesh()
        
        # Setup Spaces
        # P1: Continuous linear elements (for Pressure)
        # DP0: Discontinuous constant elements (for Velocity/Flux)
        self.p1_space = bempp_cl.api.function_space(self.grid, "P", 1)
        self.dp0_space = bempp_cl.api.function_space(self.grid, "DP", 0)
        
        # Pre-compute Geometry info
        self._setup_driver_geometry()
        self._setup_evaluation_points()
        
        # Pre-compute Identity Operator (Frequency Independent)
        self.lhs_identity = bempp_cl.api.operators.boundary.sparse.identity(
            self.p1_space, self.p1_space, self.p1_space
        )
        self.rhs_identity = bempp_cl.api.operators.boundary.sparse.identity(
            self.dp0_space, self.p1_space, self.p1_space
        )

        # Create Unit Velocity Excitation (to scale later)
        self.unit_velocity_fun = self._create_unit_velocity()

    def _load_mesh(self) -> Tuple[bempp_cl.api.Grid, np.ndarray]:
        """Loads mesh and extracts physical tags."""
        mesh_data = meshio.read(self.cfg.mesh_file)
        print(f"Scale factor applied: {self.cfg.scale_factor}")
        vertices = mesh_data.points * self.cfg.scale_factor
        
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

        grid = bempp_cl.api.Grid(vertices.T, elements.T)
        return grid, physical_tags

    def _setup_driver_geometry(self):
        # Identify the driven throat elements. All remaining tagged surfaces are rigid.
        # In DP0, DOFs map 1:1 to elements.
        self.driver_dofs = [
            i for i in range(self.dp0_space.global_dof_count)
            if self.physical_tags[i] == self.cfg.tag_throat
        ]

        if len(self.driver_dofs) == 0:
            raise ValueError(
                f"No throat elements found for tag_throat={self.cfg.tag_throat}. "
                "Check mesh physical tags."
            )

        self.rigid_boundary_dofs = [
            i for i in range(self.dp0_space.global_dof_count)
            if self.physical_tags[i] != self.cfg.tag_throat
        ]
        self.rigid_boundary_tags = np.unique(self.physical_tags[self.physical_tags != self.cfg.tag_throat])

        # Geometry for impedance integration.
        self.throat_element_areas = self.grid.volumes[self.driver_dofs]
        self.throat_vertex_indices = self.grid.elements[:, self.driver_dofs]
        print(
            f"Driven throat tag {self.cfg.tag_throat} identified with {len(self.driver_dofs)} elements. "
            f"Rigid boundaries identified with {len(self.rigid_boundary_dofs)} elements across tags "
            f"{self.rigid_boundary_tags.tolist()}."
        )

    def _create_unit_velocity(self):
        """Creates a normal velocity boundary condition with magnitude 1.0 on the throat."""
        coeffs = np.zeros(self.dp0_space.global_dof_count, dtype=np.complex128)
        coeffs[self.driver_dofs] = 1.0
        return bempp_cl.api.GridFunction(self.dp0_space, coefficients=coeffs)

    def _setup_evaluation_points(self):
        """Generates evaluation points on a Fibonacci sphere."""
        n_points = int(self.cfg.eval_point_count)
        if n_points <= 0:
            raise ValueError("eval_point_count must be positive.")

        i = np.arange(n_points, dtype=float)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        z = 1.0 - (2.0 * i + 1.0) / n_points
        r = np.sqrt(1.0 - z * z)
        phi = i * golden_angle
        x = r * np.cos(phi)
        y = r * np.sin(phi)

        r_dist = self.cfg.distance
        self.eval_points = r_dist * np.vstack([x, y, z])
        self.theta_polar = np.arccos(z)
        self.phi_azimuth = np.mod(np.arctan2(y, x), 2.0 * np.pi)
        self.r_distance = np.full(n_points, r_dist)

    def solve_sweep(self) -> Tuple[list, np.ndarray]:
        frequencies = np.logspace(
            np.log10(self.cfg.freq_min),
            np.log10(self.cfg.freq_max),
            self.cfg.freq_count
        )

        print(f"Starting solver: {len(frequencies)} frequencies.")

        results_sphere = []
        results_imp = []
        for i, freq in enumerate(frequencies):
            res_s, res_z = self._solve_single_frequency(freq)
            results_sphere.append((freq, res_s))
            results_imp.append(res_z)
            print(f"[{i+1}/{len(frequencies)}] {freq:.1f} Hz")

        results_imp = np.asarray(results_imp, dtype=float)
        return results_sphere, results_imp

    def _solve_single_frequency(self, freq):
        omega = 2 * np.pi * freq
        k = omega / self.cfg.sound_speed
        
        # 1. Update Boundary Conditions
        # v = 1 m/s. Use normal velocity on the throat only.
        # Convert to Neumann data: q = i * rho * omega * v
        velocity_fun = self.unit_velocity_fun
        neumann_fun = 1j * self.cfg.rho * omega * velocity_fun

        # 2. Assemble Operators
        dlp = bempp_cl.api.operators.boundary.helmholtz.double_layer(
            self.p1_space, self.p1_space, self.p1_space, k
        )
        slp = bempp_cl.api.operators.boundary.helmholtz.single_layer(
            self.dp0_space, self.p1_space, self.p1_space, k
        )

        # 3. Formulate LHS and RHS
        if self.cfg.use_burton_miller:  # Use Burton-Miller at higher frequencies to avoid resonances
            hyp = bempp_cl.api.operators.boundary.helmholtz.hypersingular(
                self.p1_space, self.p1_space, self.p1_space, k
            )
            adlp = bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer(
                self.dp0_space, self.p1_space, self.p1_space, k
            )
            # Exterior Neumann, Burton-Miller (BEMPP sign conventions)
            coupling = 1j / k
            lhs = 0.5 * self.lhs_identity - dlp - coupling * -hyp
            rhs = (-slp - coupling * (adlp + 0.5 * self.rhs_identity)) * neumann_fun
        else:
            # Exterior Neumann (classical)
            lhs = dlp - 0.5 * self.lhs_identity
            rhs = slp * neumann_fun
        
        # velocity_fun.plot(transformation='abs')

        # 4. Solve System
        dirichlet_fun, info = bempp_cl.api.linalg.gmres(lhs, rhs, tol=1E-3)
        if info != 0:
            print(f"  Warning: Solver did not converge at {freq:.1f}Hz")

        # 5. Post-Processing
        z_data = self._calculate_impedance(freq, dirichlet_fun)
        spl = self._evaluate_field(self.eval_points, k, dirichlet_fun, neumann_fun, omega)
        spl_norm = self._normalize_spl_to_on_axis(spl)
        
        return spl_norm, z_data

    def _calculate_impedance(self, freq, dirichlet_fun):
        # Pressure at vertices of throat elements
        # dirichlet_fun.coefficients matches P1 DOF (vertices)
        # throat_vertex_indices is shape (3, num_elements), flatten it
        vertex_indices_flat = self.throat_vertex_indices.flatten()
        
        # Ensure indices are within bounds
        valid_indices = vertex_indices_flat[vertex_indices_flat < len(dirichlet_fun.coefficients)]
        
        if len(valid_indices) == 0:
            print(f"Warning: No valid vertex indices found at {freq:.1f}Hz")
            return [freq, 0.0, 0.0]
        
        p_at_vertices = dirichlet_fun.coefficients[valid_indices]
        
        # Reshape back to (3, num_valid_vertices) for averaging
        num_complete_elements = len(valid_indices) // 3
        if num_complete_elements > 0:
            p_at_vertices = p_at_vertices[:num_complete_elements * 3].reshape(3, num_complete_elements)
            p_avg = np.mean(p_at_vertices, axis=0)
            
            # Use only the valid elements' areas
            throat_areas_valid = self.throat_element_areas[:num_complete_elements]
            
            # Force = Integral(p dS) ~ sum(p_avg * area)
            total_force = np.sum(p_avg * throat_areas_valid) * 10
        else:
            print(f"Warning: Could not extract complete elements at {freq:.1f}Hz")
            return [freq, 0.0, 0.0]
        
        # Z = Force / Velocity (v=1)
        return [freq, np.real(total_force)/2, -np.imag(total_force)/2]

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

    def _normalize_spl_to_on_axis(self, spl):
        on_axis_idx = int(np.argmin(self.theta_polar))
        return spl - spl[on_axis_idx]

    def build_output_bundle(self, sphere_results, impedance_results):
        freq_hz = np.asarray([freq for freq, _ in sphere_results], dtype=float)
        spl_norm = np.vstack([spl for _, spl in sphere_results]).astype(float, copy=False)

        return {
            "freq_hz": freq_hz,
            "r_distance_m": self.r_distance.astype(float, copy=False),
            "theta_polar_rad": self.theta_polar.astype(float, copy=False),
            "phi_azimuth_rad": self.phi_azimuth.astype(float, copy=False),
            "spl_norm": spl_norm,
            "impedance_real": impedance_results[:, 1],
            "impedance_imag": impedance_results[:, 2],
        }

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    t_start = time.time()
    solver = HornBEMSolver(CONFIG)
    sphere_results, impedance_results = solver.solve_sweep()

    output_bundle = solver.build_output_bundle(sphere_results, impedance_results)
    np.savez("pressure_data_raw.npz", **output_bundle)
    print("Saved raw pressure data")
    
    print(f"Total Analysis Time: {time.time() - t_start:.2f}s")
    print("Analysis Complete.")
