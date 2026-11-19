"""Sample an interior reporting plane from retained FEM pressure and topology."""

import numpy as np
from scipy.spatial import cKDTree

from blab.fem_fields import plane_flow_vectors, sample_pressure_velocity, tetrahedral_shape_gradients
from blab.observation_planes import ObservationPlaneType, sample_plane_points
from blab.phasor import convert_phasor
from blab.solve_results.derived import phase_deg, pressure_spl_db
from blab.solve_results.model import FEM_NODAL_PRESSURE_ID, FEM_VOLUME_DOMAIN_ID


def interior_plane(run, plane, hz, excitation, response, uncorrected):
    if plane.plane_type != ObservationPlaneType.INTERIOR:
        raise ValueError("Plane extraction currently supports interior FEM fields.")
    _, axes, nodal, meta = run.sample(
        hz, FEM_NODAL_PRESSURE_ID, excitation=excitation, response=response, uncorrected=uncorrected
    )
    if axes != ["fem_node"]:
        raise ValueError("Plane sampling requires one excitation or synthesized response.")
    if excitation:
        nodal = convert_phasor(nodal, run.manifest["phasor_convention"])
    domain = run.domains[FEM_VOLUME_DOMAIN_ID]
    points = run.domain_array(domain["id"], "points_m")
    cells = run.domain_array(domain["id"], "tetrahedra")
    full_cells = run.domain_array(domain["id"], "tetrahedra10") if "tetrahedra10" in domain["topology"] else cells
    densities = domain["metadata"].get("density_kg_per_m3")
    if densities is None:
        densities = [r["density_kg_per_m3"] for r in run.system["regions"] if r["kind"] == "bounded_air"]
    densities = np.asarray(densities, dtype=float)
    if "region_index" in domain["topology"]:
        density = densities[run.domain_array(domain["id"], "region_index", "topology")]
    elif densities.size == 1:
        density = np.full(len(cells), densities[0])
    else:
        raise ValueError("FEM topology lacks cell region indices.")
    gradients = tetrahedral_shape_gradients(points, cells)
    samples = sample_plane_points(plane, plane.sample_shape)
    pressure = np.full(len(samples), np.nan + 1j * np.nan)
    velocity = np.full((len(samples), 3), np.nan + 1j * np.nan)
    valid = np.zeros(len(samples), dtype=bool)
    vertices = points[cells]
    centers = vertices.mean(axis=1)
    radii = np.linalg.norm(vertices - centers[:, None], axis=2).max(axis=1)
    symmetry = run.manifest["solver_options"].get("symmetry", "off")
    for x in (1, -1) if symmetry in {"x", "xy"} else (1,):
        for y in (1, -1) if symmetry == "xy" else (1,):
            signs = np.array([x, y, 1])
            local = samples * signs
            tree = cKDTree(local)
            for cell, candidates in enumerate(tree.query_ball_point(centers, radii + 1e-12)):
                if not candidates:
                    continue
                ids = np.asarray(candidates)
                bary = (local[ids] - vertices[cell, 0]) @ gradients[cell].T
                bary[:, 0] += 1
                inside = (bary.min(axis=1) >= -1e-10) & ~valid[ids]
                indices = ids[inside]
                if indices.size:
                    p, v = sample_pressure_velocity(
                        nodal,
                        full_cells,
                        gradients,
                        density,
                        meta["freq_hz"],
                        np.full(indices.size, cell),
                        bary[inside],
                    )
                    pressure[indices], velocity[indices], valid[indices] = p, v * signs, True
    normal = plane.local_axes[2]
    spl = pressure_spl_db(pressure)
    finite = spl[np.isfinite(spl)]
    return {
        "points_m": samples,
        "valid_mask": valid,
        "pressure_pa": pressure,
        "spl_db": spl,
        "normalized_spl_db": spl - finite.max() if finite.size else np.full_like(spl, np.nan),
        "phase_deg": phase_deg(pressure),
        "particle_velocity_m_per_s": velocity,
        "particle_speed_rms_m_per_s": np.linalg.norm(velocity, axis=1),
        "active_in_plane_intensity_w_per_m2": plane_flow_vectors(pressure, velocity, normal)[0],
        "reactive_in_plane_intensity_w_per_m2": plane_flow_vectors(pressure, velocity, normal, component="imaginary")[
            0
        ],
    }
