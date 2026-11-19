"""Sample FEM and BEM fields on one project-space reporting grid."""

import hashlib

import numpy as np
from scipy.spatial import cKDTree

from blab.fem_fields import plane_flow_vectors, sample_pressure_velocity, tetrahedral_shape_gradients
from blab.observation_planes import ObservationPlaneType, sample_plane_points
from blab.phasor import convert_phasor
from blab.solve_results.derived import phase_deg, pressure_spl_db
from blab.solve_results.model import (
    BEM_BOUNDARY_DOMAIN_ID,
    BEM_BOUNDARY_NEUMANN_ID,
    BEM_BOUNDARY_PRESSURE_ID,
    FEM_NODAL_PRESSURE_ID,
    FEM_VOLUME_DOMAIN_ID,
)
from blab.solvers.coupled_field import BemFieldEvaluationRequest, evaluate_bem_field


def _interior_samples(run, samples, hz, excitation, response, uncorrected):
    if FEM_VOLUME_DOMAIN_ID not in run.domains:
        raise ValueError("Interior and combined planes require retained fem_nodal_pressure and FEM topology.")
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
    return pressure, velocity, valid


def _exterior_mask(points, triangles, samples, symmetry, resolution):
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ValueError("Exterior plane geometry requires PyVista. Install boundary-lab[fields].") from exc
    blocks = [
        points * (x, y, 1)
        for x in ((1, -1) if symmetry in {"x", "xy"} else (1,))
        for y in ((1, -1) if symmetry == "xy" else (1,))
    ]
    faces = np.vstack([triangles + i * len(points) for i in range(len(blocks))])
    cells = np.column_stack((np.full(len(faces), 3), faces))
    boundary = pv.PolyData(np.vstack(blocks), cells).clean(tolerance=1e-10, absolute=True)
    edges = boundary.extract_feature_edges(
        boundary_edges=True, non_manifold_edges=True, feature_edges=False, manifold_edges=False
    )
    if edges.n_cells:
        raise ValueError("Exterior classification requires a closed manifold BEM boundary.")
    classified = pv.PolyData(samples).select_interior_points(
        boundary, method="cell_locator", locator_tolerance=1e-6, check_surface=False
    )
    _, closest = boundary.find_closest_cell(samples, return_closest_point=True)
    return ~np.asarray(classified["selected_points"], dtype=bool) & (
        np.linalg.norm(samples - closest, axis=1) > max(1e-7, resolution * 0.25)
    )


def sample_plane(
    run, plane, hz, excitation, response, uncorrected, *, backend="beat_cpu", julia="julia", pressure_only=False
):
    samples = sample_plane_points(plane, plane.sample_shape)
    if plane.plane_type == ObservationPlaneType.EXTERIOR:
        pressure = np.full(len(samples), complex(np.nan, np.nan))
        velocity = np.full((len(samples), 3), complex(np.nan, np.nan))
        interior = np.zeros(len(samples), dtype=bool)
    else:
        pressure, velocity, interior = _interior_samples(run, samples, hz, excitation, response, uncorrected)
    exterior = np.zeros_like(interior)
    if plane.plane_type != ObservationPlaneType.INTERIOR:
        domain = run.domains.get(BEM_BOUNDARY_DOMAIN_ID)
        if domain is None:
            raise ValueError("Exterior evaluation requires retained bem_boundary_traces.")
        points = run.domain_array(domain["id"], "points_m")
        triangles = run.domain_array(domain["id"], "triangles")
        symmetry = domain["metadata"].get("symmetry", run.manifest["solver_options"].get("symmetry", "off"))
        traces = []
        for qid, axis in ((BEM_BOUNDARY_PRESSURE_ID, "bem_node"), (BEM_BOUNDARY_NEUMANN_ID, "bem_face")):
            _, axes, values, meta = run.sample(
                hz, qid, excitation=excitation, response=response, uncorrected=uncorrected
            )
            if axes != [axis]:
                raise ValueError(f"Expected {axis} trace for exterior evaluation.")
            traces.append(convert_phasor(values, run.manifest["phasor_convention"]) if excitation else values)
        exterior = ~interior & _exterior_mask(points, triangles, samples, symmetry, plane.resolution_m)
        region = next((r for r in run.system["regions"] if r["kind"] == "unbounded_air"), None)
        if region is None:
            raise ValueError("The compiled system lacks an exterior acoustic region.")
        if exterior.any():
            digest = hashlib.sha256(points.tobytes())
            digest.update(triangles.tobytes())
            request = BemFieldEvaluationRequest(
                domain_key=f"{digest.hexdigest()}:{symmetry}",
                points_m=samples[exterior],
                boundary_points_m=points,
                boundary_triangles=triangles,
                boundary_pressure=traces[0],
                boundary_normal_derivative=traces[1],
                frequency_hz=meta["freq_hz"],
                sound_speed_m_per_s=region["sound_speed_m_per_s"],
                symmetry=symmetry,
                pressure_gradient=not pressure_only,
            )
            values = evaluate_bem_field(request, backend_id=backend, julia_executable=julia)
            if pressure_only:
                pressure[exterior] = values
            else:
                pressure[exterior] = values[:, 0]
                velocity[exterior] = values[:, 1:] / (-2j * np.pi * meta["freq_hz"] * region["density_kg_per_m3"])
    valid = interior | exterior
    normal = plane.local_axes[2]
    spl = pressure_spl_db(pressure)
    finite = spl[np.isfinite(spl)]
    return {
        "points_m": samples,
        "valid_mask": valid,
        "interior_mask": interior,
        "exterior_mask": exterior,
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
