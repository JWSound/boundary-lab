"""Prepare exterior regions without changing editable mesh resources or FEM topology.

Stitching belongs to the physical region, before interface construction and
compilation. The derived mesh carries an explicit source physical-group map.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import meshio
import numpy as np

from blab.config import MeshConfig, normalize_symmetry
from blab.interface_conform import (
    InterfaceConformError,
    build_conforming_interface_map,
    conform_bem_interface_to_fem,
)
from blab.mesh_cache import read_mesh
from blab.mesh_clean import stitch_meshes
from blab.mesh_topology import analyze_exterior_mesh_topology
from blab.physical_model import (
    AcousticInterface,
    AcousticRegionKind,
    BoundaryKind,
    MeshPurpose,
    MeshResource,
    PhysicalGroupRef,
    PhysicalSystem,
)
from blab.symmetry import snap_points_to_symmetry_planes


def _read(resource: MeshResource, symmetry: str = "off") -> meshio.Mesh:
    if not np.isfinite(resource.scale_to_m) or resource.scale_to_m <= 0:
        raise ValueError(f"Mesh '{resource.name}' scale must be positive and finite.")
    mesh = read_mesh(resource.file)
    mesh.points = np.asarray(mesh.points, dtype=float) * resource.scale_to_m + resource.translation_m
    mesh.points = snap_points_to_symmetry_planes(mesh.points, symmetry)
    return mesh


def _surface_name(mesh: meshio.Mesh, group: PhysicalGroupRef) -> str:
    if group.name is not None:
        return group.name
    for name, (tag, dimension) in mesh.field_data.items():
        if dimension == 2 and tag == group.tag:
            return name
    raise InterfaceConformError(f"Interface surface tag {group.tag} has no physical name.")


def prepare_exterior_system(
    system: PhysicalSystem,
    *,
    stitch_tolerance_mm: float,
    symmetry_mode: str = "off",
    output_root: str | Path | None = None,
    identify_interfaces: bool = False,
) -> PhysicalSystem:
    """Stitch exterior assets, conform FEM interfaces, and return a derived system.

    Boundary/component/interface IDs remain stable; only exterior group references
    and resources change. Interior resources are never passed to the stitcher.
    Interface identification is explicit, for the editor's Build/Identify action.
    """
    symmetry_mode = normalize_symmetry(symmetry_mode)
    if not np.isfinite(stitch_tolerance_mm) or stitch_tolerance_mm <= 0:
        raise ValueError("Exterior stitch tolerance must be positive.")
    root = Path(output_root) if output_root is not None else Path.cwd() / "runs" / "exterior_meshes"
    resources = {mesh.id: mesh for mesh in system.meshes}
    boundaries = list(system.boundaries)
    regions = list(system.regions)
    interfaces = list(system.interfaces)
    provenance = []
    for region_index, region in enumerate(regions):
        if region.kind != AcousticRegionKind.UNBOUNDED_AIR:
            continue
        sources = [resources[mesh_id] for mesh_id in region.mesh_ids]
        if not sources:
            raise ValueError(f"Exterior region '{region.name}' has no mesh assets.")
        meshes = []
        group_map = {}
        for index, source in enumerate(sources):
            mesh = _read(source, symmetry_mode)
            if source.purpose != MeshPurpose.BEM_SURFACE or any(
                cell.type.startswith("tetra") and len(cell.data) for cell in mesh.cells
            ):
                raise ValueError(f"Exterior asset '{source.name}' contains FEM volume elements.")
            fields = {}
            for name, value in mesh.field_data.items():
                tag, dimension = map(int, value)
                if dimension != 2:
                    continue
                derived_name = f"asset{index}:{name}"
                fields[derived_name] = value.copy()
                group_map[(source.id, name)] = derived_name
                group_map[(source.id, tag)] = derived_name
            mesh.field_data = fields
            meshes.append(mesh)
        assembled = (
            meshes[0]
            if len(meshes) == 1
            else stitch_meshes(
                meshes,
                stitch_tol=stitch_tolerance_mm / 1000.0,
                area_tol=1e-18,
                ignored_boundary_axes={"off": (), "x": ("x",), "xy": ("x", "y")}[symmetry_mode],
            )[0]
        )
        derived_id = f"{region.id}:assembled"
        for index, boundary in enumerate(boundaries):
            if boundary.region_id != region.id:
                continue
            group = boundary.group
            name = group_map[(group.mesh_id, group.name if group.name is not None else group.tag)]
            boundaries[index] = replace(
                boundary,
                group=PhysicalGroupRef(
                    mesh_id=derived_id,
                    dimension=2,
                    name=name,
                    tag=int(assembled.field_data[name][0]),
                ),
            )
        by_id = {boundary.id: boundary for boundary in boundaries}
        exterior = [b for b in boundaries if b.region_id == region.id and b.kind == BoundaryKind.INTERFACE]
        pairs = [pair for pair in interfaces if pair.unbounded_boundary_id in {b.id for b in exterior}]
        simplified_boundary_ids = set()
        if identify_interfaces:
            used = {pair.unbounded_boundary_id for pair in pairs}
            paired_fem = {pair.bounded_boundary_id for pair in interfaces}
            bounded_regions = {r.id for r in regions if r.kind == AcousticRegionKind.BOUNDED_AIR}
            for fem_boundary in boundaries:
                if (
                    fem_boundary.kind != BoundaryKind.INTERFACE
                    or fem_boundary.region_id not in bounded_regions
                    or fem_boundary.id in paired_fem
                ):
                    continue
                fem = _read(resources[fem_boundary.group.mesh_id], symmetry_mode)
                matches = []
                failures = []
                for bem_boundary in exterior:
                    if bem_boundary.id in used:
                        continue
                    try:
                        candidate, report = conform_bem_interface_to_fem(
                            fem,
                            assembled,
                            fem_interface_name=_surface_name(fem, fem_boundary.group),
                            bem_interface_name=bem_boundary.group.name,
                            symmetry_mode=symmetry_mode,
                            protected_bem_interface_names=tuple(
                                b.group.name for b in exterior if b.id != bem_boundary.id
                            ),
                        )
                        matches.append((bem_boundary, candidate, report))
                    except InterfaceConformError as exc:
                        failures.append(str(exc))
                if len(matches) != 1:
                    raise InterfaceConformError(
                        f"Interface '{fem_boundary.name}' requires exactly one compatible exterior side; "
                        f"found {len(matches)}. " + " ".join(failures)
                    )
                bem_boundary, assembled, report = matches[0]
                if report.seam_simplification_used:
                    simplified_boundary_ids.add(bem_boundary.id)
                pair = AcousticInterface(
                    id=f"interface:{fem_boundary.id}:{bem_boundary.id}",
                    name=fem_boundary.name,
                    bounded_boundary_id=fem_boundary.id,
                    unbounded_boundary_id=bem_boundary.id,
                )
                interfaces.append(pair)
                pairs.append(pair)
                used.add(bem_boundary.id)
        for pair in pairs:
            fem_boundary = by_id[pair.bounded_boundary_id]
            bem_boundary = by_id[pair.unbounded_boundary_id]
            fem = _read(resources[fem_boundary.group.mesh_id], symmetry_mode)
            options = dict(
                fem_interface_name=_surface_name(fem, fem_boundary.group),
                bem_interface_name=bem_boundary.group.name,
                symmetry_mode=symmetry_mode,
            )
            try:
                build_conforming_interface_map(
                    fem, assembled, **options, coordinate_tolerance=pair.coordinate_tolerance_m
                )
            except InterfaceConformError:
                assembled, report = conform_bem_interface_to_fem(
                    fem,
                    assembled,
                    **options,
                    protected_bem_interface_names=tuple(b.group.name for b in exterior if b.id != bem_boundary.id),
                )
                if report.seam_simplification_used:
                    simplified_boundary_ids.add(bem_boundary.id)
        # Recheck every interface after all modifications, including closure.
        for pair in pairs:
            fem_boundary = by_id[pair.bounded_boundary_id]
            bem_boundary = by_id[pair.unbounded_boundary_id]
            fem = _read(resources[fem_boundary.group.mesh_id], symmetry_mode)
            build_conforming_interface_map(
                fem,
                assembled,
                fem_interface_name=_surface_name(fem, fem_boundary.group),
                bem_interface_name=bem_boundary.group.name,
                coordinate_tolerance=pair.coordinate_tolerance_m,
                symmetry_mode=symmetry_mode,
            )
        root.mkdir(parents=True, exist_ok=True)
        # Content addressing avoids stale artifacts when settings or source files change.
        digest = hashlib.sha256(assembled.points.tobytes())
        for cell in assembled.cells:
            digest.update(cell.data.tobytes())
        for name, values in sorted(assembled.field_data.items()):
            digest.update(name.encode())
            digest.update(np.asarray(values).tobytes())
        for name, blocks in sorted(assembled.cell_data.items()):
            digest.update(name.encode())
            for block in blocks:
                digest.update(np.asarray(block).tobytes())
        path = root / f"exterior_{digest.hexdigest()[:20]}.msh"
        if not path.exists():
            meshio.write(path, assembled, file_format="gmsh22", binary=False)
        topology = analyze_exterior_mesh_topology(
            (MeshConfig(name=region.name, file=str(path), scale_factor=1.0),),
            symmetry=symmetry_mode,
        )
        if topology.has_warnings:
            raise InterfaceConformError(
                f"Assembled exterior '{region.name}' has {topology.open_edge_count} open boundary edges "
                f"away from symmetry planes and {topology.nonmanifold_edge_count} non-manifold edges."
            )
        for source in sources:
            del resources[source.id]
        resources[derived_id] = MeshResource(
            id=derived_id,
            name=f"{region.name} assembled",
            file=str(path.resolve()),
            purpose=MeshPurpose.BEM_SURFACE,
        )
        regions[region_index] = replace(region, mesh_ids=(derived_id,))
        provenance.append(
            {
                "region_id": region.id,
                "stitch_tolerance_mm": stitch_tolerance_mm,
                "symmetry": symmetry_mode,
                "assembled_file": str(path.resolve()),
                "interfaces": [asdict(pair) for pair in pairs],
                "quality_warning_interface_ids": [
                    pair.id for pair in pairs if pair.unbounded_boundary_id in simplified_boundary_ids
                ],
                "sources": [
                    asdict(source) | {"sha256": hashlib.sha256(Path(source.file).read_bytes()).hexdigest()}
                    for source in sources
                ],
                "fem_sources": [
                    asdict(source) | {"sha256": hashlib.sha256(Path(source.file).read_bytes()).hexdigest()}
                    for source in system.meshes
                    if source.id in {by_id[pair.bounded_boundary_id].group.mesh_id for pair in pairs}
                ],
                "surface_map": [
                    {"mesh_id": mesh_id, "source_name": name, "assembled_name": target}
                    for (mesh_id, name), target in group_map.items()
                    if isinstance(name, str)
                ],
            }
        )
    if provenance:
        manifest = json.dumps(provenance, indent=2)
        manifest_path = root / f"preparation_{hashlib.sha256(manifest.encode()).hexdigest()[:20]}.json"
        manifest_path.write_text(manifest, encoding="utf-8")
    return replace(
        system,
        meshes=tuple(resources.values()),
        regions=tuple(regions),
        boundaries=tuple(boundaries),
        interfaces=tuple(interfaces),
        metadata=system.metadata | {"exterior_preparation": provenance},
    )
