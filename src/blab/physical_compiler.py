"""Compile a physical loudspeaker model into a validated solver contract."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

import meshio
import numpy as np

from blab.acoustic_materials import (
    REGION_BULK_LOSS_FACTOR_KEY,
    WALL_IMPEDANCE_KEY,
    region_bulk_loss_factor,
    wall_impedance_parameters,
)
from blab.component_symmetry import (
    SYMMETRY_PARAMETER_KEYS,
    ComponentSymmetryInferenceError,
    infer_component_symmetry,
)
from blab.config import normalize_symmetry
from blab.fem_topology import selected_volume_surface_tags
from blab.interface_conform import InterfaceConformError, build_conforming_interface_map
from blab.physical_model import (
    PHYSICAL_MODEL_VERSION,
    AcousticRegion,
    AcousticRegionKind,
    AssumptionStatus,
    Boundary,
    BoundaryKind,
    CompiledBoundary,
    CompiledInterface,
    CompiledMesh,
    CompiledPhysicalSystem,
    CompiledRegion,
    ComponentKind,
    ExcitationPortKind,
    MeshPurpose,
    MeshResource,
    PhysicalComponent,
    PhysicalGroupRef,
    PhysicalSystem,
    PhysicsAssumption,
    ResolvedPhysicalGroup,
)
from blab.symmetry import snap_points_to_symmetry_planes


class PhysicalModelCompileError(ValueError):
    """Raised when a physical model cannot be compiled safely."""

    def __init__(self, issues: list[str] | tuple[str, ...] | str):
        self.issues = (issues,) if isinstance(issues, str) else tuple(issues)
        message = (
            self.issues[0] if len(self.issues) == 1 else "Physical model is invalid:\n- " + "\n- ".join(self.issues)
        )
        super().__init__(message)


class PhysicalSystemCompiler:
    """Resolve mesh semantics and topology without constructing solver matrices."""

    def __init__(self) -> None:
        self._mesh_cache: dict[str, meshio.Mesh] = {}

    def compile(
        self,
        system: PhysicalSystem,
        *,
        symmetry_mode: str = "off",
    ) -> CompiledPhysicalSystem:
        symmetry = normalize_symmetry(symmetry_mode)
        self._validate_authoring_model(system)
        meshes_by_id = {mesh.id: mesh for mesh in system.meshes}
        compiled_meshes = tuple(self._compile_mesh(mesh) for mesh in system.meshes)
        compiled_regions = tuple(self._compile_region(region, meshes_by_id) for region in system.regions)
        compiled_boundaries = tuple(self._compile_boundary(boundary, meshes_by_id) for boundary in system.boundaries)
        compiled_boundaries_by_id = {boundary.id: boundary for boundary in compiled_boundaries}
        compiled_components = self._compile_components(system, symmetry)

        self._validate_boundary_coverage(compiled_regions, compiled_boundaries, meshes_by_id)
        compiled_interfaces = tuple(
            self._compile_interface(
                interface,
                compiled_boundaries_by_id,
                meshes_by_id,
                symmetry_mode=symmetry,
            )
            for interface in system.interfaces
        )

        return CompiledPhysicalSystem(
            id=system.id,
            name=system.name,
            meshes=compiled_meshes,
            regions=compiled_regions,
            boundaries=compiled_boundaries,
            interfaces=compiled_interfaces,
            components=compiled_components,
            excitation_ports=system.excitation_ports,
            assumptions=self._assumptions(system),
            source_model_version=system.model_version,
            metadata=dict(system.metadata),
        )

    def _compile_components(
        self,
        system: PhysicalSystem,
        symmetry_mode: str,
    ) -> tuple[PhysicalComponent, ...]:
        boundaries_by_id = {boundary.id: boundary for boundary in system.boundaries}
        resources_by_id = {resource.id: resource for resource in system.meshes}
        mesh_cache: dict[str, meshio.Mesh] = {}
        compiled = []
        for component in system.components:
            if component.kind != ComponentKind.ELECTRODYNAMIC_TRANSDUCER:
                compiled.append(component)
                continue
            boundaries = tuple(
                boundaries_by_id[boundary_id]
                for boundary_id in component.boundary_ids
                if boundary_id in boundaries_by_id
            )
            try:
                inference = infer_component_symmetry(
                    boundaries,
                    resources_by_id,
                    symmetry_mode,
                    mesh_cache=mesh_cache,
                )
            except ComponentSymmetryInferenceError as exc:
                raise PhysicalModelCompileError(
                    f"Could not infer symmetry for component '{component.name}': {exc}"
                ) from exc
            parameters = {
                key: value for key, value in component.parameters.items() if key not in SYMMETRY_PARAMETER_KEYS
            }
            parameters.update(inference.parameters())
            compiled.append(replace(component, parameters=parameters))
        return tuple(compiled)

    def _validate_authoring_model(self, system: PhysicalSystem) -> None:
        issues: list[str] = []
        if system.model_version != PHYSICAL_MODEL_VERSION:
            issues.append(
                f"Unsupported physical model version {system.model_version}; expected {PHYSICAL_MODEL_VERSION}."
            )
        if not system.id.strip():
            issues.append("Physical system id must not be empty.")
        if not system.name.strip():
            issues.append("Physical system name must not be empty.")
        self._validate_unique_ids(system, issues)

        meshes = {mesh.id: mesh for mesh in system.meshes}
        regions = {region.id: region for region in system.regions}
        boundaries = {boundary.id: boundary for boundary in system.boundaries}
        components = {component.id: component for component in system.components}

        for mesh in system.meshes:
            if not math.isfinite(mesh.scale_to_m) or mesh.scale_to_m <= 0.0:
                issues.append(f"Mesh '{mesh.id}' scale_to_m must be finite and greater than zero.")
            if len(mesh.translation_m) != 3 or not all(math.isfinite(value) for value in mesh.translation_m):
                issues.append(f"Mesh '{mesh.id}' translation_m must contain three finite values.")
            if not Path(mesh.file).is_file():
                issues.append(f"Mesh '{mesh.id}' file does not exist: {mesh.file}")

        for region in system.regions:
            self._validate_region(region, meshes, issues)

        for boundary in system.boundaries:
            region = regions.get(boundary.region_id)
            if region is None:
                issues.append(f"Boundary '{boundary.id}' references unknown region '{boundary.region_id}'.")
                continue
            if boundary.group.mesh_id not in region.mesh_ids:
                issues.append(
                    f"Boundary '{boundary.id}' mesh '{boundary.group.mesh_id}' is not part of region '{region.id}'."
                )
            self._validate_group_ref(
                boundary.group, expected_dimension=2, owner=f"Boundary '{boundary.id}'", issues=issues
            )
            self._validate_json_mapping(
                boundary.parameters, owner=f"Boundary '{boundary.id}' parameters", issues=issues
            )
            if WALL_IMPEDANCE_KEY in boundary.parameters:
                if region.kind != AcousticRegionKind.BOUNDED_AIR:
                    issues.append(f"Boundary '{boundary.id}' wall impedance requires a bounded-air region.")
                if boundary.kind != BoundaryKind.RIGID:
                    issues.append(f"Boundary '{boundary.id}' wall impedance requires a rigid assignment.")
                try:
                    wall_impedance_parameters(boundary.parameters)
                except ValueError as exc:
                    issues.append(f"Boundary '{boundary.id}' {exc}")

        for interface in system.interfaces:
            bounded = boundaries.get(interface.bounded_boundary_id)
            unbounded = boundaries.get(interface.unbounded_boundary_id)
            if bounded is None:
                issues.append(
                    f"Interface '{interface.id}' references unknown bounded boundary '{interface.bounded_boundary_id}'."
                )
            if unbounded is None:
                issues.append(
                    f"Interface '{interface.id}' references unknown unbounded boundary "
                    f"'{interface.unbounded_boundary_id}'."
                )
            if bounded is not None:
                self._validate_interface_side(interface.id, bounded, regions, AcousticRegionKind.BOUNDED_AIR, issues)
            if unbounded is not None:
                self._validate_interface_side(
                    interface.id,
                    unbounded,
                    regions,
                    AcousticRegionKind.UNBOUNDED_AIR,
                    issues,
                )
            if not math.isfinite(interface.coordinate_tolerance_m) or interface.coordinate_tolerance_m <= 0.0:
                issues.append(f"Interface '{interface.id}' coordinate_tolerance_m must be finite and positive.")

        interface_boundary_counts: Counter[str] = Counter(
            boundary_id
            for interface in system.interfaces
            for boundary_id in (interface.bounded_boundary_id, interface.unbounded_boundary_id)
        )
        for boundary in system.boundaries:
            count = interface_boundary_counts[boundary.id]
            if boundary.kind == BoundaryKind.INTERFACE and count != 1:
                issues.append(
                    f"Interface boundary '{boundary.id}' must belong to exactly one acoustic interface; found {count}."
                )
            elif boundary.kind != BoundaryKind.INTERFACE and count:
                issues.append(
                    f"Boundary '{boundary.id}' is referenced by an acoustic interface but is not assigned as interface."
                )
            if boundary.kind == BoundaryKind.PLANE_WAVE_TUBE_TERMINATION:
                region = regions.get(boundary.region_id)
                if region is not None and region.kind != AcousticRegionKind.BOUNDED_AIR:
                    issues.append(
                        f"Plane-wave tube termination '{boundary.id}' requires a bounded-air FEM region."
                    )
                if boundary.parameters:
                    issues.append(
                        f"Plane-wave tube termination '{boundary.id}' does not accept boundary parameters."
                    )

        component_by_boundary: Counter[str] = Counter()
        for component in system.components:
            if not component.boundary_ids:
                issues.append(f"Component '{component.id}' must reference at least one boundary.")
            for boundary_id in component.boundary_ids:
                component_by_boundary[boundary_id] += 1
                boundary = boundaries.get(boundary_id)
                if boundary is None:
                    issues.append(f"Component '{component.id}' references unknown boundary '{boundary_id}'.")
                elif boundary.kind != BoundaryKind.MOVING:
                    issues.append(
                        f"Component '{component.id}' boundary '{boundary_id}' must have kind '{BoundaryKind.MOVING}'."
                    )
            self._validate_json_mapping(
                component.parameters, owner=f"Component '{component.id}' parameters", issues=issues
            )
            raw_weights = component.parameters.get("boundary_motion_weights", {})
            if not isinstance(raw_weights, dict):
                issues.append(f"Component '{component.id}' boundary_motion_weights must be an object.")
            else:
                unknown_weight_boundaries = sorted(set(map(str, raw_weights)) - set(component.boundary_ids))
                if unknown_weight_boundaries:
                    issues.append(
                        f"Component '{component.id}' has motion weights for unrelated boundaries: "
                        + ", ".join(unknown_weight_boundaries)
                        + "."
                    )
                for boundary_id, value in raw_weights.items():
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or float(value) <= 0.0
                    ):
                        issues.append(
                            f"Component '{component.id}' motion weight for '{boundary_id}' "
                            "must be finite and greater than zero."
                        )

        for boundary in system.boundaries:
            if boundary.kind == BoundaryKind.MOVING and component_by_boundary[boundary.id] != 1:
                issues.append(
                    f"Moving boundary '{boundary.id}' must be owned by exactly one component; "
                    f"found {component_by_boundary[boundary.id]}."
                )

        ports_by_component: Counter[str] = Counter()
        for port in system.excitation_ports:
            ports_by_component[port.component_id] += 1
            component = components.get(port.component_id)
            if component is None:
                issues.append(f"Excitation port '{port.id}' references unknown component '{port.component_id}'.")
                continue
            expected_kind = (
                ExcitationPortKind.NORMAL_VELOCITY
                if component.kind == ComponentKind.IDEAL_VELOCITY_SOURCE
                else ExcitationPortKind.VOLTAGE
            )
            if component.kind == ComponentKind.PASSIVE_RADIATOR:
                issues.append(f"Passive component '{component.id}' must not have excitation port '{port.id}'.")
            elif port.kind != expected_kind:
                issues.append(
                    f"Excitation port '{port.id}' for component '{component.id}' must have kind "
                    f"'{expected_kind}', not '{port.kind}'."
                )
            if not port.name.strip():
                issues.append(f"Excitation port '{port.id}' name must not be empty.")

        for component in system.components:
            port_count = ports_by_component[component.id]
            expected_count = 0 if component.kind == ComponentKind.PASSIVE_RADIATOR else 1
            if port_count != expected_count:
                issues.append(
                    f"Component '{component.id}' of kind '{component.kind}' requires {expected_count} "
                    f"excitation port(s); found {port_count}."
                )

        self._validate_json_mapping(system.metadata, owner="Physical system metadata", issues=issues)
        if issues:
            raise PhysicalModelCompileError(issues)

    @staticmethod
    def _validate_unique_ids(system: PhysicalSystem, issues: list[str]) -> None:
        collections = (
            ("mesh", system.meshes),
            ("region", system.regions),
            ("boundary", system.boundaries),
            ("interface", system.interfaces),
            ("component", system.components),
            ("excitation port", system.excitation_ports),
        )
        all_ids: list[str] = []
        for label, items in collections:
            ids = [item.id for item in items]
            duplicates = sorted(item_id for item_id, count in Counter(ids).items() if count > 1)
            if duplicates:
                issues.append(f"Duplicate {label} ids: {', '.join(duplicates)}.")
            if any(not item_id.strip() for item_id in ids):
                issues.append(f"{label.capitalize()} ids must not be empty.")
            all_ids.extend(ids)
        cross_duplicates = sorted(item_id for item_id, count in Counter(all_ids).items() if count > 1)
        if cross_duplicates:
            issues.append(f"Object ids must be globally unique: {', '.join(cross_duplicates)}.")

    def _validate_region(
        self,
        region: AcousticRegion,
        meshes: dict[str, MeshResource],
        issues: list[str],
    ) -> None:
        if not region.mesh_ids:
            issues.append(f"Region '{region.id}' must reference at least one mesh.")
        for mesh_id in region.mesh_ids:
            mesh = meshes.get(mesh_id)
            if mesh is None:
                issues.append(f"Region '{region.id}' references unknown mesh '{mesh_id}'.")
                continue
            expected = (
                MeshPurpose.FEM_VOLUME if region.kind == AcousticRegionKind.BOUNDED_AIR else MeshPurpose.BEM_SURFACE
            )
            if mesh.purpose != expected:
                issues.append(
                    f"Region '{region.id}' requires mesh purpose '{expected}', but mesh '{mesh_id}' "
                    f"has purpose '{mesh.purpose}'."
                )
        if region.kind == AcousticRegionKind.BOUNDED_AIR and not region.volume_groups:
            issues.append(f"Bounded region '{region.id}' must reference at least one physical volume group.")
        if region.kind == AcousticRegionKind.UNBOUNDED_AIR and region.volume_groups:
            issues.append(f"Unbounded region '{region.id}' must not reference volume groups.")
        for group in region.volume_groups:
            self._validate_group_ref(group, expected_dimension=3, owner=f"Region '{region.id}'", issues=issues)
            if group.mesh_id not in region.mesh_ids:
                issues.append(f"Region '{region.id}' volume group mesh '{group.mesh_id}' is not in its mesh_ids.")
        if (
            not math.isfinite(region.sound_speed_m_per_s)
            or region.sound_speed_m_per_s <= 0.0
            or not math.isfinite(region.density_kg_per_m3)
            or region.density_kg_per_m3 <= 0.0
        ):
            issues.append(f"Region '{region.id}' sound speed and density must be finite and positive.")
        self._validate_json_mapping(region.loss_model, owner=f"Region '{region.id}' loss_model", issues=issues)
        unknown_loss_keys = sorted(set(region.loss_model) - {REGION_BULK_LOSS_FACTOR_KEY})
        if unknown_loss_keys:
            issues.append(
                f"Region '{region.id}' uses unsupported loss parameters: " + ", ".join(unknown_loss_keys) + "."
            )
        try:
            bulk_loss_factor = region_bulk_loss_factor(region.loss_model)
        except ValueError as exc:
            issues.append(f"Region '{region.id}' {exc}")
        else:
            if region.kind != AcousticRegionKind.BOUNDED_AIR and bulk_loss_factor != 0.0:
                issues.append(f"Region '{region.id}' FEM bulk loss requires a bounded-air region.")

    @staticmethod
    def _validate_group_ref(
        group: PhysicalGroupRef,
        *,
        expected_dimension: int,
        owner: str,
        issues: list[str],
    ) -> None:
        if group.dimension != expected_dimension:
            issues.append(f"{owner} group dimension must be {expected_dimension}; got {group.dimension}.")
        if group.name is None and group.tag is None:
            issues.append(f"{owner} group must specify a physical name or tag.")
        if group.name is not None and not group.name.strip():
            issues.append(f"{owner} physical group name must not be empty.")
        if group.tag is not None and group.tag <= 0:
            issues.append(f"{owner} physical group tag must be positive.")

    @staticmethod
    def _validate_interface_side(
        interface_id: str,
        boundary: Boundary,
        regions: dict[str, AcousticRegion],
        expected_kind: AcousticRegionKind,
        issues: list[str],
    ) -> None:
        if boundary.kind != BoundaryKind.INTERFACE:
            issues.append(f"Interface '{interface_id}' boundary '{boundary.id}' must have kind 'interface'.")
        region = regions.get(boundary.region_id)
        if region is not None and region.kind != expected_kind:
            issues.append(
                f"Interface '{interface_id}' boundary '{boundary.id}' belongs to '{region.kind}', "
                f"expected '{expected_kind}'."
            )

    @staticmethod
    def _validate_json_mapping(mapping: dict, *, owner: str, issues: list[str]) -> None:
        try:
            json.dumps(mapping, allow_nan=False)
        except (TypeError, ValueError) as exc:
            issues.append(f"{owner} must contain finite JSON-compatible values: {exc}")

    def _compile_mesh(self, mesh_resource: MeshResource) -> CompiledMesh:
        return CompiledMesh(
            id=mesh_resource.id,
            name=mesh_resource.name,
            file=str(Path(mesh_resource.file).resolve()),
            purpose=mesh_resource.purpose,
            scale_to_m=float(mesh_resource.scale_to_m),
            translation_m=tuple(float(value) for value in mesh_resource.translation_m),
        )

    def _compile_region(
        self,
        region: AcousticRegion,
        meshes_by_id: dict[str, MeshResource],
    ) -> CompiledRegion:
        volume_groups = tuple(self._resolve_group(meshes_by_id[group.mesh_id], group) for group in region.volume_groups)
        return CompiledRegion(
            id=region.id,
            name=region.name,
            kind=region.kind,
            mesh_ids=region.mesh_ids,
            volume_groups=volume_groups,
            sound_speed_m_per_s=float(region.sound_speed_m_per_s),
            density_kg_per_m3=float(region.density_kg_per_m3),
            loss_model=dict(region.loss_model),
        )

    def _compile_boundary(
        self,
        boundary: Boundary,
        meshes_by_id: dict[str, MeshResource],
    ) -> CompiledBoundary:
        return CompiledBoundary(
            id=boundary.id,
            name=boundary.name,
            region_id=boundary.region_id,
            kind=boundary.kind,
            group=self._resolve_group(meshes_by_id[boundary.group.mesh_id], boundary.group),
            parameters=dict(boundary.parameters),
        )

    def _compile_interface(
        self,
        interface,
        boundaries_by_id: dict[str, CompiledBoundary],
        meshes_by_id: dict[str, MeshResource],
        *,
        symmetry_mode: str,
    ) -> CompiledInterface:
        bounded = boundaries_by_id[interface.bounded_boundary_id]
        unbounded = boundaries_by_id[interface.unbounded_boundary_id]
        fem_resource = meshes_by_id[bounded.group.mesh_id]
        bem_resource = meshes_by_id[unbounded.group.mesh_id]
        fem_mesh = self._transformed_mesh(fem_resource, symmetry_mode=symmetry_mode)
        bem_mesh = self._transformed_mesh(bem_resource, symmetry_mode=symmetry_mode)
        try:
            topology = build_conforming_interface_map(
                fem_mesh,
                bem_mesh,
                fem_interface_name=bounded.group.name or self._name_for_tag(fem_mesh, bounded.group.tag, 2),
                bem_interface_name=unbounded.group.name or self._name_for_tag(bem_mesh, unbounded.group.tag, 2),
                coordinate_tolerance=float(interface.coordinate_tolerance_m),
                require_closed_bem=True,
                symmetry_mode=symmetry_mode,
            )
        except InterfaceConformError as exc:
            raise PhysicalModelCompileError(f"Interface '{interface.id}' is invalid: {exc}") from exc
        return CompiledInterface(
            id=interface.id,
            name=interface.name,
            bounded_boundary_id=interface.bounded_boundary_id,
            unbounded_boundary_id=interface.unbounded_boundary_id,
            topology=topology,
        )

    def _resolve_group(self, mesh_resource: MeshResource, group: PhysicalGroupRef) -> ResolvedPhysicalGroup:
        mesh = self._read_mesh(mesh_resource)
        tag = group.tag
        name = group.name
        if name is not None:
            field = mesh.field_data.get(name)
            if field is None:
                available = ", ".join(
                    sorted(
                        field_name
                        for field_name, value in mesh.field_data.items()
                        if int(np.asarray(value)[1]) == group.dimension
                    )
                )
                raise PhysicalModelCompileError(
                    f"Mesh '{mesh_resource.id}' does not contain physical group '{name}' "
                    f"in dimension {group.dimension}. Available: {available or '(none)'}."
                )
            resolved_tag, dimension = map(int, np.asarray(field).tolist())
            if dimension != group.dimension:
                raise PhysicalModelCompileError(
                    f"Physical group '{name}' on mesh '{mesh_resource.id}' has dimension {dimension}; "
                    f"expected {group.dimension}."
                )
            if tag is not None and tag != resolved_tag:
                raise PhysicalModelCompileError(
                    f"Physical group '{name}' on mesh '{mesh_resource.id}' resolves to tag {resolved_tag}, "
                    f"not requested tag {tag}."
                )
            tag = resolved_tag
        if tag is None:
            raise PhysicalModelCompileError(f"Physical group on mesh '{mesh_resource.id}' must specify a name or tag.")
        if name is None:
            name = self._name_for_tag(mesh, tag, group.dimension, required=False)

        count = len(self._group_element_indices(mesh, dimension=group.dimension, tag=tag))
        if count == 0:
            label = name or str(tag)
            raise PhysicalModelCompileError(
                f"Physical group '{label}' on mesh '{mesh_resource.id}' contains no "
                f"{'surface' if group.dimension == 2 else 'volume'} elements."
            )
        return ResolvedPhysicalGroup(
            mesh_id=mesh_resource.id,
            dimension=group.dimension,
            tag=int(tag),
            name=name,
        )

    def _validate_boundary_coverage(
        self,
        regions: tuple[CompiledRegion, ...],
        compiled_boundaries: tuple[CompiledBoundary, ...],
        meshes_by_id: dict[str, MeshResource],
    ) -> None:
        boundaries_by_region: dict[str, list[CompiledBoundary]] = {}
        for boundary in compiled_boundaries:
            boundaries_by_region.setdefault(boundary.region_id, []).append(boundary)

        issues: list[str] = []
        for region in regions:
            region_boundaries = boundaries_by_region.get(region.id, [])
            for mesh_id in region.mesh_ids:
                resource = meshes_by_id[mesh_id]
                mesh = self._read_mesh(resource)
                assigned_tags = {
                    boundary.group.tag for boundary in region_boundaries if boundary.group.mesh_id == mesh_id
                }
                if region.kind == AcousticRegionKind.BOUNDED_AIR:
                    selected_volume_tags = {
                        int(group.tag) for group in region.volume_groups if group.mesh_id == mesh_id
                    }
                    try:
                        present_tags = set(selected_volume_surface_tags(mesh, selected_volume_tags))
                    except ValueError as exc:
                        issues.append(f"Region '{region.id}' could not resolve its FEM boundary: {exc}")
                        continue
                else:
                    triangle_tags = self._physical_tags(mesh, dimension=2)
                    present_tags = set(map(int, np.unique(triangle_tags)))
                missing_tags = sorted(present_tags - assigned_tags)
                if missing_tags:
                    names = [self._name_for_tag(mesh, tag, 2, required=False) or str(tag) for tag in missing_tags]
                    issues.append(
                        f"Region '{region.id}' has unassigned physical surface groups on mesh '{mesh_id}': "
                        f"{', '.join(names)}."
                    )
                inactive_tags = sorted(assigned_tags - present_tags)
                if inactive_tags:
                    names = [
                        self._name_for_tag(mesh, tag, 2, required=False) or str(tag)
                        for tag in inactive_tags
                    ]
                    issues.append(
                        f"Region '{region.id}' assigns physical surface groups outside its selected "
                        f"volume groups on mesh '{mesh_id}': {', '.join(names)}."
                    )
                duplicate_groups = [
                    tag
                    for tag, count in Counter(
                        boundary.group.tag for boundary in region_boundaries if boundary.group.mesh_id == mesh_id
                    ).items()
                    if count > 1
                ]
                if duplicate_groups:
                    issues.append(
                        f"Region '{region.id}' assigns physical surface tags more than once on mesh "
                        f"'{mesh_id}': {', '.join(map(str, sorted(duplicate_groups)))}."
                    )
        if issues:
            raise PhysicalModelCompileError(issues)

    def _read_mesh(self, resource: MeshResource) -> meshio.Mesh:
        resolved = str(Path(resource.file).resolve())
        mesh = self._mesh_cache.get(resolved)
        if mesh is None:
            try:
                mesh = meshio.read(resolved)
            except Exception as exc:
                raise PhysicalModelCompileError(
                    f"Could not read mesh '{resource.id}' from {resource.file}: {exc}"
                ) from exc
            self._mesh_cache[resolved] = mesh
        return mesh

    def _transformed_mesh(self, resource: MeshResource, *, symmetry_mode: str = "off") -> meshio.Mesh:
        mesh = self._read_mesh(resource)
        points = np.asarray(mesh.points, dtype=float) * float(resource.scale_to_m)
        points += np.asarray(resource.translation_m, dtype=float)
        points = snap_points_to_symmetry_planes(points, symmetry_mode)
        return meshio.Mesh(
            points=points,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            field_data=mesh.field_data,
            cell_sets=mesh.cell_sets,
        )

    @classmethod
    def _group_element_indices(cls, mesh: meshio.Mesh, *, dimension: int, tag: int) -> np.ndarray:
        tags = cls._physical_tags(mesh, dimension=dimension)
        return np.flatnonzero(tags == int(tag))

    @staticmethod
    def _physical_tags(mesh: meshio.Mesh, *, dimension: int) -> np.ndarray:
        physical_blocks = mesh.cell_data.get("gmsh:physical")
        if physical_blocks is None:
            raise PhysicalModelCompileError("Mesh elements do not contain gmsh:physical tags.")
        cell_types = (
            {"triangle", "triangle3", "triangle6"}
            if dimension == 2
            else {"tetra", "tetra4", "tetra10"}
        )
        values = [
            np.asarray(physical_blocks[index], dtype=np.int64)
            for index, block in enumerate(mesh.cells)
            if block.type in cell_types
        ]
        if not values:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(values)

    @staticmethod
    def _name_for_tag(
        mesh: meshio.Mesh,
        tag: int | None,
        dimension: int,
        *,
        required: bool = True,
    ) -> str | None:
        if tag is not None:
            for name, value in mesh.field_data.items():
                resolved_tag, resolved_dimension = map(int, np.asarray(value).tolist())
                if resolved_tag == int(tag) and resolved_dimension == dimension:
                    return name
        if required:
            raise PhysicalModelCompileError(
                f"Physical tag {tag} in dimension {dimension} has no name; named interface groups are required."
            )
        return None

    @staticmethod
    def _assumptions(system: PhysicalSystem) -> tuple[PhysicsAssumption, ...]:
        assumptions = [
            PhysicsAssumption(AssumptionStatus.INCLUDED, "Linear frequency-domain pressure acoustics"),
            PhysicsAssumption(AssumptionStatus.INCLUDED, "Rigid and prescribed moving boundary roles"),
            PhysicsAssumption(AssumptionStatus.EXCLUDED, "Structural enclosure vibration"),
            PhysicsAssumption(AssumptionStatus.EXCLUDED, "Thermoviscous boundary-layer resolution"),
            PhysicsAssumption(AssumptionStatus.EXCLUDED, "Nonlinear suspension, motor, and fluid behavior"),
        ]
        if any(region.kind == AcousticRegionKind.BOUNDED_AIR for region in system.regions):
            assumptions.append(
                PhysicsAssumption(AssumptionStatus.INCLUDED, "Three-dimensional bounded-air FEM regions")
            )
        if any(region.kind == AcousticRegionKind.UNBOUNDED_AIR for region in system.regions):
            assumptions.append(
                PhysicsAssumption(AssumptionStatus.INCLUDED, "Three-dimensional unbounded-air BEM region")
            )
        if system.interfaces:
            assumptions.append(
                PhysicsAssumption(AssumptionStatus.INCLUDED, "Conforming bidirectional FEM-BEM interfaces")
            )
        if any(component.kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER for component in system.components):
            assumptions.append(
                PhysicsAssumption(
                    AssumptionStatus.INCLUDED,
                    "Linear single-axis rigid-body electrodynamic transducers with dry moving mass",
                )
            )
            assumptions.append(
                PhysicsAssumption(
                    AssumptionStatus.INCLUDED,
                    "Explicit reduced-driver completion and identical symmetry-orbit excitation",
                )
            )
            assumptions.append(
                PhysicsAssumption(
                    AssumptionStatus.EXCLUDED,
                    "Cone breakup, nonlinear motor behavior, and lossy voice-coil inductance",
                )
            )
        if system.excitation_ports:
            assumptions.append(
                PhysicsAssumption(AssumptionStatus.INCLUDED, "Independent reference-excitation transfer basis")
            )
            assumptions.append(PhysicsAssumption(AssumptionStatus.EXCLUDED, "Application DSP and channel synthesis"))
        if any(region_bulk_loss_factor(region.loss_model) > 0.0 for region in system.regions):
            assumptions.append(PhysicsAssumption(AssumptionStatus.INCLUDED, "Homogeneous per-region FEM bulk loss"))
        else:
            assumptions.append(
                PhysicsAssumption(AssumptionStatus.EXCLUDED, "Region-specific acoustic material loss models")
            )
        if any(wall_impedance_parameters(boundary.parameters) is not None for boundary in system.boundaries):
            assumptions.append(
                PhysicsAssumption(
                    AssumptionStatus.INCLUDED,
                    "Locally reacting rigid-backed Miki porous wall treatments",
                )
            )
        if any(
            boundary.kind == BoundaryKind.PLANE_WAVE_TUBE_TERMINATION
            for boundary in system.boundaries
        ):
            assumptions.append(
                PhysicsAssumption(
                    AssumptionStatus.INCLUDED,
                    "Locally reacting anechoic plane-wave tube terminations",
                )
            )
        return tuple(assumptions)
