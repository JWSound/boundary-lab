#!/usr/bin/env julia

using JSON, LinearAlgebra, SparseArrays, StaticArrays

include(joinpath(@__DIR__, "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled

const DEFAULT_TRANSDUCER_REFERENCE_VOLTAGE_V = 2.83

function object_by_id(items, object_id, label)
    for item in items
        String(item["id"]) == object_id && return item
    end
    error("Unknown $label id: $object_id")
end

function translated_volume_mesh(resource, ::Type{T}) where {T<:AbstractFloat}
    scale = T(resource["scale_to_m"])
    translation = SVector{3,T}(T.(resource["translation_m"]))
    mesh = load_gmsh41_volume(String(resource["file"]), scale)
    return VolumeMesh(
        [vertex + translation for vertex in mesh.vertices],
        mesh.tetrahedra,
        mesh.tetra_physical_tags,
        mesh.boundary_faces,
        mesh.boundary_physical_tags,
        mesh.physical_names,
    )
end

function translated_boundary_mesh(resource, ::Type{T}) where {T<:AbstractFloat}
    scale = T(resource["scale_to_m"])
    translation = SVector{3,T}(T.(resource["translation_m"]))
    mesh = load_gmsh22_with_tags(String(resource["file"]), scale)
    return BoundaryMesh([vertex + translation for vertex in mesh.vertices], mesh.faces, mesh.physical_tags)
end

function aggregate_bem_region(meshes, region, boundaries, ::Type{T}) where {T<:AbstractFloat}
    mesh_ids = String.(region["mesh_ids"])
    isempty(mesh_ids) && error("Unbounded region must contain at least one BEM mesh.")
    resources = [object_by_id(meshes, mesh_id, "mesh") for mesh_id in mesh_ids]
    all(String(resource["purpose"]) == "bem_surface" for resource in resources) ||
        error("Unbounded region meshes must be BEM surfaces.")
    combined = combine_boundary_meshes(
        [translated_boundary_mesh(resource, T) for resource in resources],
    )
    vertex_offset_by_mesh_id = Dict(zip(mesh_ids, combined.vertex_offsets))
    face_offset_by_mesh_id = Dict(zip(mesh_ids, combined.face_offsets))
    tag_map_by_mesh_id = Dict(zip(mesh_ids, combined.physical_tag_maps))
    boundary_tag_by_id = Dict{String,Int}()
    for boundary in boundaries
        String(boundary["region_id"]) == String(region["id"]) || continue
        boundary_id = String(boundary["id"])
        group = boundary["group"]
        mesh_id = String(group["mesh_id"])
        tag_map = get(tag_map_by_mesh_id, mesh_id, nothing)
        tag_map === nothing && error(
            "Exterior boundary $(repr(boundary_id)) references mesh " *
            "$(repr(mesh_id)) outside its unbounded region.",
        )
        source_tag = Int(group["tag"])
        solver_tag = get(tag_map, source_tag, 0)
        solver_tag > 0 || error(
            "Exterior boundary $(repr(boundary_id)) tag $source_tag is not " *
            "present in BEM mesh $(repr(mesh_id)).",
        )
        boundary_tag_by_id[boundary_id] = solver_tag
    end
    return (
        mesh=combined.mesh,
        vertex_offset_by_mesh_id=vertex_offset_by_mesh_id,
        face_offset_by_mesh_id=face_offset_by_mesh_id,
        boundary_tag_by_id=boundary_tag_by_id,
    )
end

function validate_volume_symmetry_fundamental_domain!(mesh, symmetry_mode; tolerance)
    active_axes = symmetry_mode == :off ? () : symmetry_mode == :x ? (1,) : (1, 2)
    for axis in active_axes
        minimum_coordinate, vertex_index = findmin(vertex[axis] for vertex in mesh.vertices)
        minimum_coordinate >= -tolerance || error(
            "FEM mesh is not in the positive $(axis == 1 ? "X" : "Y") fundamental domain for " *
            "$(uppercase(String(symmetry_mode))) symmetry. Vertex $vertex_index has coordinate " *
            "$minimum_coordinate m.",
        )
    end
    return nothing
end

function remapped_index(index_map, wire_index, label)
    original_index = Int(wire_index) + 1
    mapped = get(index_map, original_index, 0)
    mapped > 0 || error("$label lies outside the selected FEM volume groups.")
    return mapped
end

function interface_map_from_wire(raw, volume_selection)
    return ConformingInterfaceMap(
        [
            remapped_index(volume_selection.vertex_index_map, index, "FEM interface vertex")
            for index in raw["fem_vertex_indices"]
        ],
        Int.(raw["fem_to_bem_vertex_indices"]) .+ 1,
        [
            remapped_index(volume_selection.boundary_face_index_map, index, "FEM interface face")
            for index in raw["fem_face_indices"]
        ],
        Int.(raw["bem_face_indices"]) .+ 1,
        Int.(raw["normal_sign"]),
    )
end

function row_major_values(values)
    array = Array(values)
    ndims(array) <= 1 && return vec(array)
    return vec(permutedims(array, reverse(1:ndims(array))))
end

function complex_array_wire(values)
    scalar_type = typeof(real(zero(eltype(values))))
    scalar_type in (Float32, Float64) || error("Unsupported coupled result precision: $scalar_type")
    array = Complex{scalar_type}.(values)
    flattened = row_major_values(array)
    return Dict(
        "dtype" => scalar_type == Float32 ? "complex64" : "complex128",
        "shape" => collect(size(array)),
        "real" => real.(flattened),
        "imag" => imag.(flattened),
    )
end

function quantity_wire(output, values, unit, axes; metadata=Dict{String,Any}())
    return Dict(
        "id" => String(output["id"]),
        "quantity" => String(output["quantity"]),
        "unit" => unit,
        "target_id" => isempty(get(output, "target_ids", Any[])) ? nothing : String(output["target_ids"][1]),
        "axes" => axes,
        "values" => complex_array_wire(values),
        "metadata" => metadata,
    )
end

function rows(vectors, ::Type{T}) where {T<:AbstractFloat}
    isempty(vectors) && return zeros(Complex{T}, 0, 0)
    return reduce(vcat, (reshape(Complex{T}.(values), 1, :) for values in vectors))
end

function per_interface_errors(
    solution,
    fem_mesh,
    bem_mesh,
    interface_maps,
    interface_ranges,
    ::Type{T},
) where {T<:AbstractFloat}
    pressure_errors = T[]
    flux_errors = T[]
    for (interface_map, interface_range) in zip(interface_maps, interface_ranges)
        fem_trace = solution.fem_pressure[interface_map.fem_vertex_indices]
        bem_trace = solution.bem_pressure[interface_map.fem_to_bem_vertex_indices]
        pressure_scale = max(norm(fem_trace), norm(bem_trace), eps(T))
        push!(pressure_errors, T(norm(fem_trace - bem_trace) / pressure_scale))

        local_flux = solution.interface_flux[interface_range]
        interface_dof = Dict(
            vertex => index
            for (index, vertex) in enumerate(interface_map.fem_vertex_indices)
        )
        fem_integrated_flux = zero(Complex{T})
        bem_integrated_flux_along_fem_normal = zero(Complex{T})
        for local_face_index in eachindex(interface_map.fem_face_indices)
            fem_face = fem_mesh.boundary_faces[
                interface_map.fem_face_indices[local_face_index]
            ]
            fem_flux_average = sum(
                local_flux[interface_dof[vertex]] for vertex in fem_face
            ) / T(3)
            bem_face_index = interface_map.bem_face_indices[local_face_index]
            fem_integrated_flux +=
                BeatEngineCoupled._triangle_area(fem_mesh.vertices, fem_face) *
                fem_flux_average
            bem_integrated_flux_along_fem_normal +=
                T(interface_map.normal_sign[local_face_index]) *
                bem_mesh.areas[bem_face_index] *
                solution.bem_neumann[bem_face_index]
        end
        flux_scale = max(
            abs(fem_integrated_flux),
            abs(bem_integrated_flux_along_fem_normal),
            eps(T),
        )
        push!(
            flux_errors,
            T(
                abs(fem_integrated_flux - bem_integrated_flux_along_fem_normal) /
                flux_scale,
            ),
        )
    end
    return pressure_errors, flux_errors
end

function aggregate_fem_domains(
    meshes,
    bounded_regions,
    boundaries,
    ::Type{T},
) where {T<:AbstractFloat}
    vertices = SVector{3,T}[]
    tetrahedra = NTuple{4,Int}[]
    tetra_tags = Int[]
    boundary_faces = NTuple{3,Int}[]
    boundary_tags = Int[]
    domains = NamedTuple[]
    bulk_loss_factor_by_vertex = T[]
    wall_impedances = NamedTuple[]
    fem_boundary_tag_by_id = Dict{String,Int}()
    domain_by_boundary_id = Dict{String,Int}()
    next_boundary_tag = 1

    for region in bounded_regions
        length(region["mesh_ids"]) == 1 ||
            error("Each bounded region must currently contain exactly one FEM mesh.")
        mesh_id = String(only(region["mesh_ids"]))
        resource = object_by_id(meshes, mesh_id, "mesh")
        String(resource["purpose"]) == "fem_volume" ||
            error("Bounded region mesh must be a FEM volume.")
        full_mesh = translated_volume_mesh(resource, T)
        selected_volume_tags = [
            Int(group["tag"])
            for group in region["volume_groups"]
            if String(group["mesh_id"]) == mesh_id
        ]
        selection = restrict_volume_mesh(full_mesh, selected_volume_tags)
        loss_model = get(region, "loss_model", Dict{String,Any}())
        bulk_loss_factor = T(get(loss_model, "bulk_loss_factor", 0.0))
        isfinite(bulk_loss_factor) && zero(T) <= bulk_loss_factor <= one(T) || error(
            "Bounded-region FEM bulk loss factor must be finite and between 0 and 1.",
        )
        region_boundaries = [
            boundary
            for boundary in boundaries
            if String(boundary["region_id"]) == String(region["id"]) &&
               String(boundary["group"]["mesh_id"]) == mesh_id
        ]
        local_tag_map = Dict{Int,Int}()
        for boundary in region_boundaries
            source_tag = Int(boundary["group"]["tag"])
            haskey(local_tag_map, source_tag) && error(
                "FEM physical tag $source_tag on mesh $(repr(mesh_id)) is assigned more than once.",
            )
            solver_tag = next_boundary_tag
            next_boundary_tag += 1
            local_tag_map[source_tag] = solver_tag
            boundary_id = String(boundary["id"])
            fem_boundary_tag_by_id[boundary_id] = solver_tag
            domain_by_boundary_id[boundary_id] = length(domains) + 1
            parameters = get(boundary, "parameters", Dict{String,Any}())
            if haskey(parameters, "wall_impedance")
                String(boundary["kind"]) == "rigid" || error(
                    "Wall impedance boundary $(repr(boundary_id)) must be rigid.",
                )
                treatment = parameters["wall_impedance"]
                String(get(treatment, "model", "miki")) == "miki" || error(
                    "Unsupported wall impedance model on $(repr(boundary_id)).",
                )
                thickness_m = T(get(treatment, "thickness_m", 0.03))
                flow_resistivity = T(get(treatment, "flow_resistivity_pa_s_per_m2", 5000.0))
                isfinite(thickness_m) && thickness_m > zero(T) || error(
                    "Wall-lining thickness must be finite and positive.",
                )
                isfinite(flow_resistivity) && flow_resistivity > zero(T) || error(
                    "Wall-lining airflow resistivity must be finite and positive.",
                )
                push!(
                    wall_impedances,
                    (
                        boundary_id=boundary_id,
                        tag=solver_tag,
                        thickness_m=thickness_m,
                        flow_resistivity_pa_s_per_m2=flow_resistivity,
                    ),
                )
            end
        end
        remapped_boundary_tags = [
            get(local_tag_map, tag, 0)
            for tag in selection.mesh.boundary_physical_tags
        ]
        any(==(0), remapped_boundary_tags) && error(
            "Selected FEM region $(repr(String(region["id"]))) contains an unassigned boundary tag.",
        )

        vertex_offset = length(vertices)
        face_offset = length(boundary_faces)
        append!(vertices, selection.mesh.vertices)
        append!(bulk_loss_factor_by_vertex, fill(bulk_loss_factor, length(selection.mesh.vertices)))
        append!(
            tetrahedra,
            [
                ntuple(index -> tetrahedron[index] + vertex_offset, 4)
                for tetrahedron in selection.mesh.tetrahedra
            ],
        )
        append!(tetra_tags, selection.mesh.tetra_physical_tags)
        append!(
            boundary_faces,
            [
                ntuple(index -> face[index] + vertex_offset, 3)
                for face in selection.mesh.boundary_faces
            ],
        )
        append!(boundary_tags, remapped_boundary_tags)
        push!(
            domains,
            (
                id=String(region["id"]),
                mesh_id=mesh_id,
                selection=selection,
                vertex_offset=vertex_offset,
                face_offset=face_offset,
                vertex_count=length(selection.mesh.vertices),
            ),
        )
    end

    mesh = VolumeMesh{T}(
        vertices,
        tetrahedra,
        tetra_tags,
        boundary_faces,
        boundary_tags,
        Dict{Tuple{Int,Int},String}(),
    )
    return (
        mesh=mesh,
        domains=domains,
        fem_boundary_tag_by_id=fem_boundary_tag_by_id,
        domain_by_boundary_id=domain_by_boundary_id,
        bulk_loss_factor_by_vertex=bulk_loss_factor_by_vertex,
        wall_impedances=wall_impedances,
    )
end

function combined_interface_map_from_wire(
    interfaces,
    fem_domains,
    boundaries,
    bem_domain,
)
    maps = ConformingInterfaceMap[]
    ranges = UnitRange{Int}[]
    next_interface_dof = 1
    for interface in interfaces
        interface_id = String(interface["id"])
        bounded_boundary_id = String(interface["bounded_boundary_id"])
        domain_index = get(
            fem_domains.domain_by_boundary_id,
            bounded_boundary_id,
            0,
        )
        domain_index > 0 || error(
            "Interface $(repr(String(interface["id"]))) does not reference a bounded FEM boundary.",
        )
        domain = fem_domains.domains[domain_index]
        local_map = interface_map_from_wire(interface["topology"], domain.selection)
        unbounded_boundary = object_by_id(
            boundaries,
            String(interface["unbounded_boundary_id"]),
            "unbounded boundary",
        )
        bem_mesh_id = String(unbounded_boundary["group"]["mesh_id"])
        bem_vertex_offset = get(bem_domain.vertex_offset_by_mesh_id, bem_mesh_id, -1)
        bem_face_offset = get(bem_domain.face_offset_by_mesh_id, bem_mesh_id, -1)
        bem_vertex_offset >= 0 && bem_face_offset >= 0 || error(
            "Interface $(repr(interface_id)) references BEM mesh " *
            "$(repr(bem_mesh_id)) outside the unbounded region.",
        )
        mapped = offset_interface_map(
            local_map;
            fem_vertex_offset=domain.vertex_offset,
            fem_face_offset=domain.face_offset,
            bem_vertex_offset=bem_vertex_offset,
            bem_face_offset=bem_face_offset,
        )
        push!(maps, mapped)
        count = length(mapped.fem_vertex_indices)
        push!(ranges, next_interface_dof:(next_interface_dof + count - 1))
        next_interface_dof += count
    end
    if isempty(maps)
        return (
            map=ConformingInterfaceMap(Int[], Int[], Int[], Int[], Int[]),
            ranges=ranges,
            maps=maps,
        )
    end
    return (
        map=ConformingInterfaceMap(
            reduce(vcat, (map.fem_vertex_indices for map in maps)),
            reduce(vcat, (map.fem_to_bem_vertex_indices for map in maps)),
            reduce(vcat, (map.fem_face_indices for map in maps)),
            reduce(vcat, (map.bem_face_indices for map in maps)),
            reduce(vcat, (map.normal_sign for map in maps)),
        ),
        ranges=ranges,
        maps=maps,
    )
end

function electrodynamic_transducers_from_wire(
    components,
    boundaries,
    fem_boundary_tag_by_id,
    bem_boundary_tag_by_id,
    fem_mesh,
    bem_mesh,
    ::Type{T},
    symmetry_mode,
) where {T<:AbstractFloat}
    transducers = ElectrodynamicTransducer{T}[]
    index_by_component_id = Dict{String,Int}()
    for component in components
        String(component["kind"]) == "electrodynamic_transducer" || continue
        parameters = get(component, "parameters", Dict{String,Any}())
        scalar_required = (
            "re_ohm",
            "le_h",
            "bl_n_per_a",
            "mmd_kg",
            "cms_m_per_n",
            "rms_n_s_per_m",
        )
        required = (scalar_required..., "motion_axis")
        optional = (
            "motion_profile",
            "boundary_motion_signs",
            "boundary_motion_weights",
            "symmetry_role",
            "surface_completion_factor",
            "physical_driver_orbit_count",
            "fractional_symmetry_axes",
        )
        missing = [name for name in required if !haskey(parameters, name)]
        isempty(missing) || error(
            "Electrodynamic component $(repr(String(component["id"]))) is missing: " *
            join(missing, ", "),
        )
        unsupported = [
            String(name) for name in keys(parameters)
            if !(String(name) in required) && !(String(name) in optional)
        ]
        isempty(unsupported) || error(
            "Electrodynamic component $(repr(String(component["id"]))) has unsupported " *
            "parameters: " * join(sort(unsupported), ", "),
        )
        motion_profile = String(get(parameters, "motion_profile", "rigid_translation"))
        motion_profile == "rigid_translation" || error(
            "Electrodynamic components currently require a rigid-translation piston motion profile.",
        )
        raw_axis = parameters["motion_axis"]
        length(raw_axis) == 3 || error("motion_axis must contain exactly three values.")
        motion_axis = SVector{3,T}(T.(raw_axis))
        all(isfinite, motion_axis) || error("motion_axis must contain finite values.")
        axis_norm = norm(motion_axis)
        axis_norm > zero(T) || error("motion_axis must have nonzero length.")
        motion_axis /= axis_norm
        completion_factor = Int(get(parameters, "surface_completion_factor", 1))
        orbit_count = Int(get(parameters, "physical_driver_orbit_count", 1))
        completion_factor in (1, 2, 4) ||
            error("surface_completion_factor must be 1, 2, or 4.")
        orbit_count in (1, 2, 4) ||
            error("physical_driver_orbit_count must be 1, 2, or 4.")
        symmetry_role = String(
            get(parameters, "symmetry_role", "complete_representative"),
        )
        expected_role = completion_factor > 1 ?
                        "fractional_driver" :
                        "complete_representative"
        symmetry_role == expected_role || error(
            "symmetry_role is inconsistent with surface_completion_factor.",
        )
        fractional_symmetry_axes = String.(
            get(parameters, "fractional_symmetry_axes", Any[]),
        )
        length(fractional_symmetry_axes) == length(unique(fractional_symmetry_axes)) ||
            error("fractional_symmetry_axes must not contain duplicates.")
        all(axis -> axis in ("x", "y"), fractional_symmetry_axes) ||
            error("fractional_symmetry_axes may contain only x and y.")
        active_symmetry_axes = symmetry_mode == :off ?
                               String[] :
                               symmetry_mode == :x ?
                               ["x"] :
                               ["x", "y"]
        all(axis -> axis in active_symmetry_axes, fractional_symmetry_axes) ||
            error("fractional_symmetry_axes must be active in the selected symmetry mode.")
        completion_factor == 2^length(fractional_symmetry_axes) || error(
            "surface_completion_factor must equal 2 raised to the number of " *
            "fractional_symmetry_axes.",
        )
        for axis in fractional_symmetry_axes
            axis_index = axis == "x" ? 1 : 2
            abs(motion_axis[axis_index]) <= T(1e-8) || error(
                "motion_axis must lie in every symmetry plane that cuts the physical driver.",
            )
        end
        represented_images = completion_factor * orbit_count
        expected_images = BeatEngineCore.symmetry_reduction_factor(symmetry_mode)
        represented_images == expected_images || error(
            "Electrodynamic component $(repr(String(component["id"]))) represents " *
            "$represented_images symmetry images, but $(String(symmetry_mode)) symmetry " *
            "requires $expected_images.",
        )
        raw_signs = get(parameters, "boundary_motion_signs", Dict{String,Any}())
        raw_weights = get(parameters, "boundary_motion_weights", Dict{String,Any}())
        component_boundary_ids = Set(String.(component["boundary_ids"]))
        unknown_sign_boundaries = [
            String(boundary_id) for boundary_id in keys(raw_signs)
            if !(String(boundary_id) in component_boundary_ids)
        ]
        isempty(unknown_sign_boundaries) || error(
            "Electrodynamic component $(repr(String(component["id"]))) has motion signs " *
            "for unrelated boundaries: " * join(sort(unknown_sign_boundaries), ", "),
        )
        unknown_weight_boundaries = [
            String(boundary_id) for boundary_id in keys(raw_weights)
            if !(String(boundary_id) in component_boundary_ids)
        ]
        isempty(unknown_weight_boundaries) || error(
            "Electrodynamic component $(repr(String(component["id"]))) has motion weights " *
            "for unrelated boundaries: " * join(sort(unknown_weight_boundaries), ", "),
        )
        fem_tags = Int[]
        fem_signs = T[]
        bem_tags = Int[]
        bem_signs = T[]
        for boundary_id_value in component["boundary_ids"]
            boundary_id = String(boundary_id_value)
            boundary = object_by_id(boundaries, boundary_id, "boundary")
            String(boundary["kind"]) == "moving" || error(
                "Electrodynamic component $(repr(String(component["id"]))) boundary " *
                "$(repr(boundary_id)) is not moving.",
            )
            sign = T(get(raw_signs, boundary_id, 1))
            sign in (-one(T), one(T)) || error(
                "Electrodynamic boundary motion signs must be -1 or +1.",
            )
            weight = T(get(raw_weights, boundary_id, 1))
            isfinite(weight) && weight > zero(T) || error(
                "Electrodynamic boundary motion weights must be finite and greater than zero.",
            )
            coefficient = sign * weight
            if haskey(fem_boundary_tag_by_id, boundary_id)
                tag = fem_boundary_tag_by_id[boundary_id]
                any(==(tag), fem_mesh.boundary_physical_tags) || error(
                    "Electrodynamic FEM boundary tag $tag is outside the selected volume groups.",
                )
                push!(fem_tags, tag)
                push!(fem_signs, coefficient)
            elseif haskey(bem_boundary_tag_by_id, boundary_id)
                tag = bem_boundary_tag_by_id[boundary_id]
                any(==(tag), bem_mesh.physical_tags) || error(
                    "Electrodynamic BEM boundary tag $tag is not present in the exterior mesh.",
                )
                push!(bem_tags, tag)
                push!(bem_signs, coefficient)
            else
                error(
                    "Electrodynamic component $(repr(String(component["id"]))) references " *
                    "a boundary outside the active acoustic regions.",
                )
            end
        end
        isempty(fem_tags) && isempty(bem_tags) && error(
            "Electrodynamic component $(repr(String(component["id"]))) has no moving acoustic boundary.",
        )
        values = Dict(name => T(parameters[name]) for name in scalar_required)
        all(isfinite, Base.values(values)) || error(
            "Electrodynamic parameters must be finite numbers.",
        )
        values["re_ohm"] > zero(T) || error("re_ohm must be greater than zero.")
        values["le_h"] >= zero(T) || error("le_h must not be negative.")
        values["bl_n_per_a"] > zero(T) || error("bl_n_per_a must be greater than zero.")
        values["mmd_kg"] > zero(T) || error("mmd_kg must be greater than zero.")
        values["cms_m_per_n"] > zero(T) || error("cms_m_per_n must be greater than zero.")
        values["rms_n_s_per_m"] >= zero(T) || error("rms_n_s_per_m must not be negative.")
        push!(
            transducers,
            ElectrodynamicTransducer{T}(
                String(component["id"]),
                fem_tags,
                fem_signs,
                bem_tags,
                bem_signs,
                motion_axis,
                T(completion_factor),
                orbit_count,
                values["re_ohm"],
                values["le_h"],
                values["bl_n_per_a"],
                values["mmd_kg"],
                values["cms_m_per_n"],
                values["rms_n_s_per_m"],
            ),
        )
        index_by_component_id[String(component["id"])] = length(transducers)
    end
    return transducers, index_by_component_id
end

function solve_request(request; event_mode=false)
    Int(get(request, "schema_version", 0)) == 1 || error("Unsupported system solve request schema.")
    cancel_path = get(request, "cancel_path", nothing)
    cancel_requested() = cancel_path !== nothing && isfile(String(cancel_path))
    cancel_requested() && return (cancelled=true, solved_count=0)
    system = request["compiled_system"]
    meshes = system["meshes"]
    regions = system["regions"]
    boundaries = system["boundaries"]
    components = system["components"]
    ports = system["excitation_ports"]
    interfaces = system["interfaces"]

    bounded_regions = [region for region in regions if String(region["kind"]) == "bounded_air"]
    unbounded_regions = [region for region in regions if String(region["kind"]) == "unbounded_air"]
    isempty(bounded_regions) && error("Coupled backend requires at least one bounded region.")
    length(unbounded_regions) == 1 || error("Coupled backend currently requires exactly one unbounded region.")
    unbounded_region = only(unbounded_regions)
    reference_sound_speed = Float64(bounded_regions[1]["sound_speed_m_per_s"])
    reference_density = Float64(bounded_regions[1]["density_kg_per_m3"])
    for region in regions
        isapprox(
            Float64(region["sound_speed_m_per_s"]),
            reference_sound_speed;
            rtol=1e-9,
            atol=0,
        ) && isapprox(
            Float64(region["density_kg_per_m3"]),
            reference_density;
            rtol=1e-9,
            atol=0,
        ) || error(
            "Coupled backend currently requires one shared sound speed and density across all regions.",
        )
    end

    solver_options = get(request, "solver_options", Dict{String,Any}())
    precision_name = lowercase(String(get(solver_options, "precision", "float64")))
    FloatType = if precision_name in ("float32", "complex64")
        Float32
    elseif precision_name in ("float64", "complex128")
        Float64
    else
        error("Unsupported coupled precision: $precision_name. Expected float32 or float64.")
    end
    bem_backend = Symbol(lowercase(String(get(solver_options, "bem_backend", "cpu"))))
    bem_backend in (:cpu, :cuda) ||
        error("Unsupported coupled BEM backend: $bem_backend. Expected cpu or cuda.")
    symmetry_mode = BeatEngineCore.normalized_symmetry_mode(
        get(solver_options, "symmetry", "off"),
    )
    transducer_reference_voltage_v = FloatType(
        get(
            solver_options,
            "transducer_reference_voltage_v",
            DEFAULT_TRANSDUCER_REFERENCE_VOLTAGE_V,
        ),
    )
    transducer_reference_voltage_v > zero(FloatType) || error(
        "transducer_reference_voltage_v must be greater than zero.",
    )
    mesh_setup_started = time_ns()
    fem_domains = aggregate_fem_domains(
        meshes,
        bounded_regions,
        boundaries,
        FloatType,
    )
    fem_mesh = fem_domains.mesh
    bem_domain = aggregate_bem_region(meshes, unbounded_region, boundaries, FloatType)
    bem_mesh = bem_domain.mesh
    symmetry_tolerance = max(
        FloatType(1e-9),
        maximum(maximum(abs, vertex) for vertex in fem_mesh.vertices) * FloatType(1e-6),
    )
    validate_volume_symmetry_fundamental_domain!(
        fem_mesh,
        symmetry_mode;
        tolerance=symmetry_tolerance,
    )
    validate_symmetry_fundamental_domain!(
        bem_mesh,
        symmetry_mode;
        tolerance=symmetry_tolerance,
    )
    combined_interfaces = combined_interface_map_from_wire(
        interfaces,
        fem_domains,
        boundaries,
        bem_domain,
    )
    interface_map = combined_interfaces.map
    mesh_setup_s = (time_ns() - mesh_setup_started) / 1.0e9
    sound_speed = FloatType(reference_sound_speed)
    density = FloatType(reference_density)
    bem_boundary_tag_by_id = bem_domain.boundary_tag_by_id
    transducers, transducer_index_by_component_id = electrodynamic_transducers_from_wire(
        components,
        boundaries,
        fem_domains.fem_boundary_tag_by_id,
        bem_boundary_tag_by_id,
        fem_mesh,
        bem_mesh,
        FloatType,
        symmetry_mode,
    )
    transducer_operators = assemble_transducer_operators(
        fem_mesh,
        bem_mesh,
        transducers,
    )
    excitation_port_ids = String.(request["excitation_port_ids"])
    isempty(excitation_port_ids) && error("Coupled solve requires at least one excitation port.")
    excitations = NamedTuple[]
    for port_id in excitation_port_ids
        port = object_by_id(ports, port_id, "excitation port")
        component = object_by_id(components, String(port["component_id"]), "component")
        port_kind = String(port["kind"])
        component_kind = String(component["kind"])
        if port_kind == "normal_velocity" && component_kind == "ideal_velocity_source"
            parameters = get(component, "parameters", Dict{String,Any}())
            raw_weights = get(parameters, "boundary_motion_weights", Dict{String,Any}())
            candidate_boundaries = [
                object_by_id(boundaries, String(boundary_id), "boundary")
                for boundary_id in component["boundary_ids"]
            ]
            bounded_boundaries = [
                boundary
                for boundary in candidate_boundaries
                if haskey(
                    fem_domains.fem_boundary_tag_by_id,
                    String(boundary["id"]),
                )
            ]
            length(bounded_boundaries) == 1 || error(
                "Each prescribed-velocity component must own exactly one moving boundary " *
                "in the bounded region.",
            )
            radiator_tag = fem_domains.fem_boundary_tag_by_id[
                String(only(bounded_boundaries)["id"])
            ]
            boundary_id = String(only(bounded_boundaries)["id"])
            amplitude = FloatType(get(raw_weights, boundary_id, 1))
            isfinite(amplitude) && amplitude > zero(FloatType) || error(
                "Prescribed-velocity boundary motion weights must be finite and greater than zero.",
            )
            any(==(radiator_tag), fem_mesh.boundary_physical_tags) || error(
                "Moving boundary tag $radiator_tag is not on the selected FEM volume groups.",
            )
            push!(
                excitations,
                (
                    kind=:normal_velocity,
                    radiator_tag=radiator_tag,
                    transducer_index=0,
                    amplitude=Complex{FloatType}(amplitude, 0),
                ),
            )
        elseif port_kind == "voltage" && component_kind == "electrodynamic_transducer"
            transducer_index = get(
                transducer_index_by_component_id,
                String(component["id"]),
                0,
            )
            transducer_index > 0 || error(
                "Voltage port $port_id references an unresolved electrodynamic component.",
            )
            push!(
                excitations,
                (
                    kind=:voltage,
                    radiator_tag=0,
                    transducer_index=transducer_index,
                    amplitude=Complex{FloatType}(
                        transducer_reference_voltage_v,
                        0,
                    ),
                ),
            )
        else
            error(
                "Excitation port $port_id kind $(repr(port_kind)) is incompatible with " *
                "component kind $(repr(component_kind)).",
            )
        end
    end

    quadrature_order = Int(get(solver_options, "quadrature_order", 2))
    singular_order = Int(get(solver_options, "singular_order", 2))
    validation_diagnostics = Bool(get(solver_options, "validation_diagnostics", true))
    cache_frequency_invariant = Bool(get(solver_options, "cache_frequency_invariant", true))
    static_condensation_requested = Bool(
        get(
            solver_options,
            "static_condensation",
            bem_backend == :cuda && !validation_diagnostics,
        ),
    )
    static_condensation = static_condensation_requested
    transducer_fem_vertices = isempty(transducers) ?
                              Int[] :
                              unique(findnz(transducer_operators.fem_surface)[1])
    retained_fem_vertices = sort(
        unique(vcat(interface_map.fem_vertex_indices, transducer_fem_vertices)),
    )
    coupled_cache = nothing
    cache_setup_s = 0.0
    solved_count = 0
    cancelled = cancel_requested()
    cancelled && return (cancelled=true, solved_count=solved_count)
    if cache_frequency_invariant
        cache_setup_started = time_ns()
        coupled_cache = prepare_coupled_cache(
            fem_mesh,
            bem_mesh,
            interface_map;
            quadrature_order=quadrature_order,
            singular_order=singular_order,
            bem_backend=bem_backend,
            symmetry_mode=symmetry_mode,
            retained_fem_vertices=retained_fem_vertices,
            bulk_loss_factor_by_vertex=fem_domains.bulk_loss_factor_by_vertex,
            wall_impedances=fem_domains.wall_impedances,
        )
        cache_setup_s = (time_ns() - cache_setup_started) / 1.0e9
    end
    outputs = get(request, "outputs", Any[])
    for (frequency_index, frequency_value) in enumerate(request["frequencies_hz"])
        if cancel_requested()
            cancelled = true
            break
        end
        frequency_hz = FloatType(frequency_value)
        println(stderr, "Coupled $(precision_name)/$(bem_backend): assembling $(frequency_hz) Hz")
        assembly_started = time_ns()
        coupled_system = build_coupled_system(
            fem_mesh,
            bem_mesh,
            interface_map,
            frequency_hz,
            sound_speed,
            density;
            quadrature_order=quadrature_order,
            singular_order=singular_order,
            cache=coupled_cache,
            validation_diagnostics=validation_diagnostics,
            bem_backend=bem_backend,
            symmetry_mode=symmetry_mode,
            static_condensation=static_condensation,
            bulk_loss_factor_by_vertex=fem_domains.bulk_loss_factor_by_vertex,
            wall_impedances=fem_domains.wall_impedances,
            transducers=transducers,
            transducer_operators=transducer_operators,
        )
        assembly_s = (time_ns() - assembly_started) / 1.0e9
        solve_started = time_ns()
        solutions = solve_coupled_excitations(coupled_system, excitations)
        solve_s = (time_ns() - solve_started) / 1.0e9
        interface_error_sets = [
            per_interface_errors(
                solution,
                fem_mesh,
                bem_mesh,
                combined_interfaces.maps,
                combined_interfaces.ranges,
                FloatType,
            )
            for solution in solutions
        ]
        interface_pressure_errors = [
            maximum(errors[1][index] for errors in interface_error_sets)
            for index in eachindex(interfaces)
        ]
        interface_flux_errors = [
            maximum(errors[2][index] for errors in interface_error_sets)
            for index in eachindex(interfaces)
        ]
        field_s = 0.0
        quantities = Dict{String,Any}[]
        for output in outputs
            quantity = String(output["quantity"])
            if quantity == "fem_nodal_pressure"
                push!(
                    quantities,
                    quantity_wire(
                        output,
                        rows([solution.fem_pressure for solution in solutions], FloatType),
                        "Pa",
                        ["excitation", "fem_node"],
                        metadata=Dict(
                            "mesh_ids" => [domain.mesh_id for domain in fem_domains.domains],
                            "region_ids" => [domain.id for domain in fem_domains.domains],
                            "node_offsets" => [
                                domain.vertex_offset for domain in fem_domains.domains
                            ],
                            "node_counts" => [
                                domain.vertex_count for domain in fem_domains.domains
                            ],
                        ),
                    ),
                )
            elseif quantity == "bem_boundary_pressure"
                push!(
                    quantities,
                    quantity_wire(
                        output,
                        rows([solution.bem_pressure for solution in solutions], FloatType),
                        "Pa",
                        ["excitation", "bem_node"],
                        metadata=Dict("mesh_id" => String(bem_resource["id"])),
                    ),
                )
            elseif quantity == "interface_normal_derivative"
                push!(
                    quantities,
                    quantity_wire(
                        output,
                        rows([solution.interface_flux for solution in solutions], FloatType),
                        "Pa/m",
                        ["excitation", "interface_node"],
                        metadata=Dict(
                            "interface_ids" => [
                                String(interface["id"]) for interface in interfaces
                            ],
                            "interface_offsets" => [
                                first(range) - 1 for range in combined_interfaces.ranges
                            ],
                            "interface_counts" => [
                                length(range) for range in combined_interfaces.ranges
                            ],
                        ),
                    ),
                )
            elseif quantity == "diaphragm_velocity"
                push!(
                    quantities,
                    quantity_wire(
                        output,
                        rows(
                            [solution.diaphragm_velocity for solution in solutions],
                            FloatType,
                        ),
                        "m/s",
                        ["excitation", "transducer"],
                        metadata=Dict(
                            "component_ids" => [transducer.id for transducer in transducers],
                            "surface_completion_factors" => [
                                transducer.surface_completion_factor
                                for transducer in transducers
                            ],
                            "physical_driver_orbit_counts" => [
                                transducer.physical_driver_orbit_count
                                for transducer in transducers
                            ],
                        ),
                    ),
                )
            elseif quantity == "voice_coil_current"
                push!(
                    quantities,
                    quantity_wire(
                        output,
                        rows(
                            [solution.voice_coil_current for solution in solutions],
                            FloatType,
                        ),
                        "A",
                        ["excitation", "transducer"],
                        metadata=Dict(
                            "component_ids" => [transducer.id for transducer in transducers],
                            "surface_completion_factors" => [
                                transducer.surface_completion_factor
                                for transducer in transducers
                            ],
                            "physical_driver_orbit_counts" => [
                                transducer.physical_driver_orbit_count
                                for transducer in transducers
                            ],
                        ),
                    ),
                )
            elseif quantity == "exterior_pressure"
                field_started = time_ns()
                options = get(output, "options", Dict{String,Any}())
                raw_points = get(options, "points_m", Any[])
                isempty(raw_points) && error("exterior_pressure output requires options.points_m.")
                points = [SVector{3,FloatType}(FloatType.(point)) for point in raw_points]
                pressures = [
                    if bem_backend == :cuda
                        evaluate_galerkin_field_cuda(
                            points,
                            bem_mesh,
                            solution.bem_pressure,
                            solution.bem_neumann,
                            coupled_system.wavenumber,
                            coupled_system.field_cache,
                        )
                    else
                        evaluate_galerkin_field_cpu(
                            points,
                            bem_mesh,
                            solution.bem_pressure,
                            solution.bem_neumann,
                            coupled_system.wavenumber,
                            coupled_system.field_cache,
                        )
                    end
                    for solution in solutions
                ]
                push!(
                    quantities,
                    quantity_wire(
                        output,
                        rows(pressures, FloatType),
                        "Pa",
                        ["excitation", "observation"],
                        metadata=Dict("points_m" => raw_points),
                    ),
                )
                field_s += (time_ns() - field_started) / 1.0e9
            else
                error("Unsupported coupled output quantity: $quantity")
            end
        end

        diagnostics = Dict{String,Any}(
            "precision" => precision_name,
            "bem_backend" => String(bem_backend),
            "linear_backend" => String(coupled_system.linear_backend),
            "symmetry" => String(symmetry_mode),
            "formulation" => String(coupled_system.formulation),
            "linear_solver" => coupled_system.formulation == :fem_interface_condensed ?
                               "cuda_cudss_schur_plus_dense_lu" :
                               "$(coupled_system.linear_backend)_dense_lu",
            "full_system_order" => coupled_system.full_system_order,
            "solved_system_order" => coupled_system.solved_system_order,
            "bounded_region_count" => length(bounded_regions),
            "interface_count" => length(interfaces),
            "transducer_count" => length(transducers),
            "transducer_reference_voltage_v" => transducer_reference_voltage_v,
            "fem_bulk_loss_factors_by_region" => Dict(
                String(region["id"]) => Float64(
                    get(get(region, "loss_model", Dict{String,Any}()), "bulk_loss_factor", 0.0),
                )
                for region in bounded_regions
            ),
            "wall_impedance_boundary_ids" => [spec.boundary_id for spec in fem_domains.wall_impedances],
            "static_condensation_requested" => static_condensation_requested,
            "static_condensation_active" => static_condensation,
            "pressure_continuity_error" => isempty(interface_pressure_errors) ?
                                           nothing :
                                           maximum(interface_pressure_errors),
            "flux_conservation_error" => isempty(interface_flux_errors) ?
                                         nothing :
                                         maximum(interface_flux_errors),
            "interface_ids" => [String(interface["id"]) for interface in interfaces],
            "interface_pressure_continuity_errors" => interface_pressure_errors,
            "interface_flux_conservation_errors" => interface_flux_errors,
            "timings" => Dict(
                "assembly_s" => assembly_s,
                "solve_s" => solve_s,
                "field_s" => field_s,
                "mesh_setup_s" => frequency_index == 1 ? mesh_setup_s : 0.0,
                "cache_setup_s" => frequency_index == 1 ? cache_setup_s : 0.0,
                "fem_matrix_cache_s" => frequency_index == 1 ?
                                        coupled_system.cache.timings.fem_matrix_cache_s : 0.0,
                "interface_operator_cache_s" => frequency_index == 1 ?
                                                coupled_system.cache.timings.interface_operator_cache_s : 0.0,
                "bem_space_cache_s" => frequency_index == 1 ?
                                       coupled_system.cache.timings.bem_space_cache_s : 0.0,
                "bem_singular_cache_s" => frequency_index == 1 ?
                                          coupled_system.cache.timings.bem_singular_cache_s : 0.0,
                "bem_cpu_assembly_cache_s" => frequency_index == 1 ?
                                              coupled_system.cache.timings.bem_cpu_assembly_cache_s : 0.0,
                "bem_device_regular_cache_s" => frequency_index == 1 ?
                                                coupled_system.cache.timings.bem_device_regular_cache_s : 0.0,
                "bem_device_singular_cache_s" => frequency_index == 1 ?
                                                 coupled_system.cache.timings.bem_device_singular_cache_s : 0.0,
                "bem_device_image_cache_s" => frequency_index == 1 ?
                                              coupled_system.cache.timings.bem_device_image_cache_s : 0.0,
                "bem_identity_cache_s" => frequency_index == 1 ?
                                          coupled_system.cache.timings.bem_identity_cache_s : 0.0,
                "device_block_cache_s" => frequency_index == 1 ?
                                          coupled_system.cache.timings.device_block_cache_s : 0.0,
                "field_cache_s" => frequency_index == 1 ?
                                   coupled_system.cache.timings.field_cache_s : 0.0,
                "fem_system_s" => coupled_system.timings.fem_system_s,
                "bem_operator_s" => coupled_system.timings.bem_operator_s,
                "bem_matrix_s" => coupled_system.timings.bem_matrix_s,
                "fem_condensation_s" => coupled_system.timings.fem_condensation_s,
                "fem_condensation_analysis_s" => isnothing(coupled_system.condensation) ?
                                                 0.0 :
                                                 coupled_system.condensation.timings.analysis_s,
                "fem_condensation_factorization_s" => isnothing(coupled_system.condensation) ?
                                                      0.0 :
                                                      coupled_system.condensation.timings.factorization_s,
                "fem_schur_extraction_s" => isnothing(coupled_system.condensation) ?
                                            0.0 :
                                            coupled_system.condensation.timings.schur_extraction_s,
                "block_assembly_s" => coupled_system.timings.block_assembly_s,
                "coupled_factorization_s" => coupled_system.timings.coupled_factorization_s,
                "replay_factorization_s" => coupled_system.timings.replay_factorization_s,
            ),
        )
        if validation_diagnostics
            diagnostics["relative_residual"] = maximum(solution.relative_residual for solution in solutions)
            diagnostics["all_bem_replay_error"] = maximum(
                solution.all_bem_replay_error for solution in solutions
            )
        end
        result = Dict(
            "schema_version" => 1,
            "freq_hz" => frequency_hz,
            "excitation_port_ids" => excitation_port_ids,
            "quantities" => quantities,
            "diagnostics" => diagnostics,
        )
        if event_mode
            println(JSON.json(Dict("type" => "result", "result" => result)))
        else
            println(JSON.json(result))
        end
        flush(stdout)
        release_coupled_system!(coupled_system)
        solved_count = frequency_index
    end
    coupled_cache === nothing || release_coupled_cache!(coupled_cache)
    cancelled = cancelled || cancel_requested()
    return (cancelled=cancelled, solved_count=solved_count)
end

function reclaim_accelerator_memory_after_failure()
    GC.gc(true)
    try
        cuda = BeatEngineCore.cuda_module()
        cuda.reclaim()
    catch
        nothing
    end
    return nothing
end

function run_worker()
    println(JSON.json(Dict("type" => "ready")))
    flush(stdout)
    for line in eachline(stdin)
        isempty(strip(line)) && continue
        try
            submission = JSON.parse(line)
            request_path = String(submission["request"])
            request = JSON.parse(read(request_path, String))
            outcome = solve_request(request; event_mode=true)
            event_type = outcome.cancelled ? "cancelled" : "completed"
            println(JSON.json(Dict("type" => event_type, "solved_count" => outcome.solved_count)))
        catch exception
            reclaim_accelerator_memory_after_failure()
            error_text = sprint(showerror, exception, catch_backtrace())
            println(JSON.json(Dict("type" => "failed", "error" => error_text)))
        end
        flush(stdout)
    end
end

if "--worker" in ARGS
    try
        run_worker()
    catch exception
        reclaim_accelerator_memory_after_failure()
        showerror(stderr, exception, catch_backtrace())
        println(stderr)
        exit(1)
    end
else
    try
        request = JSON.parse(read(stdin, String))
        solve_request(request)
    catch exception
        showerror(stderr, exception, catch_backtrace())
        println(stderr)
        exit(1)
    end
end
