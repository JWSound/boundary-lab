#!/usr/bin/env julia

using JSON, LinearAlgebra, SparseArrays, StaticArrays, Statistics

include(joinpath(@__DIR__, "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled
include(joinpath(@__DIR__, "src", "BeatEngineCoupledCondensed.jl"))
using .BeatEngineCoupledCondensed

const DEFAULT_TRANSDUCER_REFERENCE_VOLTAGE_V = 2.83
const BEM_FIELD_EVALUATION_CACHES = Dict{String,Any}()
const BEM_FIELD_EVALUATION_CACHE_ORDER = String[]
const MAX_BEM_FIELD_EVALUATION_CACHES = 2

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
    translated_meshes = [translated_boundary_mesh(resource, T) for resource in resources]
    combined = combine_boundary_meshes(translated_meshes)
    vertex_offset_by_mesh_id = Dict(zip(mesh_ids, combined.vertex_offsets))
    face_offset_by_mesh_id = Dict(zip(mesh_ids, combined.face_offsets))
    vertex_count_by_mesh_id = Dict(zip(mesh_ids, [length(mesh.vertices) for mesh in translated_meshes]))
    face_count_by_mesh_id = Dict(zip(mesh_ids, [length(mesh.faces) for mesh in translated_meshes]))
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
        vertex_count_by_mesh_id=vertex_count_by_mesh_id,
        face_count_by_mesh_id=face_count_by_mesh_id,
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

function exterior_quadrature_selection(options, mesh::BoundaryMesh{T}, frequency_hz::T, sound_speed::T, base_order::Int, backend::Symbol) where {T<:AbstractFloat}
    default_mode = backend == :cpu ? "wavelength" : "fixed"
    mode = lowercase(String(get(options, "regular_quadrature_mode", default_mode)))
    mode in ("fixed", "wavelength") || error(
        "Unsupported regular quadrature mode: $mode. Expected fixed or wavelength.",
    )
    backend == :cpu || mode == "fixed" || error(
        "Wavelength-driven regular quadrature is currently available only on the CPU backend.",
    )
    if mode == "fixed"
        return (order=base_order, mode=mode, mesh_stat=nothing, area=nothing, length=nothing, kh=nothing)
    end
    mesh_stat = lowercase(String(get(options, "wavelength_mesh_stat", "p90")))
    areas = collect(Float64.(mesh.areas))
    area = if mesh_stat == "median"
        median(areas)
    elseif mesh_stat == "p75"
        quantile(areas, 0.75)
    elseif mesh_stat == "p90"
        quantile(areas, 0.90)
    elseif mesh_stat == "max"
        maximum(areas)
    else
        error("Unsupported wavelength mesh stat: $mesh_stat.")
    end
    element_length = sqrt(area)
    kh = Float64(2pi * frequency_hz / sound_speed) * element_length
    q1_cutoff = Float64(get(options, "wavelength_kh_q1_max", 0.0))
    q2_cutoff = Float64(get(options, "wavelength_kh_q2_max", 2.0))
    q1_cutoff >= 0.0 || error("wavelength_kh_q1_max must be non-negative.")
    q2_cutoff > q1_cutoff || error("wavelength_kh_q2_max must exceed wavelength_kh_q1_max.")
    order = kh <= q1_cutoff ? 1 : kh <= q2_cutoff ? 2 : base_order
    return (order=order, mode=mode, mesh_stat=mesh_stat, area=area, length=element_length, kh=kh)
end

function exterior_excitations(ports, components, boundaries, boundary_tag_by_id, exterior_region, ::Type{T}) where {T<:AbstractFloat}
    excitations = NamedTuple[]
    for port_id in String.(ports)
        port = object_by_id(components.ports, port_id, "excitation port")
        String(port["kind"]) == "normal_velocity" || error(
            "Exterior excitation port $port_id must be a normal-velocity port.",
        )
        component = object_by_id(components.items, String(port["component_id"]), "component")
        String(component["kind"]) == "ideal_velocity_source" || error(
            "Exterior excitation port $port_id must reference an ideal velocity source.",
        )
        parameters = get(component, "parameters", Dict{String,Any}())
        raw_weights = get(parameters, "boundary_motion_weights", Dict{String,Any}())
        tags = Int[]
        amplitudes = T[]
        for boundary_id in String.(component["boundary_ids"])
            boundary = object_by_id(boundaries, boundary_id, "boundary")
            String(boundary["region_id"]) == String(exterior_region["id"]) || error(
                "Exterior component $(repr(String(component["id"]))) references a boundary outside its region.",
            )
            String(boundary["kind"]) == "moving" || continue
            tag = get(boundary_tag_by_id, boundary_id, 0)
            tag > 0 || error("Exterior moving boundary $boundary_id did not resolve to a BEM tag.")
            amplitude = T(get(raw_weights, boundary_id, 1.0))
            isfinite(amplitude) && amplitude > zero(T) || error(
                "Exterior boundary motion weights must be finite and positive.",
            )
            push!(tags, tag)
            push!(amplitudes, amplitude)
        end
        isempty(tags) && error(
            "Exterior component $(repr(String(component["id"]))) has no moving boundaries.",
        )
        push!(excitations, (port_id=port_id, component_id=String(component["id"]), tags=tags, amplitudes=amplitudes))
    end
    return excitations
end

function exterior_neumann(mesh, excitation, density::T, omega::T) where {T<:AbstractFloat}
    values = zeros(Complex{T}, length(mesh.faces))
    for (tag, amplitude) in zip(excitation.tags, excitation.amplitudes)
        for face_index in eachindex(mesh.faces)
            mesh.physical_tags[face_index] == tag || continue
            values[face_index] = Complex{T}(0, density * omega * amplitude)
        end
    end
    return values
end

function exterior_field(points, mesh, pressure, neumann, wavenumber, cache, backend)
    if backend == :cuda
        return evaluate_galerkin_field_cuda(points, mesh, pressure, neumann, wavenumber, cache)
    elseif backend == :rocm
        return evaluate_galerkin_field_rocm(points, mesh, pressure, neumann, wavenumber, cache)
    end
    return evaluate_galerkin_field_cpu(points, mesh, pressure, neumann, wavenumber, cache)
end

function exterior_component_impedance(mesh, pressure, excitation, symmetry_mode, ::Type{T}) where {T<:AbstractFloat}
    force = zero(Complex{T})
    selected_tags = Set(excitation.tags)
    for face_index in eachindex(mesh.faces)
        mesh.physical_tags[face_index] in selected_tags || continue
        face = mesh.faces[face_index]
        average_pressure = (pressure[face[1]] + pressure[face[2]] + pressure[face[3]]) / T(3)
        force += average_pressure * T(mesh.areas[face_index])
    end
    force *= T(symmetry_reduction_factor(symmetry_mode)) * T(10)
    return Complex{T}(real(force) / T(2), -imag(force) / T(2))
end

function solve_exterior_request(request, system, unbounded_region; event_mode=false)
    meshes = system["meshes"]
    boundaries = system["boundaries"]
    components = system["components"]
    port_objects = system["excitation_ports"]
    options = get(request, "solver_options", Dict{String,Any}())
    precision_name = lowercase(String(get(options, "precision", "float32")))
    FloatType = precision_name == "float64" ? Float64 : precision_name == "float32" ? Float32 :
                error("Exterior precision must be float32 or float64.")
    backend = Symbol(lowercase(String(get(options, "bem_backend", "cpu"))))
    backend in (:cpu, :cuda, :rocm) || error("Exterior BEM backend must be cpu, cuda, or rocm.")
    symmetry_mode = BeatEngineCore.normalized_symmetry_mode(get(options, "symmetry", "off"))
    sound_speed = FloatType(unbounded_region["sound_speed_m_per_s"])
    density = FloatType(unbounded_region["density_kg_per_m3"])
    sound_speed > zero(FloatType) && density > zero(FloatType) || error(
        "Exterior sound speed and density must be positive.",
    )
    mesh_setup_started = time_ns()
    bem_domain = aggregate_bem_region(meshes, unbounded_region, boundaries, FloatType)
    mesh = snap_symmetry_planes(bem_domain.mesh, symmetry_mode)
    validate_symmetry_fundamental_domain!(mesh, symmetry_mode)
    excitation_port_ids = String.(request["excitation_port_ids"])
    excitations = exterior_excitations(
        excitation_port_ids,
        (ports=port_objects, items=components),
        boundaries,
        bem_domain.boundary_tag_by_id,
        unbounded_region,
        FloatType,
    )
    mesh_setup_s = (time_ns() - mesh_setup_started) / 1.0e9
    p1_space = build_p1_space(mesh)
    dp0_space = build_dp0_space(mesh)
    base_order = Int(get(options, "quadrature_order", 4))
    singular_order = Int(get(options, "singular_order", 4))
    base_rule = triangle_rule(FloatType, base_order)
    singular_cache = build_singular_correction_cache(mesh, singular_order)
    identity_p1_p1 = assemble_l2_identity_matrix(
        mesh, p1_space, dp0_space, base_rule, :p1, :p1; symmetry_mode=symmetry_mode,
    )
    identity_p1_dp0 = assemble_l2_identity_matrix(
        mesh, p1_space, dp0_space, base_rule, :p1, :dp0; symmetry_mode=symmetry_mode,
    )
    cpu_field_cache = build_field_evaluation_cache(mesh, base_rule; symmetry_mode=symmetry_mode)
    accelerator_backend = backend in (:cuda, :rocm)
    device_cache = if backend == :cuda
        build_cuda_regular_assembly_cache(mesh, base_rule)
    elseif backend == :rocm
        build_rocm_regular_assembly_cache(
            mesh,
            p1_space,
            dp0_space,
            base_rule;
            singular_order=singular_order,
            symmetry_mode=symmetry_mode,
        )
    else
        nothing
    end
    field_cache = if backend == :cuda
        build_cuda_field_evaluation_cache(cpu_field_cache)
    elseif backend == :rocm
        build_rocm_field_evaluation_cache(cpu_field_cache)
    else
        cpu_field_cache
    end
    device_singular_cache = if backend == :cuda
        BeatEngineCore.build_cuda_singular_correction_cache(singular_cache, p1_space, dp0_space)
    elseif backend == :rocm
        build_rocm_singular_correction_cache(singular_cache)
    else
        nothing
    end
    device_image_singular_cache = backend == :cuda && symmetry_mode != :off ?
                                  build_cuda_image_singular_correction_cache(
        mesh, p1_space, dp0_space, singular_order, eachindex(mesh.faces), symmetry_mode,
    ) : nothing
    device_identity_cache = if backend == :cuda
        build_cuda_burton_miller_identity_cache(identity_p1_p1, identity_p1_dp0, FloatType)
    elseif backend == :rocm
        build_rocm_burton_miller_identity_cache(identity_p1_p1, identity_p1_dp0, FloatType)
    else
        nothing
    end
    cpu_blas_threads = backend == :cpu ?
                       configure_beat_cpu_blas_threads!(p1_space.global_dof_count) :
                       BLAS.get_num_threads()
    rule_cache = Dict{Int,Any}(base_order => base_rule)
    identity_cache = Dict{Int,Any}(base_order => (identity_p1_p1, identity_p1_dp0))
    cpu_field_cache_by_order = Dict{Int,Any}(base_order => cpu_field_cache)
    cpu_assembly_cache_by_order = Dict{Int,Any}()
    outputs = get(request, "outputs", Any[])
    cancel_path = get(request, "cancel_path", nothing)
    cancel_requested() = cancel_path !== nothing && isfile(String(cancel_path))
    solved_count = 0
    try
        for (frequency_index, raw_frequency) in enumerate(request["frequencies_hz"])
            cancel_requested() && return (cancelled=true, solved_count=solved_count)
            frequency_hz = FloatType(raw_frequency)
            omega = FloatType(2pi) * frequency_hz
            wavenumber = omega / sound_speed
            selection = exterior_quadrature_selection(
                options, mesh, frequency_hz, sound_speed, base_order, backend,
            )
            rule = backend == :cpu ? get!(rule_cache, selection.order) do
                triangle_rule(FloatType, selection.order)
            end : base_rule
            selected_identity = backend == :cpu ? get!(identity_cache, selection.order) do
                (
                    assemble_l2_identity_matrix(
                        mesh, p1_space, dp0_space, rule, :p1, :p1; symmetry_mode=symmetry_mode,
                    ),
                    assemble_l2_identity_matrix(
                        mesh, p1_space, dp0_space, rule, :p1, :dp0; symmetry_mode=symmetry_mode,
                    ),
                )
            end : (identity_p1_p1, identity_p1_dp0)
            selected_field_cache = backend == :cpu ? get!(cpu_field_cache_by_order, selection.order) do
                build_field_evaluation_cache(mesh, rule; symmetry_mode=symmetry_mode)
            end : field_cache
            selected_cpu_cache = backend == :cpu ? get!(cpu_assembly_cache_by_order, selection.order) do
                build_beat_cpu_assembly_cache(
                    mesh,
                    p1_space,
                    dp0_space,
                    rule;
                    singular_order=singular_order,
                    symmetry_mode=symmetry_mode,
                )
            end : nothing
            assembly_started = time_ns()
            operators = assemble_regular_galerkin_operators(
                mesh,
                p1_space,
                dp0_space,
                wavenumber,
                rule;
                skip_singular=false,
                singular_order=singular_order,
                backend=backend,
                device_cache=device_cache,
                return_device=accelerator_backend,
                accelerator_quadrature=accelerator_backend,
                singular_cache=singular_cache,
                cpu_cache=selected_cpu_cache,
                device_singular_cache=device_singular_cache,
                device_image_singular_cache=device_image_singular_cache,
                symmetry_mode=symmetry_mode,
            )
            assembly_s = (time_ns() - assembly_started) / 1.0e9
            solve_started = time_ns()
            cpu_system = backend == :cpu ? build_burton_miller_neumann_cpu_system(
                operators, selected_identity[1], selected_identity[2], wavenumber,
            ) : nothing
            pressures = Vector{Vector{Complex{FloatType}}}()
            neumann_values = Vector{Vector{Complex{FloatType}}}()
            for excitation in excitations
                neumann = exterior_neumann(mesh, excitation, density, omega)
                pressure = backend == :cpu ?
                           solve_burton_miller_neumann_cpu_system(cpu_system, neumann, FloatType) :
                           solve_burton_miller_neumann(
                    operators, device_identity_cache, neumann, wavenumber,
                )
                push!(pressures, Complex{FloatType}.(pressure))
                push!(neumann_values, neumann)
            end
            solve_s = (time_ns() - solve_started) / 1.0e9
            quantities = Dict{String,Any}[]
            field_s = 0.0
            for output in outputs
                quantity = String(output["quantity"])
                if quantity == "exterior_pressure"
                    field_started = time_ns()
                    raw_points = get(get(output, "options", Dict{String,Any}()), "points_m", Any[])
                    isempty(raw_points) && error("exterior_pressure output requires options.points_m.")
                    points = [SVector{3,FloatType}(FloatType.(point)) for point in raw_points]
                    values = [
                        exterior_field(
                            points,
                            mesh,
                            pressure,
                            neumann,
                            wavenumber,
                            selected_field_cache,
                            backend,
                        ) for (pressure, neumann) in zip(pressures, neumann_values)
                    ]
                    push!(
                        quantities,
                        quantity_wire(
                            output,
                            rows(values, FloatType),
                            "Pa",
                            ["excitation", "observation"],
                            metadata=Dict("points_m" => raw_points),
                        ),
                    )
                    field_s += (time_ns() - field_started) / 1.0e9
                elseif quantity == "bem_boundary_pressure"
                    push!(
                        quantities,
                        quantity_wire(
                            output,
                            rows(pressures, FloatType),
                            "Pa",
                            ["excitation", "bem_node"],
                        ),
                    )
                elseif quantity == "bem_boundary_neumann"
                    push!(
                        quantities,
                        quantity_wire(
                            output,
                            rows(neumann_values, FloatType),
                            "Pa/m",
                            ["excitation", "bem_face"],
                        ),
                    )
                elseif quantity == "radiation_impedance"
                    impedance_by_component = Complex{FloatType}[]
                    pressure_by_component = Dict(
                        excitation.component_id => pressure
                        for (excitation, pressure) in zip(excitations, pressures)
                    )
                    excitation_by_component = Dict(
                        excitation.component_id => excitation for excitation in excitations
                    )
                    for component in components
                        component_id = String(component["id"])
                        push!(
                            impedance_by_component,
                            exterior_component_impedance(
                                mesh,
                                pressure_by_component[component_id],
                                excitation_by_component[component_id],
                                symmetry_mode,
                                FloatType,
                            ),
                        )
                    end
                    push!(
                        quantities,
                        quantity_wire(
                            output,
                            impedance_by_component,
                            "Pa*s/m^3",
                            ["radiator"],
                        ),
                    )
                else
                    error("Unsupported exterior output quantity: $quantity")
                end
            end
            diagnostics = Dict{String,Any}(
                "precision" => precision_name,
                "bem_backend" => String(backend),
                "symmetry" => String(symmetry_mode),
                "formulation" => "exterior_burton_miller_neumann",
                "linear_solver" => backend == :cpu ?
                                   "cpu_dense_lu" :
                                   backend == :cuda ? "cuda_dense_lu" : "rocm_rocsolver_dense_lu",
                "bounded_region_count" => 0,
                "interface_count" => 0,
                "regular_quadrature_mode" => selection.mode,
                "regular_quadrature_order" => selection.order,
                "blas_threads" => cpu_blas_threads,
                "timings" => Dict(
                    "assembly_s" => assembly_s,
                    "solve_s" => solve_s,
                    "field_s" => field_s,
                    "mesh_setup_s" => frequency_index == 1 ? mesh_setup_s : 0.0,
                    "cache_setup_s" => 0.0,
                ),
            )
            result = Dict(
                "schema_version" => 1,
                "freq_hz" => frequency_hz,
                "excitation_port_ids" => excitation_port_ids,
                "quantities" => quantities,
                "diagnostics" => diagnostics,
            )
            println(JSON.json(event_mode ? Dict("type" => "result", "result" => result) : result))
            flush(stdout)
            release_operator_storage!(operators)
            solved_count = frequency_index
        end
    finally
        if device_identity_cache !== nothing
            backend == :cuda ?
            release_cuda_burton_miller_identity_cache!(device_identity_cache) :
            release_rocm_burton_miller_identity_cache!(device_identity_cache)
        end
        device_image_singular_cache === nothing ||
            release_cuda_image_singular_correction_cache!(device_image_singular_cache)
        if backend == :rocm
            device_singular_cache === nothing ||
                release_rocm_singular_correction_cache!(device_singular_cache)
            field_cache === cpu_field_cache || release_rocm_field_evaluation_cache!(field_cache)
            device_cache === nothing || release_rocm_regular_assembly_cache!(device_cache)
        end
    end
    return (cancelled=cancel_requested(), solved_count=solved_count)
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
    plane_wave_terminations = NamedTuple[]
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
            boundary_kind = String(boundary["kind"])
            if boundary_kind == "plane_wave_tube_termination"
                isempty(parameters) || error(
                    "Plane-wave tube termination $(repr(boundary_id)) does not accept parameters.",
                )
                push!(
                    plane_wave_terminations,
                    (boundary_id=boundary_id, tag=solver_tag),
                )
            end
            if haskey(parameters, "wall_impedance")
                boundary_kind == "rigid" || error(
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
        plane_wave_terminations=plane_wave_terminations,
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

function solve_interior_request(request, system, bounded_regions; event_mode=false)
    isempty(bounded_regions) && error("Interior FEM solving requires at least one bounded region.")
    isempty(system["interfaces"]) || error("Interior FEM solving does not accept FEM-BEM interfaces.")
    meshes = system["meshes"]
    boundaries = system["boundaries"]
    components = system["components"]
    ports = system["excitation_ports"]
    solver_options = get(request, "solver_options", Dict{String,Any}())
    precision_name = lowercase(String(get(solver_options, "precision", "float64")))
    FloatType = if precision_name in ("float32", "complex64")
        Float32
    elseif precision_name in ("float64", "complex128")
        Float64
    else
        error("Unsupported interior FEM precision: $precision_name. Expected float32 or float64.")
    end
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

    reference_sound_speed = Float64(bounded_regions[1]["sound_speed_m_per_s"])
    reference_density = Float64(bounded_regions[1]["density_kg_per_m3"])
    for region in bounded_regions
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
            "Interior FEM solving currently requires one shared sound speed and density.",
        )
    end
    sound_speed = FloatType(reference_sound_speed)
    density = FloatType(reference_density)

    mesh_setup_started = time_ns()
    fem_domains = aggregate_fem_domains(meshes, bounded_regions, boundaries, FloatType)
    fem_mesh = fem_domains.mesh
    symmetry_tolerance = symmetry_plane_tolerance(fem_mesh.vertices)
    fem_mesh = VolumeMesh(
        snap_symmetry_plane_vertices(
            fem_mesh.vertices,
            symmetry_mode;
            tolerance=symmetry_tolerance,
        ),
        fem_mesh.tetrahedra,
        fem_mesh.tetra_physical_tags,
        fem_mesh.boundary_faces,
        fem_mesh.boundary_physical_tags,
        fem_mesh.physical_names,
    )
    validate_volume_symmetry_fundamental_domain!(
        fem_mesh,
        symmetry_mode;
        tolerance=symmetry_tolerance,
    )
    mesh_setup_s = (time_ns() - mesh_setup_started) / 1.0e9

    cache_started = time_ns()
    stiffness, mass = assemble_p1_fem_matrices(fem_mesh)
    bulk_loss_mass = spdiagm(
        0 => FloatType.(fem_domains.bulk_loss_factor_by_vertex),
    ) * mass
    wall_operators = [
        begin
            face_indices = findall(==(Int(spec.tag)), fem_mesh.boundary_physical_tags)
            isempty(face_indices) && error(
                "Wall-impedance boundary $(repr(spec.boundary_id)) contains no selected FEM faces.",
            )
            (
                boundary_id=spec.boundary_id,
                matrix=assemble_boundary_mass_matrix(
                    fem_mesh,
                    face_indices,
                    collect(1:length(fem_mesh.vertices)),
                ),
                thickness_m=FloatType(spec.thickness_m),
                flow_resistivity_pa_s_per_m2=FloatType(spec.flow_resistivity_pa_s_per_m2),
            )
        end
        for spec in fem_domains.wall_impedances
    ]
    termination_operators = [
        begin
            face_indices = findall(==(Int(spec.tag)), fem_mesh.boundary_physical_tags)
            isempty(face_indices) && error(
                "Plane-wave tube termination $(repr(spec.boundary_id)) contains no selected FEM faces.",
            )
            (
                boundary_id=spec.boundary_id,
                matrix=assemble_boundary_mass_matrix(
                    fem_mesh,
                    face_indices,
                    collect(1:length(fem_mesh.vertices)),
                ),
            )
        end
        for spec in fem_domains.plane_wave_terminations
    ]
    empty_bem_mesh = BoundaryMesh(
        SVector{3,FloatType}[],
        NTuple{3,Int}[],
        Int[],
    )
    transducers, transducer_index_by_component_id = electrodynamic_transducers_from_wire(
        components,
        boundaries,
        fem_domains.fem_boundary_tag_by_id,
        Dict{String,Int}(),
        fem_mesh,
        empty_bem_mesh,
        FloatType,
        symmetry_mode,
    )
    transducer_operators = assemble_transducer_operators(
        fem_mesh,
        empty_bem_mesh,
        transducers,
    )
    cache_setup_s = (time_ns() - cache_started) / 1.0e9

    excitation_port_ids = String.(request["excitation_port_ids"])
    isempty(excitation_port_ids) && error("Interior FEM solving requires at least one excitation port.")
    excitations = NamedTuple[]
    for port_id in excitation_port_ids
        port = object_by_id(ports, port_id, "excitation port")
        component = object_by_id(components, String(port["component_id"]), "component")
        port_kind = String(port["kind"])
        component_kind = String(component["kind"])
        if port_kind == "normal_velocity" && component_kind == "ideal_velocity_source"
            parameters = get(component, "parameters", Dict{String,Any}())
            raw_weights = get(parameters, "boundary_motion_weights", Dict{String,Any}())
            tags = Int[]
            weights = FloatType[]
            for boundary_id_value in component["boundary_ids"]
                boundary_id = String(boundary_id_value)
                boundary = object_by_id(boundaries, boundary_id, "boundary")
                String(boundary["kind"]) == "moving" || error(
                    "Prescribed-velocity component $(repr(String(component["id"]))) references " *
                    "non-moving boundary $(repr(boundary_id)).",
                )
                tag = get(fem_domains.fem_boundary_tag_by_id, boundary_id, 0)
                tag > 0 || error(
                    "Interior prescribed-velocity boundary $(repr(boundary_id)) did not resolve to a FEM tag.",
                )
                weight = FloatType(get(raw_weights, boundary_id, 1))
                isfinite(weight) && weight > zero(FloatType) || error(
                    "Prescribed-velocity boundary weights must be finite and positive.",
                )
                push!(tags, tag)
                push!(weights, weight)
            end
            isempty(tags) && error(
                "Prescribed-velocity component $(repr(String(component["id"]))) has no moving FEM boundary.",
            )
            push!(excitations, (kind=:normal_velocity, tags=tags, weights=weights, transducer_index=0))
        elseif port_kind == "voltage" && component_kind == "electrodynamic_transducer"
            transducer_index = get(transducer_index_by_component_id, String(component["id"]), 0)
            transducer_index > 0 || error(
                "Voltage excitation $(repr(port_id)) did not resolve to an electrodynamic transducer.",
            )
            push!(
                excitations,
                (kind=:voltage, tags=Int[], weights=FloatType[], transducer_index=transducer_index),
            )
        else
            error("Interior excitation port $(repr(port_id)) is incompatible with its component.")
        end
    end

    outputs = get(request, "outputs", Any[])
    frequencies = Float64.(request["frequencies_hz"])
    cancel_path = get(request, "cancel_path", nothing)
    cancel_requested() = cancel_path !== nothing && isfile(String(cancel_path))
    solved_count = 0
    for (frequency_index, frequency_hz_value) in enumerate(frequencies)
        cancel_requested() && return (cancelled=true, solved_count=solved_count)
        frequency_hz = FloatType(frequency_hz_value)
        omega = FloatType(2pi) * frequency_hz
        wavenumber = omega / sound_speed
        assembly_started = time_ns()
        fem_system = assemble_fem_dynamic_stiffness(
            stiffness,
            mass,
            wavenumber;
            bulk_loss_mass=bulk_loss_mass,
        )
        for operator in wall_operators
            admittance = miki_rigid_backed_surface_admittance(
                frequency_hz,
                sound_speed,
                density,
                operator.thickness_m,
                operator.flow_resistivity_pa_s_per_m2,
            )
            fem_system -= Complex{FloatType}(0, density * omega) * admittance .* operator.matrix
        end
        for operator in termination_operators
            fem_system -= Complex{FloatType}(0, wavenumber) .* operator.matrix
        end

        fem_count = length(fem_mesh.vertices)
        transducer_count = length(transducers)
        mechanical_range = transducer_count == 0 ?
                           (1:0) :
                           ((fem_count + 1):(fem_count + transducer_count))
        electrical_range = transducer_count == 0 ?
                           (1:0) :
                           ((fem_count + transducer_count + 1):(fem_count + 2 * transducer_count))
        system_order = fem_count + 2 * transducer_count
        coupled = if transducer_count == 0
            SparseMatrixCSC{ComplexF64,Int}(fem_system)
        else
            normal_derivative_scale = Complex{FloatType}(0, density * omega)
            fem_motion = -normal_derivative_scale .* Complex{FloatType}.(
                transducer_operators.fem_surface,
            )
            fem_force = -Complex{FloatType}.(transpose(transducer_operators.fem_force))
            mechanical_impedance = Complex{FloatType}[
                Complex{FloatType}(
                    transducer.rms_n_s_per_m,
                    -omega * transducer.mmd_kg + inv(omega * transducer.cms_m_per_n),
                )
                for transducer in transducers
            ]
            electrical_impedance = Complex{FloatType}[
                Complex{FloatType}(transducer.re_ohm, -omega * transducer.le_h)
                for transducer in transducers
            ]
            force_factor = Complex{FloatType}[transducer.bl_n_per_a for transducer in transducers]
            sparse_system = [
                fem_system fem_motion spzeros(Complex{FloatType}, fem_count, transducer_count)
                fem_force spdiagm(0 => mechanical_impedance) -spdiagm(0 => force_factor)
                spzeros(Complex{FloatType}, transducer_count, fem_count) spdiagm(0 => force_factor) spdiagm(0 => electrical_impedance)
            ]
            SparseMatrixCSC{ComplexF64,Int}(sparse_system)
        end
        factorization = lu(coupled)
        assembly_s = (time_ns() - assembly_started) / 1.0e9

        rhs = zeros(ComplexF64, system_order, length(excitations))
        for (column, excitation) in enumerate(excitations)
            if excitation.kind == :normal_velocity
                for (tag, weight) in zip(excitation.tags, excitation.weights)
                    rhs[1:fem_count, column] .+= ComplexF64.(
                        assemble_prescribed_velocity_load(
                            fem_mesh,
                            tag,
                            density,
                            omega,
                            Complex{FloatType}(weight, 0),
                        ),
                    )
                end
            else
                rhs[first(electrical_range) + excitation.transducer_index - 1, column] =
                    ComplexF64(transducer_reference_voltage_v)
            end
        end
        solve_started = time_ns()
        solution = factorization \ rhs
        solve_s = (time_ns() - solve_started) / 1.0e9
        fem_pressures = [
            Complex{FloatType}.(solution[1:fem_count, column])
            for column in axes(solution, 2)
        ]
        diaphragm_velocities = [
            Complex{FloatType}.(solution[mechanical_range, column])
            for column in axes(solution, 2)
        ]
        voice_coil_currents = [
            Complex{FloatType}.(solution[electrical_range, column])
            for column in axes(solution, 2)
        ]

        quantities = Dict{String,Any}[]
        for output in outputs
            quantity = String(output["quantity"])
            if quantity == "fem_nodal_pressure"
                push!(
                    quantities,
                    quantity_wire(
                        output,
                        rows(fem_pressures, FloatType),
                        "Pa",
                        ["excitation", "fem_node"],
                        metadata=Dict(
                            "mesh_ids" => [domain.mesh_id for domain in fem_domains.domains],
                            "region_ids" => [domain.id for domain in fem_domains.domains],
                            "node_offsets" => [domain.vertex_offset for domain in fem_domains.domains],
                            "node_counts" => [domain.vertex_count for domain in fem_domains.domains],
                        ),
                    ),
                )
            elseif quantity == "diaphragm_velocity"
                push!(
                    quantities,
                    quantity_wire(
                        output,
                        rows(diaphragm_velocities, FloatType),
                        "m/s",
                        ["excitation", "transducer"],
                        metadata=Dict("component_ids" => [transducer.id for transducer in transducers]),
                    ),
                )
            elseif quantity == "voice_coil_current"
                push!(
                    quantities,
                    quantity_wire(
                        output,
                        rows(voice_coil_currents, FloatType),
                        "A",
                        ["excitation", "transducer"],
                        metadata=Dict("component_ids" => [transducer.id for transducer in transducers]),
                    ),
                )
            else
                error("Unsupported interior FEM output quantity: $quantity")
            end
        end
        relative_residual = maximum(
            norm(coupled * solution[:, column] - rhs[:, column]) /
            max(norm(rhs[:, column]), eps(Float64))
            for column in axes(solution, 2)
        )
        diagnostics = Dict{String,Any}(
            "precision" => precision_name,
            "linear_backend" => "cpu",
            "linear_solver" => "cpu_umfpack_sparse_lu_fp64",
            "symmetry" => String(symmetry_mode),
            "formulation" => "interior_fem_sparse",
            "full_system_order" => system_order,
            "solved_system_order" => system_order,
            "bounded_region_count" => length(bounded_regions),
            "interface_count" => 0,
            "transducer_count" => transducer_count,
            "transducer_reference_voltage_v" => transducer_reference_voltage_v,
            "plane_wave_termination_boundary_ids" => [
                operator.boundary_id for operator in termination_operators
            ],
            "wall_impedance_boundary_ids" => [operator.boundary_id for operator in wall_operators],
            "relative_residual" => relative_residual,
            "timings" => Dict(
                "assembly_s" => assembly_s,
                "solve_s" => solve_s,
                "field_s" => 0.0,
                "mesh_setup_s" => frequency_index == 1 ? mesh_setup_s : 0.0,
                "cache_setup_s" => frequency_index == 1 ? cache_setup_s : 0.0,
            ),
        )
        result = Dict(
            "schema_version" => 1,
            "freq_hz" => frequency_hz_value,
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
        solved_count += 1
    end
    return (cancelled=false, solved_count=solved_count)
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
    isempty(unbounded_regions) && return solve_interior_request(
        request,
        system,
        bounded_regions;
        event_mode=event_mode,
    )
    length(unbounded_regions) == 1 || error("Coupled backend currently requires exactly one unbounded region.")
    unbounded_region = only(unbounded_regions)
    isempty(bounded_regions) && return solve_exterior_request(
        request,
        system,
        unbounded_region;
        event_mode=event_mode,
    )
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
    bem_backend in (:cpu, :cuda, :rocm) ||
        error("Unsupported coupled BEM backend: $bem_backend. Expected cpu, cuda, or rocm.")
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
        symmetry_plane_tolerance(fem_mesh.vertices),
        symmetry_plane_tolerance(bem_mesh.vertices),
    )
    fem_mesh = VolumeMesh(
        snap_symmetry_plane_vertices(
            fem_mesh.vertices,
            symmetry_mode;
            tolerance=symmetry_tolerance,
        ),
        fem_mesh.tetrahedra,
        fem_mesh.tetra_physical_tags,
        fem_mesh.boundary_faces,
        fem_mesh.boundary_physical_tags,
        fem_mesh.physical_names,
    )
    bem_mesh = snap_symmetry_planes(
        bem_mesh,
        symmetry_mode;
        tolerance=symmetry_tolerance,
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
    prescribed_bem_rows = Int[]
    prescribed_bem_cols = Int[]
    prescribed_bem_values = FloatType[]
    prescribed_bem_count = 0
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
            fem_boundary_tags = Int[]
            fem_boundary_weights = FloatType[]
            bem_boundary_tags = Int[]
            bem_boundary_weights = FloatType[]
            for boundary in candidate_boundaries
                boundary_id = String(boundary["id"])
                String(boundary["kind"]) == "moving" || error(
                    "Prescribed-velocity component $(repr(String(component["id"]))) " *
                    "references non-moving boundary $(repr(boundary_id)).",
                )
                weight = FloatType(get(raw_weights, boundary_id, 1))
                isfinite(weight) && weight > zero(FloatType) || error(
                    "Prescribed-velocity boundary motion weights must be finite and greater than zero.",
                )
                if haskey(fem_domains.fem_boundary_tag_by_id, boundary_id)
                    tag = fem_domains.fem_boundary_tag_by_id[boundary_id]
                    any(==(tag), fem_mesh.boundary_physical_tags) || error(
                        "Moving boundary tag $tag is not on the selected FEM volume groups.",
                    )
                    push!(fem_boundary_tags, tag)
                    push!(fem_boundary_weights, weight)
                elseif haskey(bem_boundary_tag_by_id, boundary_id)
                    tag = bem_boundary_tag_by_id[boundary_id]
                    any(==(tag), bem_mesh.physical_tags) || error(
                        "Moving boundary tag $tag is not present in the exterior BEM mesh.",
                    )
                    push!(bem_boundary_tags, tag)
                    push!(bem_boundary_weights, weight)
                else
                    error(
                        "Prescribed-velocity component $(repr(String(component["id"]))) " *
                        "references boundary $(repr(boundary_id)) outside the active acoustic regions.",
                    )
                end
            end
            isempty(fem_boundary_tags) && isempty(bem_boundary_tags) && error(
                "Prescribed-velocity component $(repr(String(component["id"]))) " *
                "must own at least one moving FEM or BEM boundary.",
            )
            bem_source_index = 0
            if !isempty(bem_boundary_tags)
                prescribed_bem_count += 1
                bem_source_index = prescribed_bem_count
                weight_by_tag = Dict(zip(bem_boundary_tags, bem_boundary_weights))
                for face_index in eachindex(bem_mesh.faces)
                    weight = get(
                        weight_by_tag,
                        bem_mesh.physical_tags[face_index],
                        zero(FloatType),
                    )
                    weight == zero(FloatType) && continue
                    push!(prescribed_bem_rows, face_index)
                    push!(prescribed_bem_cols, bem_source_index)
                    push!(prescribed_bem_values, weight)
                end
            end
            push!(
                excitations,
                (
                    kind=:normal_velocity,
                    fem_boundary_tags=fem_boundary_tags,
                    fem_boundary_weights=fem_boundary_weights,
                    bem_source_index=bem_source_index,
                    transducer_index=0,
                    amplitude=Complex{FloatType}(1, 0),
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
    prescribed_bem_normal_velocity = sparse(
        prescribed_bem_rows,
        prescribed_bem_cols,
        prescribed_bem_values,
        length(bem_mesh.faces),
        prescribed_bem_count,
    )

    quadrature_order = Int(get(solver_options, "quadrature_order", 2))
    singular_order = Int(get(solver_options, "singular_order", 2))
    validation_diagnostics = Bool(get(solver_options, "validation_diagnostics", true))
    cache_frequency_invariant = Bool(get(solver_options, "cache_frequency_invariant", true))
    static_condensation_requested = Bool(
        get(
            solver_options,
            "static_condensation",
            !validation_diagnostics,
        ),
    )
    static_condensation = static_condensation_requested
    transducer_fem_vertices = isempty(transducers) ?
                              Int[] :
                              unique(findnz(transducer_operators.fem_surface)[1])
    retained_fem_vertices = sort(
        unique(vcat(interface_map.fem_vertex_indices, transducer_fem_vertices)),
    )
    # CPU condensation is deliberately isolated from the accelerator implementations. CUDA
    # continues to use cuDSS and ROCm continues to use the hybrid CPU-Schur/ROCm-dense path in
    # BeatEngineCoupled.
    use_condensed_solver = static_condensation && bem_backend == :cpu
    quadrature_selections = if use_condensed_solver
        mode = lowercase(String(get(solver_options, "regular_quadrature_mode", "fixed")))
        mode in ("fixed", "wavelength") || error(
            "Unsupported regular quadrature mode: $mode. Expected fixed or wavelength.",
        )
        [
            mode == "fixed" ?
            (
                order=quadrature_order, base_order=quadrature_order, mode=mode,
                mesh_stat=nothing, area=nothing, length=nothing, kh=nothing,
                q1_max=nothing, q2_max=nothing,
            ) :
            merge(
                wavelength_quadrature_order(
                    bem_mesh.areas,
                    FloatType(frequency_value),
                    sound_speed,
                    quadrature_order;
                    mesh_stat=lowercase(String(get(solver_options, "wavelength_mesh_stat", "p90"))),
                    q1_max=Float64(get(solver_options, "wavelength_kh_q1_max", 0.0)),
                    q2_max=Float64(get(solver_options, "wavelength_kh_q2_max", 2.0)),
                ),
                (mode=mode,),
            )
            for frequency_value in request["frequencies_hz"]
        ]
    else
        nothing
    end
    coupled_cache = nothing
    cache_setup_s = 0.0
    solved_count = 0
    cancelled = cancel_requested()
    cancelled && return (cancelled=true, solved_count=solved_count)
    if cache_frequency_invariant
        cache_setup_started = time_ns()
        coupled_cache = if use_condensed_solver
            prepare_condensed_coupled_cache(
                fem_mesh,
                bem_mesh,
                interface_map;
                quadrature_order=quadrature_order,
                regular_quadrature_orders=sort(
                    unique(selection.order for selection in quadrature_selections),
                ),
                singular_order=singular_order,
                symmetry_mode=symmetry_mode,
                retained_fem_vertices=retained_fem_vertices,
                bulk_loss_factor_by_vertex=fem_domains.bulk_loss_factor_by_vertex,
                wall_impedances=fem_domains.wall_impedances,
            )
        else
            prepare_coupled_cache(
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
        end
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
        coupled_system = if use_condensed_solver
            build_condensed_coupled_system(
                fem_mesh,
                bem_mesh,
                interface_map,
                frequency_hz,
                sound_speed,
                density;
                quadrature_order=quadrature_order,
                regular_quadrature_order=quadrature_selections[frequency_index].order,
                singular_order=singular_order,
                cache=coupled_cache,
                validation_diagnostics=validation_diagnostics,
                symmetry_mode=symmetry_mode,
                bulk_loss_factor_by_vertex=fem_domains.bulk_loss_factor_by_vertex,
                wall_impedances=fem_domains.wall_impedances,
                transducers=transducers,
                transducer_operators=transducer_operators,
                prescribed_bem_normal_velocity=prescribed_bem_normal_velocity,
            )
        else
            build_coupled_system(
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
                prescribed_bem_normal_velocity=prescribed_bem_normal_velocity,
            )
        end
        assembly_s = (time_ns() - assembly_started) / 1.0e9
        solve_started = time_ns()
        solutions = use_condensed_solver ?
                    solve_condensed_coupled_excitations(coupled_system, excitations) :
                    solve_coupled_excitations(coupled_system, excitations)
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
                        metadata=Dict(
                            "mesh_ids" => String.(unbounded_region["mesh_ids"]),
                            "vertex_offsets" => [
                                bem_domain.vertex_offset_by_mesh_id[String(mesh_id)]
                                for mesh_id in unbounded_region["mesh_ids"]
                            ],
                            "vertex_counts" => [
                                bem_domain.vertex_count_by_mesh_id[String(mesh_id)]
                                for mesh_id in unbounded_region["mesh_ids"]
                            ],
                        ),
                    ),
                )
            elseif quantity == "bem_boundary_neumann"
                push!(
                    quantities,
                    quantity_wire(
                        output,
                        rows([solution.bem_neumann for solution in solutions], FloatType),
                        "Pa/m",
                        ["excitation", "bem_face"],
                        metadata=Dict(
                            "mesh_ids" => String.(unbounded_region["mesh_ids"]),
                            "face_offsets" => [
                                bem_domain.face_offset_by_mesh_id[String(mesh_id)]
                                for mesh_id in unbounded_region["mesh_ids"]
                            ],
                            "face_counts" => [
                                bem_domain.face_count_by_mesh_id[String(mesh_id)]
                                for mesh_id in unbounded_region["mesh_ids"]
                            ],
                        ),
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
                    elseif bem_backend == :rocm
                        evaluate_galerkin_field_rocm(
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
                        metadata=Dict(
                            "points_m" => raw_points,
                            "observation_domains" => get(
                                options,
                                "observation_domains",
                                Any[],
                            ),
                        ),
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
            "linear_solver" => if coupled_system.formulation == :fem_interface_condensed
                if coupled_system.linear_backend == :rocm
                    "rocm_hybrid_cpu_sparse_schur_plus_rocsolver_dense_lu"
                elseif coupled_system.linear_backend == :cuda
                    "cuda_cudss_schur_plus_dense_lu"
                elseif coupled_system.condensation.backend == :cpu_noop
                    "cpu_dense_lu_noop_schur"
                else
                    "cpu_umfpack_schur_plus_dense_lu"
                end
            elseif coupled_system.linear_backend == :rocm
                "rocm_rocsolver_dense_lu"
            else
                "$(coupled_system.linear_backend)_dense_lu"
            end,
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
            "fem_condensation_backend" => isnothing(coupled_system.condensation) ?
                                          nothing :
                                          String(coupled_system.condensation.backend),
            "fem_schur_block_size" => isnothing(coupled_system.condensation) ||
                                      !hasproperty(
                coupled_system.condensation,
                :schur_block_size,
            ) ? 0 : coupled_system.condensation.schur_block_size,
            "fem_schur_thread_count" => isnothing(coupled_system.condensation) ||
                                        !hasproperty(
                coupled_system.condensation,
                :schur_thread_count,
            ) ? 0 : coupled_system.condensation.schur_thread_count,
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
                "fem_condensation_partition_s" => isnothing(coupled_system.condensation) ||
                                                   !hasproperty(
                    coupled_system.condensation.timings,
                    :partition_s,
                ) ? 0.0 : coupled_system.condensation.timings.partition_s,
                "fem_schur_extraction_s" => isnothing(coupled_system.condensation) ?
                                            0.0 :
                                            coupled_system.condensation.timings.schur_extraction_s,
                "fem_schur_upload_s" => isnothing(coupled_system.condensation) ||
                                        !hasproperty(
                    coupled_system.condensation.timings,
                    :upload_s,
                ) ? 0.0 : coupled_system.condensation.timings.upload_s,
                "fem_rhs_condensation_s" => maximum(
                    something(solution.fem_rhs_condensation_s, 0.0) for solution in solutions
                ),
                "fem_reconstruction_s" => maximum(
                    something(solution.fem_reconstruction_s, 0.0) for solution in solutions
                ),
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
        diagnostics["fem_interior_residual"] = use_condensed_solver ?
            maximum(solution.fem_interior_residual for solution in solutions) : nothing
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
        use_condensed_solver ? release_condensed_coupled_system!(coupled_system) :
        release_coupled_system!(coupled_system)
        solved_count = frequency_index
    end
    if coupled_cache !== nothing
        use_condensed_solver ? release_condensed_coupled_cache!(coupled_cache) :
        release_coupled_cache!(coupled_cache)
    end
    cancelled = cancelled || cancel_requested()
    return (cancelled=cancelled, solved_count=solved_count)
end

function reclaim_accelerator_memory_after_failure()
    try
        release_all_bem_field_evaluation_caches!()
    catch
        nothing
    end
    GC.gc(true)
    try
        cuda = BeatEngineCore.cuda_module()
        cuda.reclaim()
    catch
        nothing
    end
    try
        amdgpu = BeatEngineCore.amdgpu_module()
        amdgpu.reclaim()
    catch
        nothing
    end
    return nothing
end

function binary_dtype_name(::Type{Float32})
    return "float32"
end

function binary_dtype_name(::Type{Float64})
    return "float64"
end

function binary_dtype_name(::Type{Int64})
    return "int64"
end

function binary_dtype_name(::Type{ComplexF32})
    return "complex64"
end

function binary_dtype_name(::Type{ComplexF64})
    return "complex128"
end

function read_field_binary_array(request, request_path, name, ::Type{T}) where {T}
    Int(get(request, "binary_array_schema_version", 0)) == 1 ||
        error("Unsupported BEM field binary-array schema.")
    arrays = get(request, "binary_arrays", nothing)
    arrays isa AbstractDict || error("BEM field request is missing binary-array metadata.")
    descriptor = get(arrays, name, nothing)
    descriptor isa AbstractDict || error("BEM field request is missing binary array $(repr(name)).")
    String(get(descriptor, "dtype", "")) == binary_dtype_name(T) ||
        error("BEM field binary array $(repr(name)) has the wrong data type.")
    String(get(descriptor, "order", "")) == "C" ||
        error("BEM field binary array $(repr(name)) must use row-major order.")
    String(get(descriptor, "byte_order", "")) == "little" ||
        error("BEM field binary array $(repr(name)) must use little-endian byte order.")
    shape = Int.(get(descriptor, "shape", Int[]))
    all(>=(0), shape) || error("BEM field binary array $(repr(name)) has an invalid shape.")
    count = prod(shape)
    offset = Int(get(descriptor, "offset", -1))
    nbytes = Int(get(descriptor, "nbytes", -1))
    offset >= 0 || error("BEM field binary array $(repr(name)) has an invalid offset.")
    nbytes == count * sizeof(T) ||
        error("BEM field binary array $(repr(name)) has inconsistent byte metadata.")
    filename = String(get(descriptor, "file", ""))
    !isempty(filename) && basename(filename) == filename ||
        error("BEM field binary array $(repr(name)) has an invalid file path.")
    path = joinpath(dirname(request_path), filename)
    isfile(path) || error("BEM field binary array file does not exist: $filename")
    filesize(path) >= offset + nbytes ||
        error("BEM field binary array $(repr(name)) is incomplete.")
    values = Vector{T}(undef, count)
    open(path, "r") do io
        seek(io, offset)
        read!(io, values)
    end
    return values, shape
end

function write_field_binary_result(request, request_path, values)
    filename = String(get(request, "binary_result_file", ""))
    !isempty(filename) && basename(filename) == filename ||
        error("BEM field request has an invalid binary result path.")
    scalar_type = typeof(real(zero(eltype(values))))
    scalar_type in (Float32, Float64) || error("Unsupported BEM field result precision: $scalar_type")
    array = Complex{scalar_type}.(values)
    path = joinpath(dirname(request_path), filename)
    open(path, "w") do io
        write(io, array)
    end
    return Dict(
        "file" => filename,
        "offset" => 0,
        "nbytes" => sizeof(eltype(array)) * length(array),
        "dtype" => binary_dtype_name(eltype(array)),
        "shape" => collect(size(array)),
        "order" => "C",
        "byte_order" => "little",
    )
end

function release_bem_field_evaluation_cache!(cache_key::String)
    entry = pop!(BEM_FIELD_EVALUATION_CACHES, cache_key, nothing)
    if entry !== nothing
        if entry.backend == :cuda
            BeatEngineCoupled._unsafe_free_cuda_fields!(
                entry.field_cache,
                (
                    :source_points,
                    :source_normals,
                    :source_weights,
                    :source_faces,
                    :source_elements,
                    :basis_values,
                ),
            )
        elseif entry.backend == :rocm
            release_rocm_field_evaluation_cache!(entry.field_cache)
        end
    end
    filter!(!=(cache_key), BEM_FIELD_EVALUATION_CACHE_ORDER)
    return nothing
end

function release_all_bem_field_evaluation_caches!()
    for cache_key in copy(BEM_FIELD_EVALUATION_CACHE_ORDER)
        release_bem_field_evaluation_cache!(cache_key)
    end
    return nothing
end

function retained_bem_field_evaluation_cache(
    request,
    request_path,
    ::Type{T},
    backend,
    symmetry,
) where {T<:AbstractFloat}
    domain_key = strip(String(get(request, "domain_key", "")))
    isempty(domain_key) && error("Exterior field request is missing its domain cache key.")
    quadrature_order = Int(get(request, "quadrature_order", 2))
    cache_key = "$domain_key|$(T)|$backend|$symmetry|q$quadrature_order"
    cached = get(BEM_FIELD_EVALUATION_CACHES, cache_key, nothing)
    if cached !== nothing
        filter!(!=(cache_key), BEM_FIELD_EVALUATION_CACHE_ORDER)
        push!(BEM_FIELD_EVALUATION_CACHE_ORDER, cache_key)
        return cached
    end

    vertex_values, vertex_shape = read_field_binary_array(
        request,
        request_path,
        "boundary_points_m",
        Float32,
    )
    length(vertex_shape) == 2 && vertex_shape[2] == 3 ||
        error("BEM boundary points must have shape (bem_node, 3).")
    vertices = [
        SVector{3,T}(
            T(vertex_values[3 * index - 2]),
            T(vertex_values[3 * index - 1]),
            T(vertex_values[3 * index]),
        ) for index in 1:vertex_shape[1]
    ]
    face_values, face_shape = read_field_binary_array(
        request,
        request_path,
        "boundary_triangles",
        Int64,
    )
    length(face_shape) == 2 && face_shape[2] == 3 ||
        error("BEM boundary triangles must have shape (bem_face, 3).")
    faces = [
        (
            Int(face_values[3 * index - 2]) + 1,
            Int(face_values[3 * index - 1]) + 1,
            Int(face_values[3 * index]) + 1,
        ) for index in 1:face_shape[1]
    ]
    mesh = BoundaryMesh(vertices, faces, ones(Int, length(faces)))
    rule = triangle_rule(T, quadrature_order)
    cpu_cache = build_field_evaluation_cache(mesh, rule; symmetry_mode=symmetry)
    field_cache = if backend == :cuda
        build_cuda_field_evaluation_cache(cpu_cache)
    elseif backend == :rocm
        build_rocm_field_evaluation_cache(cpu_cache)
    else
        cpu_cache
    end
    entry = (mesh=mesh, field_cache=field_cache, backend=backend)
    BEM_FIELD_EVALUATION_CACHES[cache_key] = entry
    push!(BEM_FIELD_EVALUATION_CACHE_ORDER, cache_key)
    while length(BEM_FIELD_EVALUATION_CACHE_ORDER) > MAX_BEM_FIELD_EVALUATION_CACHES
        release_bem_field_evaluation_cache!(first(BEM_FIELD_EVALUATION_CACHE_ORDER))
    end
    return entry
end

function evaluate_bem_field_request(request, request_path)
    precision_name = lowercase(String(get(request, "precision", "float32")))
    FloatType = precision_name == "float64" ? Float64 : precision_name == "float32" ? Float32 :
                error("Exterior field precision must be float32 or float64.")
    backend = Symbol(lowercase(String(get(request, "bem_backend", "cpu"))))
    backend in (:cpu, :cuda, :rocm) || error("Exterior field backend must be cpu, cuda, or rocm.")
    symmetry = BeatEngineCore.normalized_symmetry_mode(get(request, "symmetry", "off"))
    retained = retained_bem_field_evaluation_cache(request, request_path, FloatType, backend, symmetry)
    mesh = retained.mesh
    pressure, pressure_shape = read_field_binary_array(
        request,
        request_path,
        "boundary_pressure",
        Complex{FloatType},
    )
    normal_derivative, normal_shape = read_field_binary_array(
        request,
        request_path,
        "boundary_normal_derivative",
        Complex{FloatType},
    )
    length(pressure_shape) == 1 || error("Retained BEM pressure must be one-dimensional.")
    length(normal_shape) == 1 || error("Retained BEM normal derivative must be one-dimensional.")
    length(pressure) == length(mesh.vertices) ||
        error("Retained BEM pressure does not align with boundary vertices.")
    length(normal_derivative) == length(mesh.faces) ||
        error("Retained BEM normal derivative does not align with boundary faces.")
    point_values, point_shape = read_field_binary_array(
        request,
        request_path,
        "points_m",
        Float32,
    )
    length(point_shape) == 2 && point_shape[2] == 3 ||
        error("Exterior evaluation points must have shape (point, 3).")
    points = [
        SVector{3,FloatType}(
            FloatType(point_values[3 * index - 2]),
            FloatType(point_values[3 * index - 1]),
            FloatType(point_values[3 * index]),
        ) for index in 1:point_shape[1]
    ]
    frequency_hz = FloatType(request["frequency_hz"])
    sound_speed = FloatType(request["sound_speed_m_per_s"])
    frequency_hz > zero(FloatType) || error("Exterior field frequency must be positive.")
    sound_speed > zero(FloatType) || error("Exterior field sound speed must be positive.")
    field_cache = retained.field_cache
    wavenumber = FloatType(2pi) * frequency_hz / sound_speed
    if backend == :cuda
        return evaluate_galerkin_field_cuda(
            points,
            mesh,
            pressure,
            normal_derivative,
            wavenumber,
            field_cache,
        )
    elseif backend == :rocm
        return evaluate_galerkin_field_rocm(
            points,
            mesh,
            pressure,
            normal_derivative,
            wavenumber,
            field_cache,
        )
    end
    return evaluate_galerkin_field_cpu(
        points,
        mesh,
        pressure,
        normal_derivative,
        wavenumber,
        field_cache,
    )
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
            operation = String(get(submission, "operation", "solve"))
            if operation == "bem_field"
                values = evaluate_bem_field_request(request, request_path)
                println(
                    JSON.json(
                        Dict(
                            "type" => "field_result",
                            "values_binary" => write_field_binary_result(
                                request,
                                request_path,
                                values,
                            ),
                        ),
                    ),
                )
                println(JSON.json(Dict("type" => "completed")))
            elseif operation == "solve"
                release_all_bem_field_evaluation_caches!()
                outcome = solve_request(request; event_mode=true)
                event_type = outcome.cancelled ? "cancelled" : "completed"
                println(JSON.json(Dict("type" => event_type, "solved_count" => outcome.solved_count)))
            else
                error("Unsupported coupled worker operation: $operation")
            end
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
