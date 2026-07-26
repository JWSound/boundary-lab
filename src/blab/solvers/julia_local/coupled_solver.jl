#!/usr/bin/env julia

using JSON, LinearAlgebra, StaticArrays

include(joinpath(@__DIR__, "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled

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

function solve_request(request; event_mode=false)
    Int(get(request, "schema_version", 0)) == 1 || error("Unsupported system solve request schema.")
    system = request["compiled_system"]
    meshes = system["meshes"]
    regions = system["regions"]
    boundaries = system["boundaries"]
    components = system["components"]
    ports = system["excitation_ports"]
    interfaces = system["interfaces"]
    length(interfaces) == 1 || error("Coupled backend currently requires exactly one FEM-BEM interface.")

    bounded_regions = [region for region in regions if String(region["kind"]) == "bounded_air"]
    unbounded_regions = [region for region in regions if String(region["kind"]) == "unbounded_air"]
    length(bounded_regions) == 1 || error("Coupled backend currently requires exactly one bounded region.")
    length(unbounded_regions) == 1 || error("Coupled backend currently requires exactly one unbounded region.")
    bounded_region = only(bounded_regions)
    unbounded_region = only(unbounded_regions)
    length(bounded_region["mesh_ids"]) == 1 || error("Bounded region must contain exactly one FEM mesh.")
    length(unbounded_region["mesh_ids"]) == 1 || error("Unbounded region must contain exactly one BEM mesh.")
    fem_resource = object_by_id(meshes, String(only(bounded_region["mesh_ids"])), "mesh")
    bem_resource = object_by_id(meshes, String(only(unbounded_region["mesh_ids"])), "mesh")
    String(fem_resource["purpose"]) == "fem_volume" || error("Bounded region mesh must be a FEM volume.")
    String(bem_resource["purpose"]) == "bem_surface" || error("Unbounded region mesh must be a BEM surface.")

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

    mesh_setup_started = time_ns()
    full_fem_mesh = translated_volume_mesh(fem_resource, FloatType)
    selected_volume_tags = [
        Int(group["tag"])
        for group in bounded_region["volume_groups"]
        if String(group["mesh_id"]) == String(fem_resource["id"])
    ]
    volume_selection = restrict_volume_mesh(full_fem_mesh, selected_volume_tags)
    fem_mesh = volume_selection.mesh
    bem_mesh = translated_boundary_mesh(bem_resource, FloatType)
    interface_map = interface_map_from_wire(only(interfaces)["topology"], volume_selection)
    mesh_setup_s = (time_ns() - mesh_setup_started) / 1.0e9
    sound_speed = FloatType(bounded_region["sound_speed_m_per_s"])
    density = FloatType(bounded_region["density_kg_per_m3"])
    excitation_port_ids = String.(request["excitation_port_ids"])
    isempty(excitation_port_ids) && error("Coupled solve requires at least one excitation port.")
    radiator_tags = Int[]
    for port_id in excitation_port_ids
        port = object_by_id(ports, port_id, "excitation port")
        String(port["kind"]) == "normal_velocity" || error(
            "Coupled backend currently supports only normal_velocity excitation ports.",
        )
        component = object_by_id(components, String(port["component_id"]), "component")
        candidate_boundaries = [
            object_by_id(boundaries, String(boundary_id), "boundary")
            for boundary_id in component["boundary_ids"]
        ]
        bounded_boundaries = [
            boundary
            for boundary in candidate_boundaries
            if String(boundary["region_id"]) == String(bounded_region["id"])
        ]
        length(bounded_boundaries) == 1 || error(
            "Each excitation component must own exactly one moving boundary in the bounded region.",
        )
        radiator_tag = Int(only(bounded_boundaries)["group"]["tag"])
        any(==(radiator_tag), fem_mesh.boundary_physical_tags) || error(
            "Moving boundary tag $radiator_tag is not on the selected FEM volume groups.",
        )
        push!(radiator_tags, radiator_tag)
    end

    quadrature_order = Int(get(solver_options, "quadrature_order", 2))
    singular_order = Int(get(solver_options, "singular_order", 2))
    validation_diagnostics = Bool(get(solver_options, "validation_diagnostics", true))
    cache_frequency_invariant = Bool(get(solver_options, "cache_frequency_invariant", true))
    coupled_cache = nothing
    cache_setup_s = 0.0
    if cache_frequency_invariant
        cache_setup_started = time_ns()
        coupled_cache = prepare_coupled_cache(
            fem_mesh,
            bem_mesh,
            interface_map;
            quadrature_order=quadrature_order,
            singular_order=singular_order,
            bem_backend=bem_backend,
        )
        cache_setup_s = (time_ns() - cache_setup_started) / 1.0e9
    end
    outputs = get(request, "outputs", Any[])
    for (frequency_index, frequency_value) in enumerate(request["frequencies_hz"])
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
        )
        assembly_s = (time_ns() - assembly_started) / 1.0e9
        solve_started = time_ns()
        solutions = [
            solve_coupled_system(coupled_system, radiator_tag)
            for radiator_tag in radiator_tags
        ]
        solve_s = (time_ns() - solve_started) / 1.0e9
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
                        metadata=Dict("mesh_id" => String(fem_resource["id"])),
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
                        metadata=Dict("interface_id" => String(only(interfaces)["id"])),
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
            "linear_solver" => "dense_lu",
            "pressure_continuity_error" => maximum(
                solution.pressure_continuity_error for solution in solutions
            ),
            "flux_conservation_error" => maximum(
                solution.flux_conservation_error for solution in solutions
            ),
            "timings" => Dict(
                "assembly_s" => assembly_s,
                "solve_s" => solve_s,
                "field_s" => field_s,
                "mesh_setup_s" => frequency_index == 1 ? mesh_setup_s : 0.0,
                "cache_setup_s" => frequency_index == 1 ? cache_setup_s : 0.0,
                "fem_system_s" => coupled_system.timings.fem_system_s,
                "bem_operator_s" => coupled_system.timings.bem_operator_s,
                "bem_matrix_s" => coupled_system.timings.bem_matrix_s,
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
        coupled_system.owns_cache &&
            release_coupled_cache!(coupled_system.cache)
    end
    coupled_cache === nothing || release_coupled_cache!(coupled_cache)
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
            solve_request(request; event_mode=true)
            println(JSON.json(Dict("type" => "completed")))
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
