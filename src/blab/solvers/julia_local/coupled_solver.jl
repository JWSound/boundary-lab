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

function translated_volume_mesh(resource)
    scale = Float64(resource["scale_to_m"])
    translation = SVector{3,Float64}(Float64.(resource["translation_m"]))
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

function translated_boundary_mesh(resource)
    scale = Float64(resource["scale_to_m"])
    translation = SVector{3,Float64}(Float64.(resource["translation_m"]))
    mesh = load_gmsh22_with_tags(String(resource["file"]), scale)
    return BoundaryMesh([vertex + translation for vertex in mesh.vertices], mesh.faces, mesh.physical_tags)
end

function interface_map_from_wire(raw)
    return ConformingInterfaceMap(
        Int.(raw["fem_vertex_indices"]) .+ 1,
        Int.(raw["bem_vertex_indices"]) .+ 1,
        Int.(raw["fem_to_bem_vertex_indices"]) .+ 1,
        Int.(raw["fem_face_indices"]) .+ 1,
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
    array = ComplexF64.(values)
    flattened = row_major_values(array)
    return Dict(
        "dtype" => "complex128",
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

function rows(vectors)
    isempty(vectors) && return zeros(ComplexF64, 0, 0)
    return reduce(vcat, (reshape(ComplexF64.(values), 1, :) for values in vectors))
end

function solve_request(request)
    Int(get(request, "schema_version", 0)) == 1 || error("Unsupported system solve request schema.")
    system = request["compiled_system"]
    meshes = system["meshes"]
    regions = system["regions"]
    boundaries = system["boundaries"]
    components = system["components"]
    ports = system["excitation_ports"]
    interfaces = system["interfaces"]
    length(interfaces) == 1 || error("Reference coupled backend currently requires exactly one FEM-BEM interface.")

    bounded_regions = [region for region in regions if String(region["kind"]) == "bounded_air"]
    unbounded_regions = [region for region in regions if String(region["kind"]) == "unbounded_air"]
    length(bounded_regions) == 1 || error("Reference coupled backend currently requires exactly one bounded region.")
    length(unbounded_regions) == 1 || error("Reference coupled backend currently requires exactly one unbounded region.")
    bounded_region = only(bounded_regions)
    unbounded_region = only(unbounded_regions)
    length(bounded_region["mesh_ids"]) == 1 || error("Bounded region must contain exactly one FEM mesh.")
    length(unbounded_region["mesh_ids"]) == 1 || error("Unbounded region must contain exactly one BEM mesh.")
    fem_resource = object_by_id(meshes, String(only(bounded_region["mesh_ids"])), "mesh")
    bem_resource = object_by_id(meshes, String(only(unbounded_region["mesh_ids"])), "mesh")
    String(fem_resource["purpose"]) == "fem_volume" || error("Bounded region mesh must be a FEM volume.")
    String(bem_resource["purpose"]) == "bem_surface" || error("Unbounded region mesh must be a BEM surface.")

    fem_mesh = translated_volume_mesh(fem_resource)
    bem_mesh = translated_boundary_mesh(bem_resource)
    interface_map = interface_map_from_wire(only(interfaces)["topology"])
    sound_speed = Float64(bounded_region["sound_speed_m_per_s"])
    density = Float64(bounded_region["density_kg_per_m3"])
    excitation_port_ids = String.(request["excitation_port_ids"])
    isempty(excitation_port_ids) && error("Reference coupled solve requires at least one excitation port.")
    radiator_tags = Int[]
    for port_id in excitation_port_ids
        port = object_by_id(ports, port_id, "excitation port")
        String(port["kind"]) == "normal_velocity" || error(
            "Reference coupled backend currently supports only normal_velocity excitation ports.",
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
            "Each reference excitation component must own exactly one moving boundary in the bounded region.",
        )
        push!(radiator_tags, Int(only(bounded_boundaries)["group"]["tag"]))
    end

    solver_options = get(request, "solver_options", Dict{String,Any}())
    quadrature_order = Int(get(solver_options, "quadrature_order", 2))
    singular_order = Int(get(solver_options, "singular_order", 2))
    outputs = get(request, "outputs", Any[])
    for frequency_value in request["frequencies_hz"]
        frequency_hz = Float64(frequency_value)
        println(stderr, "Coupled reference: assembling $(frequency_hz) Hz")
        reference_system = build_coupled_reference_system(
            fem_mesh,
            bem_mesh,
            interface_map,
            frequency_hz,
            sound_speed,
            density;
            quadrature_order=quadrature_order,
            singular_order=singular_order,
        )
        solutions = [
            solve_coupled_reference_system(reference_system, radiator_tag)
            for radiator_tag in radiator_tags
        ]
        quantities = Dict{String,Any}[]
        for output in outputs
            quantity = String(output["quantity"])
            if quantity == "fem_nodal_pressure"
                push!(
                    quantities,
                    quantity_wire(
                        output,
                        rows([solution.fem_pressure for solution in solutions]),
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
                        rows([solution.bem_pressure for solution in solutions]),
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
                        rows([solution.interface_flux for solution in solutions]),
                        "Pa/m",
                        ["excitation", "interface_node"],
                        metadata=Dict("interface_id" => String(only(interfaces)["id"])),
                    ),
                )
            elseif quantity == "exterior_pressure"
                options = get(output, "options", Dict{String,Any}())
                raw_points = get(options, "points_m", Any[])
                isempty(raw_points) && error("exterior_pressure output requires options.points_m.")
                points = [SVector{3,Float64}(Float64.(point)) for point in raw_points]
                pressures = [
                    evaluate_galerkin_field_cpu(
                        points,
                        bem_mesh,
                        solution.bem_pressure,
                        solution.bem_neumann,
                        reference_system.wavenumber,
                        reference_system.field_cache,
                    )
                    for solution in solutions
                ]
                push!(
                    quantities,
                    quantity_wire(
                        output,
                        rows(pressures),
                        "Pa",
                        ["excitation", "observation"],
                        metadata=Dict("points_m" => raw_points),
                    ),
                )
            else
                error("Unsupported coupled reference output quantity: $quantity")
            end
        end

        result = Dict(
            "schema_version" => 1,
            "freq_hz" => frequency_hz,
            "excitation_port_ids" => excitation_port_ids,
            "quantities" => quantities,
            "diagnostics" => Dict(
                "precision" => "float64",
                "linear_solver" => "dense_lu",
                "relative_residual" => maximum(solution.relative_residual for solution in solutions),
                "pressure_continuity_error" => maximum(
                    solution.pressure_continuity_error for solution in solutions
                ),
                "flux_conservation_error" => maximum(
                    solution.flux_conservation_error for solution in solutions
                ),
                "all_bem_replay_error" => maximum(solution.all_bem_replay_error for solution in solutions),
            ),
        )
        println(JSON.json(result))
        flush(stdout)
    end
end

try
    request = JSON.parse(read(stdin, String))
    solve_request(request)
catch exception
    showerror(stderr, exception, catch_backtrace())
    println(stderr)
    exit(1)
end
