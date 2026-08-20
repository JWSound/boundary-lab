using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

relative_error(actual, reference) = norm(actual - reference) / max(norm(reference), eps(real(eltype(reference))))

function normalized_level(values)
    reference_index = argmin(abs.(collect(-180.0f0:5.0f0:180.0f0)))
    return Float32.(20 .* log10.(abs.(values) ./ abs(values[reference_index])))
end

function integrated_radiator_pressure(mesh, pressure, driven_tag, symmetry_mode)
    total = zero(eltype(pressure))
    scale = eltype(pressure)(symmetry_reduction_factor(symmetry_mode))
    for element_index in eachindex(mesh.faces)
        mesh.physical_tags[element_index] == driven_tag || continue
        face = mesh.faces[element_index]
        average = (pressure[face[1]] + pressure[face[2]] + pressure[face[3]]) / eltype(pressure)(3)
        total += average * eltype(pressure)(mesh.areas[element_index]) * scale
    end
    return total
end

function diagnose_rocm_polar()
    @assert BeatEngineCore._is_corrupt_rocsolver_info(
        ArgumentError("invalid argument #1178117628 to LAPACK call"),
    )
    @assert !BeatEngineCore._is_corrupt_rocsolver_info(
        ArgumentError("invalid argument #3 to LAPACK call"),
    )
    mesh_path = get(ENV, "BLAB_DIAG_MESH", joinpath(@__DIR__, "..", "test_meshes", "sample_quarter.msh"))
    frequency_hz = parse(Float32, get(ENV, "BLAB_DIAG_FREQUENCY_HZ", "8000"))
    regular_order = parse(Int, get(ENV, "BLAB_DIAG_REGULAR_ORDER", "4"))
    singular_order = parse(Int, get(ENV, "BLAB_DIAG_SINGULAR_ORDER", "4"))
    driven_tag = parse(Int, get(ENV, "BLAB_DIAG_DRIVEN_TAG", "2"))
    symmetry_mode = Symbol(get(ENV, "BLAB_DIAG_SYMMETRY", "xy"))
    solve_repeats = parse(Int, get(ENV, "BLAB_DIAG_SOLVE_REPEATS", "1"))

    mesh = load_gmsh22_with_tags(mesh_path, Float32(0.001))
    validate_symmetry_fundamental_domain!(mesh, symmetry_mode)
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, regular_order)
    singular_cache = build_singular_correction_cache(mesh, singular_order)
    k = Float32(2pi) * frequency_hz / 343.0f0

    cpu_cache = build_beat_cpu_assembly_cache(
        mesh, p1, dp0, rule;
        singular_order=singular_order,
        symmetry_mode=symmetry_mode,
    )
    cpu_operators = assemble_regular_galerkin_operators(
        mesh, p1, dp0, k, rule;
        skip_singular=false,
        singular_order=singular_order,
        backend=:cpu,
        singular_cache=singular_cache,
        cpu_cache=cpu_cache,
        symmetry_mode=symmetry_mode,
    )
    rocm_cache = build_rocm_regular_assembly_cache(
        mesh, p1, dp0, rule;
        singular_order=singular_order,
        symmetry_mode=symmetry_mode,
    )
    rocm_operators = assemble_regular_galerkin_operators(
        mesh, p1, dp0, k, rule;
        skip_singular=false,
        singular_order=singular_order,
        backend=:rocm,
        device_cache=rocm_cache,
        singular_cache=singular_cache,
        rocm_assembly_mode=:native,
        symmetry_mode=symmetry_mode,
    )

    identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry_mode)
    identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry_mode)
    q_neumann = zeros(ComplexF32, length(mesh.faces))
    q_neumann[mesh.physical_tags .== driven_tag] .= ComplexF32(0, 1)
    cpu_pressure = solve_burton_miller_neumann(
        cpu_operators, identity_p1_p1, identity_p1_dp0, q_neumann, k,
    )
    identity_cache = build_rocm_burton_miller_identity_cache(identity_p1_p1, identity_p1_dp0, Float32)
    rocm_pressure = solve_burton_miller_neumann(rocm_operators, identity_cache, q_neumann, k)
    for repeat_index in 2:solve_repeats
        repeated_pressure = solve_burton_miller_neumann(rocm_operators, identity_cache, q_neumann, k)
        relative_error(repeated_pressure, rocm_pressure) <= 1.0f-5 ||
            error("ROCm solve changed at repeat $repeat_index.")
        repeat_index % 100 == 0 && println("solve_repeat=$repeat_index")
    end

    angles = collect(-180.0f0:5.0f0:180.0f0)
    radius = 2.0f0
    points = [(radius * sind(angle), 0.0f0, radius * cosd(angle)) for angle in angles]
    cpu_field_cache = build_field_evaluation_cache(mesh, rule; symmetry_mode=symmetry_mode)
    rocm_field_cache = build_rocm_field_evaluation_cache(cpu_field_cache)
    cpu_cpu_field = evaluate_galerkin_field_cpu(points, mesh, cpu_pressure, q_neumann, k, cpu_field_cache)
    rocm_cpu_field = evaluate_galerkin_field_cpu(points, mesh, rocm_pressure, q_neumann, k, cpu_field_cache)
    cpu_rocm_field = evaluate_galerkin_field_rocm(points, mesh, cpu_pressure, q_neumann, k, rocm_field_cache)
    rocm_rocm_field = evaluate_galerkin_field_rocm(points, mesh, rocm_pressure, q_neumann, k, rocm_field_cache)

    operator_error = maximum((
        relative_error(Array(rocm_operators.single_layer), cpu_operators.single_layer),
        relative_error(Array(rocm_operators.double_layer), cpu_operators.double_layer),
        relative_error(Array(rocm_operators.adjoint_double_layer), cpu_operators.adjoint_double_layer),
        relative_error(Array(rocm_operators.hypersingular), cpu_operators.hypersingular),
    ))
    levels = normalized_level(cpu_cpu_field)
    cpu_integrated_pressure = integrated_radiator_pressure(mesh, cpu_pressure, driven_tag, symmetry_mode)
    rocm_integrated_pressure = integrated_radiator_pressure(mesh, rocm_pressure, driven_tag, symmetry_mode)
    selected = Dict(angle => levels[findfirst(==(angle), angles)] for angle in (-180.0f0, -90.0f0, 0.0f0, 90.0f0, 180.0f0))
    println((
        mesh=mesh_path,
        frequency_hz=frequency_hz,
        solve_repeats=solve_repeats,
        faces=length(mesh.faces),
        operator_error=operator_error,
        pressure_error=relative_error(rocm_pressure, cpu_pressure),
        impedance_numerator_error=abs(rocm_integrated_pressure - cpu_integrated_pressure) /
            max(abs(cpu_integrated_pressure), eps(Float32)),
        rocm_boundary_cpu_field_error=relative_error(rocm_cpu_field, cpu_cpu_field),
        cpu_boundary_rocm_field_error=relative_error(cpu_rocm_field, cpu_cpu_field),
        rocm_end_to_end_field_error=relative_error(rocm_rocm_field, cpu_cpu_field),
        cpu_normalized_levels_db=selected,
        cpu_level_range_db=(minimum(levels), maximum(levels)),
    ))

    release_rocm_field_evaluation_cache!(rocm_field_cache)
    release_rocm_burton_miller_identity_cache!(identity_cache)
    release_operator_storage!(rocm_operators)
    release_rocm_regular_assembly_cache!(rocm_cache)
end

diagnose_rocm_polar()
