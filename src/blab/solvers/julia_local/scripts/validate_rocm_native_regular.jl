using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

relative_error(actual, reference) = norm(actual - reference) / max(norm(reference), eps(Float32))

function validate_rocm_native_regular()
    amdgpu = BeatEngineCore.AMDGPU_MODULE
    amdgpu === nothing && error("AMDGPU.jl did not load. Run with the julia_rocm project.")
    amdgpu.functional() || error("AMDGPU.functional() is false.")

    mesh_path = joinpath(@__DIR__, "..", "test_meshes", "sample.msh")
    mesh = load_gmsh22_with_tags(mesh_path, Float32(0.001))
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, 1)
    k = Float32(2pi * 500.0 / 343.0)
    singular_cache = build_singular_correction_cache(mesh, 2)

    cpu_cache = build_beat_cpu_assembly_cache(mesh, p1, dp0, rule; singular_order=2)
    cpu_operators = nothing
    cpu_elapsed = @elapsed begin
        cpu_operators = assemble_regular_galerkin_operators(
            mesh,
            p1,
            dp0,
            k,
            rule;
            skip_singular=true,
            backend=:cpu,
            singular_cache=singular_cache,
            cpu_cache=cpu_cache,
        )
    end

    rocm_cache = build_rocm_regular_assembly_cache(mesh, p1, dp0, rule; singular_order=2)
    rocm_operators = nothing
    timing = Dict{String,Float64}()
    rocm_elapsed = @elapsed begin
        rocm_operators = assemble_regular_galerkin_operators(
            mesh,
            p1,
            dp0,
            k,
            rule;
            skip_singular=true,
            backend=:rocm,
            device_cache=rocm_cache,
            singular_cache=singular_cache,
            timing=timing,
            rocm_assembly_mode=:native,
        )
    end

    errors = Dict(
        "single_layer" => relative_error(Array(rocm_operators.single_layer), cpu_operators.single_layer),
        "double_layer" => relative_error(Array(rocm_operators.double_layer), cpu_operators.double_layer),
        "adjoint_double_layer" => relative_error(
            Array(rocm_operators.adjoint_double_layer),
            cpu_operators.adjoint_double_layer,
        ),
        "hypersingular" => relative_error(Array(rocm_operators.hypersingular), cpu_operators.hypersingular),
    )

    println("fixture=$(mesh_path)")
    println("device=$(amdgpu.device())")
    println("faces=$(length(mesh.faces)) p1_dofs=$(p1.global_dof_count) dp0_dofs=$(dp0.global_dof_count)")
    println("cpu_regular_assembly_s=$(cpu_elapsed)")
    println("rocm_native_regular_assembly_s=$(rocm_elapsed)")
    println("rocm_timing=$(timing)")
    println("operator_relative_errors=$(errors)")
    flush(stdout)

    maximum(values(errors)) <= 5.0f-5 || error("Native ROCm regular operators differ from BEAT CPU.")
    expected_mode = BeatEngineCore._normalized_rocm_regular_kernel_mode() == :pair_owned ?
        :rocm_native_colored_pair_owned : :rocm_native_entry_owned
    rocm_operators.regular_assembly_mode == expected_mode ||
        error("Unexpected ROCm regular assembly mode.")
    release_operator_storage!(rocm_operators)
    release_rocm_regular_assembly_cache!(rocm_cache)
    println("ROCM_NATIVE_REGULAR_VALIDATION_OK")
end

validate_rocm_native_regular()
