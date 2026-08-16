using AMDGPU

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

function inspect_rocm_regular_kernels(args=ARGS)
    output = isempty(args) ?
        joinpath(@__DIR__, "..", "results", "rocm_regular_pair_kernels.gcn") : args[1]
    mesh_path = joinpath(@__DIR__, "..", "test_meshes", "sample.msh")
    mesh = load_gmsh22_with_tags(mesh_path, Float32(0.001))
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, 1)
    cache = build_rocm_regular_assembly_cache(mesh, p1, dp0, rule; singular_order=2)
    operators = (
        single_layer=AMDGPU.zeros(ComplexF32, p1.global_dof_count, dp0.global_dof_count),
        double_layer=AMDGPU.zeros(ComplexF32, p1.global_dof_count, p1.global_dof_count),
        adjoint_double_layer=AMDGPU.zeros(ComplexF32, p1.global_dof_count, dp0.global_dof_count),
        hypersingular=AMDGPU.zeros(ComplexF32, p1.global_dof_count, p1.global_dof_count),
    )
    k = Float32(2pi * 500 / 343)
    test_start = Int32(cache.color_offsets[1])
    test_count = Int32(cache.color_offsets[2] - cache.color_offsets[1])
    trial_start = test_start
    trial_count = test_count

    mkpath(dirname(output))
    open(output, "w") do io
        redirect_stdout(io) do
            AMDGPU.@device_code_gcn begin
                AMDGPU.@roc launch=false BeatEngineCore._rocm_regular_slp_adjoint_pairs_kernel!(
                    operators.single_layer,
                    operators.adjoint_double_layer,
                    cache.face_vertices,
                    cache.normals,
                    cache.areas,
                    cache.faces,
                    cache.p1_dofs,
                    cache.element_dp0_dofs,
                    cache.rule_points,
                    cache.rule_weights,
                    cache.color_elements,
                    test_start,
                    test_count,
                    trial_start,
                    trial_count,
                    k,
                    cache.p1_dof_count,
                    cache.face_count,
                    cache.rule_count,
                    cache.vertex_offsets,
                    cache.incident_elements,
                    Int32(0),
                    one(k),
                    one(k),
                    one(k),
                )
                AMDGPU.@roc launch=false BeatEngineCore._rocm_regular_dlp_hyp_pairs_kernel!(
                    operators.double_layer,
                    operators.hypersingular,
                    cache.face_vertices,
                    cache.normals,
                    cache.areas,
                    cache.faces,
                    cache.curls,
                    cache.p1_dofs,
                    cache.rule_points,
                    cache.rule_weights,
                    cache.color_elements,
                    test_start,
                    test_count,
                    trial_start,
                    trial_count,
                    k,
                    cache.p1_dof_count,
                    cache.face_count,
                    cache.rule_count,
                    cache.vertex_offsets,
                    cache.incident_elements,
                    Int32(0),
                    one(k),
                    one(k),
                    one(k),
                    one(k),
                    one(k),
                    one(k),
                )
            end
            operator_mode = BeatEngineCore._normalized_rocm_pair_operator_mode()
            if operator_mode == :split
                AMDGPU.@device_code_gcn begin
                    AMDGPU.@roc launch=false BeatEngineCore._rocm_regular_dlp_pairs_kernel!(
                        operators.double_layer,
                        cache.face_vertices,
                        cache.normals,
                        cache.areas,
                        cache.faces,
                        cache.p1_dofs,
                        cache.rule_points,
                        cache.rule_weights,
                        cache.color_elements,
                        test_start,
                        test_count,
                        trial_start,
                        trial_count,
                        k,
                        cache.p1_dof_count,
                        cache.face_count,
                        cache.rule_count,
                        cache.vertex_offsets,
                        cache.incident_elements,
                        Int32(0),
                        one(k),
                        one(k),
                        one(k),
                    )
                    AMDGPU.@roc launch=false BeatEngineCore._rocm_regular_hyp_pairs_kernel!(
                        operators.hypersingular,
                        cache.face_vertices,
                        cache.normals,
                        cache.areas,
                        cache.faces,
                        cache.curls,
                        cache.p1_dofs,
                        cache.rule_points,
                        cache.rule_weights,
                        cache.color_elements,
                        test_start,
                        test_count,
                        trial_start,
                        trial_count,
                        k,
                        cache.p1_dof_count,
                        cache.face_count,
                        cache.rule_count,
                        cache.vertex_offsets,
                        cache.incident_elements,
                        Int32(0),
                        one(k),
                        one(k),
                        one(k),
                        one(k),
                        one(k),
                        one(k),
                    )
                end
            elseif operator_mode == :partial_fused
                AMDGPU.@device_code_gcn begin
                    AMDGPU.@roc launch=false BeatEngineCore._rocm_regular_slp_adjoint_dlp_pairs_kernel!(
                        operators.single_layer,
                        operators.adjoint_double_layer,
                        operators.double_layer,
                        cache.face_vertices,
                        cache.normals,
                        cache.areas,
                        cache.faces,
                        cache.p1_dofs,
                        cache.element_dp0_dofs,
                        cache.rule_points,
                        cache.rule_weights,
                        cache.color_elements,
                        test_start,
                        test_count,
                        trial_start,
                        trial_count,
                        k,
                        cache.p1_dof_count,
                        cache.face_count,
                        cache.rule_count,
                        cache.vertex_offsets,
                        cache.incident_elements,
                        Int32(0),
                        one(k),
                        one(k),
                        one(k),
                    )
                    AMDGPU.@roc launch=false BeatEngineCore._rocm_regular_hyp_pairs_kernel!(
                        operators.hypersingular,
                        cache.face_vertices,
                        cache.normals,
                        cache.areas,
                        cache.faces,
                        cache.curls,
                        cache.p1_dofs,
                        cache.rule_points,
                        cache.rule_weights,
                        cache.color_elements,
                        test_start,
                        test_count,
                        trial_start,
                        trial_count,
                        k,
                        cache.p1_dof_count,
                        cache.face_count,
                        cache.rule_count,
                        cache.vertex_offsets,
                        cache.incident_elements,
                        Int32(0),
                        one(k),
                        one(k),
                        one(k),
                        one(k),
                        one(k),
                        one(k),
                    )
                end
            end
        end
    end

    object_directory = splitext(output)[1] * "_objects"
    mkpath(object_directory)
    for kernel in values(AMDGPU.Compiler._kernel_instances)
        kernel_name = string(nameof(kernel.f))
        startswith(kernel_name, "_rocm_regular_") || continue
        function_type, argument_types = typeof(kernel).parameters
        config = AMDGPU.Compiler.compiler_config(AMDGPU.device())
        source = AMDGPU.Compiler.methodinstance(function_type, argument_types)
        job = AMDGPU.Compiler.CompilerJob(source, config)
        result = AMDGPU.Compiler.compile_or_lookup(job)
        write(joinpath(object_directory, kernel_name * ".co"), result.obj)
    end

    release_operator_storage!(operators)
    release_rocm_regular_assembly_cache!(cache)
    println("Wrote $(output)")
end

inspect_rocm_regular_kernels()
