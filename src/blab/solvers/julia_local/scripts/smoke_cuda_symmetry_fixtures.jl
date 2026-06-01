include(joinpath(@__DIR__, "..", "src", "JBEMCore.jl"))

using .JBEMCore
using LinearAlgebra

const CUDA_MODULE = try
    @eval import CUDA
    CUDA
catch
    nothing
end

if CUDA_MODULE === nothing || !CUDA_MODULE.functional()
    println("CUDA unavailable; skipping CUDA symmetry fixture smoke test.")
    exit()
end

function fixture_check(name::String, mode::Symbol, subset)
    mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", name), Float32(0.001))
    validate_symmetry_fundamental_domain!(mesh, mode)
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, 2)
    k = Float32(2pi * 1000.0 / 343.0)

    off = assemble_regular_galerkin_operators(
        mesh,
        p1,
        dp0,
        k,
        rule;
        skip_singular=true,
        element_indices=subset,
        use_cuda_regular=true,
        symmetry_mode=:off,
    )
    symmetric = assemble_regular_galerkin_operators(
        mesh,
        p1,
        dp0,
        k,
        rule;
        skip_singular=true,
        element_indices=subset,
        use_cuda_regular=true,
        symmetry_mode=mode,
    )

    image_count = symmetry_reduction_factor(mode) - 1
    expected_regular_pairs = off.regular_pairs + image_count * length(subset) * length(subset)
    @assert symmetric.regular_pairs == expected_regular_pairs
    delta = norm(symmetric.single_layer - off.single_layer)
    @assert delta > 0

    return (
        name=name,
        mode=mode,
        subset_faces=length(subset),
        vertices=length(mesh.vertices),
        faces=length(mesh.faces),
        off_regular_pairs=off.regular_pairs,
        symmetric_regular_pairs=symmetric.regular_pairs,
        image_single_layer_delta=delta,
    )
end

println(fixture_check("sample_half.msh", :x, 1:24))
println(fixture_check("sample_quarter.msh", :xy, 1:24))
