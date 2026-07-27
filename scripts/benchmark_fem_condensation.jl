using CUDA, CUDSS, LinearAlgebra, SparseArrays

include(
    joinpath(
        @__DIR__,
        "..",
        "src",
        "blab",
        "solvers",
        "julia_local",
        "src",
        "BeatEngineCore.jl",
    ),
)
using .BeatEngineCore
include(
    joinpath(
        @__DIR__,
        "..",
        "src",
        "blab",
        "solvers",
        "julia_local",
        "src",
        "BeatEngineCoupled.jl",
    ),
)
using .BeatEngineCoupled

function timed(action, label)
    CUDA.synchronize()
    started = time_ns()
    value = action()
    CUDA.synchronize()
    elapsed = (time_ns() - started) / 1.0e9
    println("$label=$(round(elapsed; digits=6))s")
    flush(stdout)
    return value
end

fixture = joinpath(
    @__DIR__,
    "..",
    "tests",
    "fixtures",
    "SAWMOD",
    "SMfemvolume_reduced_xy_detailed.msh",
)
frequency_hz = Float32(length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 20_000)
mesh = timed("mesh_load") do
    load_gmsh41_volume(fixture, Float32(0.001))
end
stiffness, mass = timed("fem_assembly") do
    assemble_p1_fem_matrices(mesh)
end
interface_tag = physical_tag(mesh, 2, "INTERFACE")
interface_faces = findall(==(interface_tag), mesh.boundary_physical_tags)
interface_nodes = sort(unique(vcat([collect(mesh.boundary_faces[index]) for index in interface_faces]...)))
interface_set = Set(interface_nodes)
interior_nodes = [index for index in eachindex(mesh.vertices) if index ∉ interface_set]
wavenumber = Float32(2pi) * frequency_hz / Float32(343)
fem_system = stiffness - wavenumber^2 .* mass
interior_system = fem_system[interior_nodes, interior_nodes]
interior_interface = fem_system[interior_nodes, interface_nodes]
println(
    "interior=$(length(interior_nodes)) interface=$(length(interface_nodes)) " *
    "nnz_interior=$(nnz(interior_system)) nnz_coupling=$(nnz(interior_interface))",
)
flush(stdout)

permutation = vcat(interior_nodes, interface_nodes)
permuted_system = ComplexF32.(fem_system[permutation, permutation])
device_system = timed("system_upload") do
    CUDA.CUSPARSE.CuSparseMatrixCSR(permuted_system)
end
system_size = length(permutation)
interior_count = length(interior_nodes)
interface_count = length(interface_nodes)
device_rhs = CUDA.zeros(ComplexF32, system_size)
device_solution = similar(device_rhs)
factorization = timed("cudss_analysis") do
    result = CUDSS.CudssSolver(device_system, "G", 'F')
    CUDSS.cudss_set(result, "schur_mode", 1)
    CUDSS.cudss_set(
        result,
        "user_schur_indices",
        vcat(zeros(Cint, interior_count), ones(Cint, interface_count)),
    )
    CUDSS.cudss("analysis", result, device_solution, device_rhs; asynchronous=false)
    result
end
timed("cudss_partial_factorization") do
    CUDSS.cudss(
        "factorization",
        factorization,
        device_solution,
        device_rhs;
        asynchronous=false,
    )
end
schur_shape = CUDSS.cudss_get(factorization, "schur_shape")
device_schur = CUDA.zeros(ComplexF32, schur_shape[1], schur_shape[2])
timed("cudss_schur_extract") do
    cudss_schur = CUDSS.CudssMatrix(device_schur)
    CUDSS.cudss_set(factorization, "schur_matrix", cudss_schur.matrix)
    CUDSS.cudss_get(factorization, "schur_matrix")
end

target = CUDA.ones(ComplexF32, system_size)
mul!(device_rhs, device_system, target)
timed("cudss_forward_schur") do
    CUDSS.cudss(
        "solve_fwd_schur",
        factorization,
        device_solution,
        device_rhs;
        asynchronous=false,
    )
end
condensed_rhs = copy(view(device_solution, (interior_count + 1):system_size))
condensed_factor = timed("schur_dense_lu") do
    lu!(device_schur)
end
condensed_solution = timed("schur_dense_solve") do
    condensed_factor \ condensed_rhs
end
view(device_solution, (interior_count + 1):system_size) .= condensed_solution
timed("cudss_backward_schur") do
    CUDSS.cudss(
        "solve_bwd_schur",
        factorization,
        device_rhs,
        device_solution;
        asynchronous=false,
    )
end
recovered_error = norm(Array(device_rhs[1:interior_count]) .- 1) / sqrt(interior_count)
println(
    "schur_mib=$(round(sizeof(device_schur) / 2.0^20; digits=2)) " *
    "recovered_rms_error=$recovered_error",
)
