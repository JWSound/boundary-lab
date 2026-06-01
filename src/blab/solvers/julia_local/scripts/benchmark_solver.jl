include(joinpath(@__DIR__, "..", "src", "JBEMCore.jl"))

using Dates
using InteractiveUtils
using LinearAlgebra
using Printf
using Profile
using Statistics
using .JBEMCore

LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())

const CUDA_MODULE = try
    @eval import CUDA
    CUDA
catch
    nothing
end

const GIT_COMMIT_CACHE = Ref{Any}(:unset)

Base.@kwdef mutable struct BenchmarkConfig
    mesh::String = joinpath(@__DIR__, "..", "test_meshes", "sample.msh")
    frequency::Float64 = 1000.0
    precision_name::String = "Float32"
    quadrature_order::Int = 4
    singular_order::Int = 4
    eval_points::Int = 0
    subset_faces::Int = 0
    repetitions::Int = 1
    warmups::Int = 1
    scale_factor::Float64 = 0.001
    sound_speed::Float64 = 343.0
    rho::Float64 = 1.21
    tag_throat::Int = 2
    distance::Float64 = 2.0
    skip_solve::Bool = false
    skip_field::Bool = false
    profile::String = "none"
    output::String = joinpath(@__DIR__, "..", "results", "benchmark_sample.json")
    cuda_parallel_quadrature::Bool = true
    return_gpu::Bool = true
    profile_regular_kernel::Bool = false
    regular_probe_pair_limit::Int = 1_000_000
    regular_assembly_mode::String = "fused"
    verbose::Bool = false
end

function print_usage()
    println("""
    Usage:
      julia scripts/benchmark_solver.jl [options]

    Options:
      --mesh PATH                    Mesh path. Default: test_meshes/sample.msh
      --freq HZ                      Frequency in Hz. Default: 1000
      --precision Float32|Float64    Numeric precision. Default: Float32
      --quadrature-order N           Regular quadrature order. Default: 4
      --singular-order N             Singular quadrature order. Default: 4
      --eval-points N                Field evaluation point count. Default: 0
      --subset-faces N               Use first N faces for assembly experiments. 0 means full mesh.
      --repetitions N                Measured repetitions. Default: 1
      --warmups N                    Warmup repetitions. Default: 1
      --skip-solve                   Do not build/solve the Burton-Miller system.
      --skip-field                   Do not evaluate the radiated field.
      --profile none|cpu|allocs      Print CPU or allocation profile for one measured run.
      --json PATH                    Write JSON results. Default: results/benchmark_sample.json
      --serial-cuda-quadrature       Use the serial-pair CUDA regular assembly kernel.
      --cpu-transfer-cuda-operators  Return CUDA assembled operators to CPU memory before GPU solving.
      --profile-regular-kernel       Run extra CUDA regular-kernel probe launches for diagnostics.
      --regular-probe-pair-limit N   Max element pairs for lightweight probe kernels. 0 means full mesh.
      --regular-assembly-mode MODE   CUDA regular assembly mode: fused or split_atomic. Default: fused.
      --verbose                      Print every timing bucket in the console summary.
      --help                         Print this message.
    """)
end

function parse_args(args)
    config = BenchmarkConfig()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            print_usage()
            exit()
        elseif arg == "--mesh"
            i += 1; config.mesh = args[i]
        elseif arg == "--freq"
            i += 1; config.frequency = parse(Float64, args[i])
        elseif arg == "--precision"
            i += 1; config.precision_name = args[i]
        elseif arg == "--quadrature-order"
            i += 1; config.quadrature_order = parse(Int, args[i])
        elseif arg == "--singular-order"
            i += 1; config.singular_order = parse(Int, args[i])
        elseif arg == "--eval-points"
            i += 1; config.eval_points = parse(Int, args[i])
        elseif arg == "--subset-faces"
            i += 1; config.subset_faces = parse(Int, args[i])
        elseif arg == "--repetitions"
            i += 1; config.repetitions = parse(Int, args[i])
        elseif arg == "--warmups"
            i += 1; config.warmups = parse(Int, args[i])
        elseif arg == "--scale"
            i += 1; config.scale_factor = parse(Float64, args[i])
        elseif arg == "--tag-throat"
            i += 1; config.tag_throat = parse(Int, args[i])
        elseif arg == "--skip-solve"
            config.skip_solve = true
        elseif arg == "--skip-field"
            config.skip_field = true
        elseif arg == "--profile"
            i += 1; config.profile = lowercase(args[i])
        elseif arg == "--json"
            i += 1; config.output = args[i]
        elseif arg == "--serial-cuda-quadrature"
            config.cuda_parallel_quadrature = false
        elseif arg == "--cpu-transfer-cuda-operators"
            config.return_gpu = false
        elseif arg == "--profile-regular-kernel"
            config.profile_regular_kernel = true
        elseif arg == "--regular-probe-pair-limit"
            i += 1; config.regular_probe_pair_limit = parse(Int, args[i])
        elseif arg == "--regular-assembly-mode"
            i += 1; config.regular_assembly_mode = lowercase(args[i])
        elseif arg == "--verbose"
            config.verbose = true
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end
    return config
end

function precision_type(name::String)
    name == "Float32" && return Float32
    name == "Float64" && return Float64
    error("Unsupported precision: $name")
end

function cuda_available()
    CUDA_MODULE === nothing && return false
    try
        return CUDA_MODULE.functional()
    catch
        return false
    end
end

function maybe_cuda_device()
    cuda_available() || return nothing
    try
        return string(CUDA_MODULE.device())
    catch
        return "available"
    end
end

function git_commit()
    GIT_COMMIT_CACHE[] !== :unset && return GIT_COMMIT_CACHE[]
    try
        cmd = Cmd(`git -c safe.directory=$(abspath(joinpath(@__DIR__, ".."))) rev-parse --short HEAD`; dir=joinpath(@__DIR__, ".."))
        commit = strip(read(cmd, String))
        GIT_COMMIT_CACHE[] = isempty(commit) ? nothing : commit
    catch
        GIT_COMMIT_CACHE[] = nothing
    end
    return GIT_COMMIT_CACHE[]
end

function timed_stage!(timings::Dict{String,Float64}, name::String, thunk)
    value = nothing
    elapsed = @elapsed value = thunk()
    timings[name] = elapsed
    return value
end

timed_stage!(thunk, timings::Dict{String,Float64}, name::String) = timed_stage!(timings, name, thunk)

function throat_rhs(mesh, config::BenchmarkConfig, ::Type{T}) where {T<:AbstractFloat}
    ComplexType = Complex{T}
    omega = T(2pi) * T(config.frequency)
    throat_indices = findall(t -> t == config.tag_throat, mesh.physical_tags)
    q_neumann = zeros(ComplexType, length(mesh.faces))
    q_neumann[throat_indices] .= ComplexType(im * T(config.rho)) * omega
    return q_neumann, throat_indices
end

function add_singular_corrections_cpu!(timings, operators, mesh, p1_space, dp0_space, k::T, singular_order::Int, element_indices, singular_cache) where {T<:AbstractFloat}
    timings["singular_correction_alloc"] = 0.0
    singular_pairs = timed_stage!(timings, "singular_correction_compute_scatter") do
        JBEMCore.assemble_singular_galerkin_corrections!(
            operators.single_layer,
            operators.double_layer,
            operators.adjoint_double_layer,
            operators.hypersingular,
            mesh,
            p1_space,
            dp0_space,
            k,
            singular_order,
            element_indices,
            singular_cache,
        )
    end
    timings["singular_correction_transfer_to_gpu"] = 0.0
    timings["singular_correction_gpu_add"] = 0.0
    return singular_pairs
end

function add_singular_corrections_gpu!(timings, operators, mesh, p1_space, dp0_space, k::T, singular_order::Int, element_indices, singular_cache, cuda_singular_cache, cuda_regular_cache) where {T<:AbstractFloat}
    singular_pairs = timed_stage!(timings, "singular_correction_compute_scatter") do
        JBEMCore.add_singular_corrections_cuda_compact!(
            operators,
            mesh,
            p1_space,
            dp0_space,
            k,
            singular_order,
            element_indices,
            singular_cache,
            cuda_singular_cache=cuda_singular_cache,
            cuda_regular_cache=cuda_regular_cache,
            timing=timings,
        )
    end
    timings["singular_correction_alloc"] = get(timings, "singular_correction_gpu_alloc", 0.0)
    timings["singular_correction_transfer_to_gpu"] = get(timings, "singular_correction_compact_transfer", 0.0)
    return singular_pairs
end

function assemble_operators_timed!(timings, mesh, p1_space, dp0_space, k, rule, config::BenchmarkConfig, element_indices, cache, singular_cache, cuda_singular_cache)
    operators = nothing
    operators = timed_stage!(timings, "regular_operator_assembly") do
        assemble_regular_galerkin_operators(
            mesh,
            p1_space,
            dp0_space,
            k,
            rule;
            skip_singular=true,
            singular_order=config.singular_order,
            element_indices=element_indices,
            use_cuda_regular=true,
            cuda_cache=cache,
            return_gpu=config.return_gpu,
            parallel_quadrature=config.cuda_parallel_quadrature,
            timing=timings,
            singular_cache=singular_cache,
            cuda_singular_cache=cuda_singular_cache,
            profile_regular_kernel=config.profile_regular_kernel,
            regular_probe_pair_limit=config.regular_probe_pair_limit,
            regular_assembly_mode=Symbol(config.regular_assembly_mode),
        )
    end
    regular_probe_keys = [
        "regular_operator_thread_sweep_32",
        "regular_operator_thread_sweep_64",
        "regular_operator_thread_sweep_128",
        "regular_operator_probe_green_kernel",
        "regular_operator_probe_all_terms_kernel",
        "regular_operator_probe_slp_kernel",
        "regular_operator_probe_adjoint_kernel",
        "regular_operator_probe_dlp_kernel",
        "regular_operator_probe_hypersingular_kernel",
        "regular_operator_probe_slp_adjoint_colored_kernel",
        "regular_operator_probe_split_slp_adjoint_atomic_kernel",
        "regular_operator_probe_split_dlp_hyp_atomic_kernel",
    ]
    regular_probe_total = sum(get(timings, key, 0.0) for key in regular_probe_keys)
    timings["regular_operator_profile_probe_total"] = regular_probe_total
    timings["regular_operator_assembly_without_probes"] = timings["regular_operator_assembly"] - regular_probe_total

    singular_pairs = timed_stage!(timings, "singular_corrections") do
        if get(operators, :on_gpu, false)
            add_singular_corrections_gpu!(timings, operators, mesh, p1_space, dp0_space, k, config.singular_order, element_indices, singular_cache, cuda_singular_cache, cache)
        else
            add_singular_corrections_cpu!(timings, operators, mesh, p1_space, dp0_space, k, config.singular_order, element_indices, singular_cache)
        end
    end

    timings["operator_total_assembly"] = timings["regular_operator_assembly"] + timings["singular_corrections"]
    timings["operator_total_assembly_without_probes"] = timings["regular_operator_assembly_without_probes"] + timings["singular_corrections"]
    skipped_pairs = max(get(operators, :skipped_pairs, 0) - singular_pairs, 0)
    return merge(operators, (singular_pairs=singular_pairs, skipped_pairs=skipped_pairs))
end

function run_workload(config::BenchmarkConfig; measured::Bool=true)
    T = precision_type(config.precision_name)
    timings = Dict{String,Float64}()
    if !cuda_available()
        error("CUDA backend requested, but CUDA is not functional.")
    end

    mesh = timed_stage!(timings, "mesh_load") do
        load_gmsh22_with_tags(config.mesh, T(config.scale_factor))
    end
    p1_space = nothing
    dp0_space = nothing
    timed_stage!(timings, "space_build") do
        p1_space = build_p1_space(mesh)
        dp0_space = build_dp0_space(mesh)
        nothing
    end

    rule = triangle_rule(T, config.quadrature_order)
    cpu_field_cache = timed_stage!(timings, "field_cache_build_cpu") do
        build_field_evaluation_cache(mesh, rule)
    end
    field_cache = timed_stage!(timings, "field_cache_build_gpu") do
        build_cuda_field_evaluation_cache(cpu_field_cache)
    end
    element_count = config.subset_faces > 0 ? min(config.subset_faces, length(mesh.faces)) : length(mesh.faces)
    element_indices = 1:element_count
    subset_run = element_count != length(mesh.faces)
    if subset_run && !config.skip_solve
        @warn "subset-faces creates an assembly microbenchmark, not a physically complete solve. Use --skip-solve for pure assembly timing."
    end
    singular_cache = timed_stage!(timings, "singular_correction_cache_build") do
        build_singular_correction_cache(mesh, config.singular_order, element_indices)
    end
    cuda_singular_cache = timed_stage!(timings, "singular_correction_cuda_cache_build_request") do
        JBEMCore.build_cuda_singular_correction_cache(singular_cache, p1_space, dp0_space)
    end

    identity_p1_p1 = timed_stage!(timings, "identity_assembly_p1_p1") do
        assemble_l2_identity_matrix(mesh, p1_space, dp0_space, rule, :p1, :p1)
    end
    identity_p1_dp0 = timed_stage!(timings, "identity_assembly_p1_dp0") do
        assemble_l2_identity_matrix(mesh, p1_space, dp0_space, rule, :p1, :dp0)
    end

    cache = timed_stage!(timings, "cuda_cache_build") do
        build_cuda_regular_assembly_cache(mesh, rule; element_indices=element_indices)
    end

    k = T(2pi * config.frequency / config.sound_speed)
    operators = assemble_operators_timed!(timings, mesh, p1_space, dp0_space, k, rule, config, element_indices, cache, singular_cache, cuda_singular_cache)

    q_neumann, throat_indices = throat_rhs(mesh, config, T)
    pressure = nothing
    if config.skip_solve
        timings["lhs_rhs_build"] = 0.0
        timings["linear_solve"] = 0.0
        timings["solve_total"] = 0.0
    else
        pressure = timed_stage!(timings, "solve_total") do
            solve_burton_miller_neumann(operators, identity_p1_p1, identity_p1_dp0, q_neumann, k)
        end
        timings["lhs_rhs_build"] = 0.0
        timings["linear_solve"] = timings["solve_total"]
    end

    field_norm = nothing
    if config.skip_field || config.eval_points == 0 || pressure === nothing
        timings["field_evaluation"] = 0.0
    else
        field_norm = timed_stage!(timings, "field_evaluation") do
            eval_points = fibonacci_sphere(config.eval_points, T(config.distance))
            pot = evaluate_galerkin_field_cuda(eval_points, mesh, pressure, q_neumann, k, field_cache)
            Float64(norm(pot))
        end
    end

    total_stage_keys = [
        "mesh_load",
        "space_build",
        "identity_assembly_p1_p1",
        "identity_assembly_p1_dp0",
        "cuda_cache_build",
        "field_cache_build_cpu",
        "field_cache_build_gpu",
        "singular_correction_cache_build",
        "singular_correction_cuda_cache_build_request",
        "regular_operator_assembly",
        "singular_corrections",
        "solve_total",
        "field_evaluation",
    ]
    total = sum(get(timings, key, 0.0) for key in total_stage_keys)
    release_operator_storage!(operators)
    metadata = Dict{String,Any}(
        "timestamp" => string(now()),
        "git_commit" => git_commit(),
        "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "cuda_available" => cuda_available(),
        "cuda_device" => maybe_cuda_device(),
        "mesh" => abspath(config.mesh),
        "mesh_faces" => length(mesh.faces),
        "mesh_vertices" => length(mesh.vertices),
        "p1_dofs" => p1_space.global_dof_count,
        "dp0_dofs" => dp0_space.global_dof_count,
        "element_count" => element_count,
        "subset_run" => subset_run,
        "frequency_hz" => config.frequency,
        "precision" => config.precision_name,
        "backend" => "cuda",
        "quadrature_order" => config.quadrature_order,
        "singular_order" => config.singular_order,
        "eval_points" => config.eval_points,
        "regular_pairs" => get(operators, :regular_pairs, nothing),
        "singular_pairs" => get(operators, :singular_pairs, nothing),
        "singular_cache_pairs" => singular_cache.pair_count,
        "skipped_pairs" => get(operators, :skipped_pairs, nothing),
        "throat_elements" => length(throat_indices),
        "return_gpu" => config.return_gpu,
        "cuda_parallel_quadrature" => config.cuda_parallel_quadrature,
        "regular_kernel_threads" => get(operators, :regular_kernel_threads, nothing),
        "regular_kernel_blocks" => get(operators, :regular_kernel_blocks, nothing),
        "regular_kernel_shared_memory_bytes" => get(operators, :regular_kernel_shared_memory_bytes, nothing),
        "regular_kernel_qpair_count" => get(operators, :regular_kernel_qpair_count, nothing),
        "regular_kernel_total_pairs" => get(operators, :regular_kernel_total_pairs, nothing),
        "regular_probe_pair_count" => get(operators, :regular_probe_pair_count, nothing),
        "regular_kernel_mode" => get(operators, :regular_kernel_mode, nothing),
        "regular_assembly_mode" => string(get(operators, :regular_assembly_mode, config.regular_assembly_mode)),
        "regular_color_count" => get(operators, :regular_color_count, nothing),
        "profile_regular_kernel" => config.profile_regular_kernel,
        "regular_probe_pair_limit" => config.regular_probe_pair_limit,
        "pressure_norm" => pressure === nothing ? nothing : Float64(norm(pressure)),
        "field_norm" => field_norm,
        "timings_seconds" => timings,
        "total_seconds" => total,
    )
    return metadata
end

function summarize_runs(runs)
    keys_seen = sort(collect(keys(runs[1]["timings_seconds"])))
    summary = Dict{String,Any}()
    for key in keys_seen
        values_for_key = [run["timings_seconds"][key] for run in runs]
        summary[key] = Dict(
            "min" => minimum(values_for_key),
            "median" => median(values_for_key),
            "max" => maximum(values_for_key),
        )
    end
    totals = [run["total_seconds"] for run in runs]
    summary["total_seconds"] = Dict(
        "min" => minimum(totals),
        "median" => median(totals),
        "max" => maximum(totals),
    )
    return summary
end

json_escape(s::AbstractString) = replace(replace(replace(replace(s, "\\" => "\\\\"), "\"" => "\\\""), "\n" => "\\n"), "\r" => "\\r")

function json_value(io::IO, value)
    if value === nothing
        print(io, "null")
    elseif value isa Bool
        print(io, value ? "true" : "false")
    elseif value isa Number
        if value isa AbstractFloat && !isfinite(value)
            print(io, "null")
        else
            print(io, value)
        end
    elseif value isa AbstractString
        print(io, "\"", json_escape(value), "\"")
    elseif value isa Dict
        print(io, "{")
        first_item = true
        for key in sort(collect(keys(value)); by=string)
            first_item || print(io, ",")
            first_item = false
            print(io, "\"", json_escape(string(key)), "\":")
            json_value(io, value[key])
        end
        print(io, "}")
    elseif value isa AbstractVector || value isa Tuple
        print(io, "[")
        for (i, item) in enumerate(value)
            i > 1 && print(io, ",")
            json_value(io, item)
        end
        print(io, "]")
    else
        print(io, "\"", json_escape(string(value)), "\"")
    end
end

function write_json(path::String, payload)
    mkpath(dirname(path))
    open(path, "w") do io
        json_value(io, payload)
        println(io)
    end
end

function print_summary(payload)
    config = payload["config"]
    base = payload["runs"][1]
    println(@sprintf(
        "Benchmark: %s | %s | %.1f Hz | %d/%d faces | q%d/s%d | eval %d",
        config["backend"],
        config["precision"],
        config["frequency_hz"],
        base["element_count"],
        base["mesh_faces"],
        config["quadrature_order"],
        config["singular_order"],
        config["eval_points"],
    ))
    println(@sprintf(
        "Dofs: P1 %d | DP0 %d | pairs regular %s singular %s skipped %s",
        base["p1_dofs"],
        base["dp0_dofs"],
        string(base["regular_pairs"]),
        string(base["singular_pairs"]),
        string(base["skipped_pairs"]),
    ))

    summary = payload["summary_seconds"]
    key_stages = [
        "total_seconds",
        "operator_total_assembly",
        "operator_total_assembly_without_probes",
        "regular_operator_assembly",
        "regular_operator_assembly_without_probes",
        "regular_operator_kernel",
        "regular_operator_profile_probe_total",
        "regular_operator_probe_split_atomic_total",
        "singular_corrections",
        "singular_correction_compute_scatter",
        "solve_total",
        "field_evaluation",
    ]

    println("Stage medians:")
    for key in key_stages
        haskey(summary, key) || continue
        println(@sprintf("  %-36s %.6f s", key, summary[key]["median"]))
    end

    if get(config, "verbose", false)
        println("Detailed medians:")
        for key in sort(collect(keys(summary)))
            key in key_stages && continue
            println(@sprintf("  %-36s %.6f s", key, summary[key]["median"]))
        end
    end
end

function benchmark_payload(config::BenchmarkConfig)
    T = precision_type(config.precision_name)
    normalized_config = Dict{String,Any}(
        "mesh" => abspath(config.mesh),
        "frequency_hz" => config.frequency,
        "precision" => string(T),
        "backend" => "cuda",
        "quadrature_order" => config.quadrature_order,
        "singular_order" => config.singular_order,
        "eval_points" => config.eval_points,
        "subset_faces" => config.subset_faces,
        "repetitions" => config.repetitions,
        "warmups" => config.warmups,
        "skip_solve" => config.skip_solve,
        "skip_field" => config.skip_field,
        "profile" => config.profile,
        "profile_regular_kernel" => config.profile_regular_kernel,
        "regular_probe_pair_limit" => config.regular_probe_pair_limit,
        "regular_assembly_mode" => config.regular_assembly_mode,
        "verbose" => config.verbose,
    )

    for i in 1:config.warmups
        println(@sprintf("Warmup %d/%d", i, config.warmups))
        run_workload(config; measured=false)
        GC.gc()
    end

    runs = Dict{String,Any}[]
    for i in 1:config.repetitions
        println(@sprintf("Measured run %d/%d", i, config.repetitions))
        if config.profile == "cpu" && i == 1
            Profile.clear()
            result = Profile.@profile run_workload(config)
            push!(runs, result)
            Profile.print(format=:flat, sortedby=:count, maxdepth=20)
        elseif config.profile == "allocs" && i == 1
            bytes = @allocated result = run_workload(config)
            result["allocated_bytes_outer"] = bytes
            push!(runs, result)
            println(@sprintf("Outer allocated bytes: %d", bytes))
        else
            push!(runs, run_workload(config))
        end
        GC.gc()
    end

    return Dict{String,Any}(
        "config" => normalized_config,
        "runs" => runs,
        "summary_seconds" => summarize_runs(runs),
    )
end

function main(args=ARGS)
    config = parse_args(args)
    payload = benchmark_payload(config)
    write_json(config.output, payload)
    print_summary(payload)
    println("Wrote $(config.output)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
