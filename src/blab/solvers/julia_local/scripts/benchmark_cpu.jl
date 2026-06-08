include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))

using Dates
using LinearAlgebra
using Printf
using Profile
using Statistics
using .BeatEngineCore

const GIT_COMMIT_CACHE = Ref{Any}(:unset)

Base.@kwdef mutable struct CpuBenchmarkConfig
    mesh::String = joinpath(@__DIR__, "..", "test_meshes", "sample.msh")
    frequency::Float64 = 1000.0
    precision_name::String = "Float32"
    quadrature_order::Int = 2
    singular_order::Int = 2
    eval_points::Int = 0
    subset_faces::Int = 128
    symmetry::String = "off"
    repetitions::Int = 1
    warmups::Int = 1
    blas_threads::Int = 1
    scale_factor::Float64 = 0.001
    sound_speed::Float64 = 343.0
    rho::Float64 = 1.21
    tag_throat::Int = 2
    distance::Float64 = 2.0
    skip_solve::Bool = false
    skip_field::Bool = false
    threaded_assembly::Bool = true
    profile::String = "none"
    output::String = joinpath(@__DIR__, "..", "results", "benchmark_cpu_sample.json")
    verbose::Bool = false
end

function print_usage()
    println("""
    Usage:
      julia scripts/benchmark_cpu.jl [options]

    Options:
      --mesh PATH                    Mesh path. Default: test_meshes/sample.msh
      --freq HZ                      Frequency in Hz. Default: 1000
      --precision Float32|Float64    Numeric precision. Default: Float32
      --quadrature-order N           Regular quadrature order. Default: 2
      --singular-order N             Singular quadrature order. Default: 2
      --eval-points N                CPU field evaluation point count. Default: 0
      --subset-faces N               Use first N faces. Default: 128. 0 means full mesh.
      --symmetry off|x|xy            Symmetry mode. Default: off
      --repetitions N                Measured repetitions. Default: 1
      --warmups N                    Warmup repetitions. Default: 1
      --blas-threads N               BLAS thread count. Default: 1.
      --scale FACTOR                 Mesh scale factor. Default: 0.001
      --tag-throat N                 Physical tag used for Neumann throat RHS. Default: 2
      --skip-solve                   Do not build/solve the Burton-Miller system.
      --skip-field                   Do not evaluate the radiated field.
      --serial-assembly              Disable colored threaded CPU operator assembly.
      --profile none|cpu|allocs      Print CPU or allocation profile for one measured run.
      --json PATH                    Write JSON results. Default: results/benchmark_cpu_sample.json
      --verbose                      Print every timing bucket in the console summary.
      --help                         Print this message.
    """)
end

function parse_args(args)
    config = CpuBenchmarkConfig()
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
        elseif arg == "--symmetry"
            i += 1; config.symmetry = lowercase(strip(args[i]))
        elseif arg == "--repetitions"
            i += 1; config.repetitions = parse(Int, args[i])
        elseif arg == "--warmups"
            i += 1; config.warmups = parse(Int, args[i])
        elseif arg == "--blas-threads"
            i += 1; config.blas_threads = parse(Int, args[i])
        elseif arg == "--scale"
            i += 1; config.scale_factor = parse(Float64, args[i])
        elseif arg == "--tag-throat"
            i += 1; config.tag_throat = parse(Int, args[i])
        elseif arg == "--skip-solve"
            config.skip_solve = true
        elseif arg == "--skip-field"
            config.skip_field = true
        elseif arg == "--serial-assembly"
            config.threaded_assembly = false
        elseif arg == "--profile"
            i += 1; config.profile = lowercase(args[i])
        elseif arg == "--json"
            i += 1; config.output = args[i]
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

function git_commit()
    GIT_COMMIT_CACHE[] !== :unset && return GIT_COMMIT_CACHE[]
    try
        repo_root = abspath(joinpath(@__DIR__, "..", "..", "..", "..", ".."))
        cmd = Cmd(`git -c safe.directory=$(repo_root) rev-parse --short HEAD`; dir=repo_root)
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

function throat_rhs(mesh, config::CpuBenchmarkConfig, ::Type{T}) where {T<:AbstractFloat}
    omega = T(2pi) * T(config.frequency)
    throat_indices = findall(t -> t == config.tag_throat, mesh.physical_tags)
    q_neumann = zeros(Complex{T}, length(mesh.faces))
    q_neumann[throat_indices] .= Complex{T}(0, T(config.rho) * omega)
    return q_neumann, throat_indices
end

function solve_cpu_timed!(timings, operators, identity_p1_p1, identity_p1_dp0, q_neumann, k::T) where {T<:AbstractFloat}
    lhs = nothing
    rhs = nothing
    timed_stage!(timings, "lhs_rhs_build") do
        coupling = Complex{T}(0, 1) / k
        identity_p1_p1_complex = Complex{T}.(identity_p1_p1)
        identity_p1_dp0_complex = Complex{T}.(identity_p1_dp0)
        q_neumann_complex = Complex{T}.(q_neumann)
        lhs = Complex{T}(0.5) .* identity_p1_p1_complex .- operators.double_layer .+ coupling .* operators.hypersingular
        rhs = (-operators.single_layer .- coupling .* (operators.adjoint_double_layer .+ Complex{T}(0.5) .* identity_p1_dp0_complex)) * q_neumann_complex
        nothing
    end
    pressure = timed_stage!(timings, "linear_solve") do
        Complex{T}.(lhs \ rhs)
    end
    timings["solve_total"] = timings["lhs_rhs_build"] + timings["linear_solve"]
    return pressure
end

function run_workload(config::CpuBenchmarkConfig; measured::Bool=true)
    T = precision_type(config.precision_name)
    BLAS.set_num_threads(config.blas_threads)
    timings = Dict{String,Float64}()

    mesh = timed_stage!(timings, "mesh_load") do
        load_gmsh22_with_tags(config.mesh, T(config.scale_factor))
    end
    symmetry_mode = Symbol(config.symmetry)
    validate_symmetry_fundamental_domain!(mesh, symmetry_mode)

    p1_space = nothing
    dp0_space = nothing
    timed_stage!(timings, "space_build") do
        p1_space = build_p1_space(mesh)
        dp0_space = build_dp0_space(mesh)
        nothing
    end

    rule = timed_stage!(timings, "quadrature_rule_build") do
        triangle_rule(T, config.quadrature_order)
    end
    element_count = config.subset_faces > 0 ? min(config.subset_faces, length(mesh.faces)) : length(mesh.faces)
    element_indices = 1:element_count
    subset_run = element_count != length(mesh.faces)
    if measured && subset_run && !config.skip_solve
        @warn "subset-faces creates an assembly microbenchmark, not a physically complete solve. Use --subset-faces 0 for full end-to-end timing."
    end

    singular_cache = timed_stage!(timings, "singular_correction_cache_build") do
        build_singular_correction_cache(mesh, config.singular_order, element_indices)
    end

    identity_p1_p1 = timed_stage!(timings, "identity_assembly_p1_p1") do
        assemble_l2_identity_matrix(mesh, p1_space, dp0_space, rule, :p1, :p1; symmetry_mode=symmetry_mode)
    end
    identity_p1_dp0 = timed_stage!(timings, "identity_assembly_p1_dp0") do
        assemble_l2_identity_matrix(mesh, p1_space, dp0_space, rule, :p1, :dp0; symmetry_mode=symmetry_mode)
    end

    field_cache = nothing
    if !config.skip_field && config.eval_points > 0
        field_cache = timed_stage!(timings, "field_cache_build_cpu") do
            build_field_evaluation_cache(mesh, rule; symmetry_mode=symmetry_mode)
        end
    else
        timings["field_cache_build_cpu"] = 0.0
    end

    k = T(2pi * config.frequency / config.sound_speed)
    operators = timed_stage!(timings, "operator_total_assembly") do
        assemble_regular_galerkin_operators(
            mesh,
            p1_space,
            dp0_space,
            k,
            rule;
            skip_singular=false,
            singular_order=config.singular_order,
            element_indices=element_indices,
            use_cuda_regular=false,
            threaded=config.threaded_assembly,
            timing=timings,
            singular_cache=singular_cache,
            symmetry_mode=symmetry_mode,
        )
    end
    timings["regular_operator_assembly"] = get(timings, "regular_operator_cpu_scatter", 0.0)
    timings["singular_corrections"] = get(timings, "singular_corrections_cpu_scatter", 0.0)
    timings["operator_allocation_overhead"] = max(
        timings["operator_total_assembly"] - timings["regular_operator_assembly"] - timings["singular_corrections"],
        0.0,
    )

    q_neumann, throat_indices = throat_rhs(mesh, config, T)
    pressure = nothing
    if config.skip_solve
        timings["lhs_rhs_build"] = 0.0
        timings["linear_solve"] = 0.0
        timings["solve_total"] = 0.0
    else
        pressure = solve_cpu_timed!(timings, operators, identity_p1_p1, identity_p1_dp0, q_neumann, k)
    end

    field_norm = nothing
    if config.skip_field || config.eval_points == 0 || pressure === nothing
        timings["field_evaluation"] = 0.0
    else
        field_norm = timed_stage!(timings, "field_evaluation") do
            eval_points = fibonacci_sphere(config.eval_points, T(config.distance))
            pot = evaluate_galerkin_field_cpu(eval_points, mesh, pressure, q_neumann, k, field_cache)
            Float64(norm(pot))
        end
    end

    total_stage_keys = [
        "mesh_load",
        "space_build",
        "quadrature_rule_build",
        "identity_assembly_p1_p1",
        "identity_assembly_p1_dp0",
        "field_cache_build_cpu",
        "singular_correction_cache_build",
        "operator_total_assembly",
        "solve_total",
        "field_evaluation",
    ]
    total = sum(get(timings, key, 0.0) for key in total_stage_keys)

    return Dict{String,Any}(
        "timestamp" => string(now()),
        "git_commit" => git_commit(),
        "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "mesh" => abspath(config.mesh),
        "mesh_faces" => length(mesh.faces),
        "mesh_vertices" => length(mesh.vertices),
        "p1_dofs" => p1_space.global_dof_count,
        "dp0_dofs" => dp0_space.global_dof_count,
        "element_count" => element_count,
        "subset_run" => subset_run,
        "frequency_hz" => config.frequency,
        "precision" => config.precision_name,
        "backend" => "cpu",
        "quadrature_order" => config.quadrature_order,
        "singular_order" => config.singular_order,
        "eval_points" => config.eval_points,
        "regular_pairs" => get(operators, :regular_pairs, nothing),
        "singular_pairs" => get(operators, :singular_pairs, nothing),
        "singular_cache_pairs" => singular_cache.pair_count,
        "skipped_pairs" => get(operators, :skipped_pairs, nothing),
        "throat_elements" => length(throat_indices),
        "regular_kernel_mode" => get(operators, :regular_kernel_mode, nothing),
        "cpu_color_count" => get(operators, :cpu_color_count, nothing),
        "regular_assembly_mode" => string(get(operators, :regular_assembly_mode, :cpu_serial)),
        "symmetry" => config.symmetry,
        "pressure_norm" => pressure === nothing ? nothing : Float64(norm(pressure)),
        "field_norm" => field_norm,
        "timings_seconds" => timings,
        "total_seconds" => total,
    )
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
    summary["total_seconds"] = Dict("min" => minimum(totals), "median" => median(totals), "max" => maximum(totals))
    return summary
end

json_escape(s::AbstractString) = replace(replace(replace(replace(s, "\\" => "\\\\"), "\"" => "\\\""), "\n" => "\\n"), "\r" => "\\r")

function json_value(io::IO, value)
    if value === nothing
        print(io, "null")
    elseif value isa Bool
        print(io, value ? "true" : "false")
    elseif value isa Number
        print(io, value isa AbstractFloat && !isfinite(value) ? "null" : value)
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
        "Benchmark: %s | %s | %.1f Hz | %d/%d faces | q%d/s%d | eval %d | BLAS %d",
        config["backend"],
        config["precision"],
        config["frequency_hz"],
        base["element_count"],
        base["mesh_faces"],
        config["quadrature_order"],
        config["singular_order"],
        config["eval_points"],
        base["blas_threads"],
    ))
    println(@sprintf(
        "Dofs: P1 %d | DP0 %d | colors %s | pairs regular %s singular %s skipped %s",
        base["p1_dofs"],
        base["dp0_dofs"],
        string(get(base, "cpu_color_count", nothing)),
        string(base["regular_pairs"]),
        string(base["singular_pairs"]),
        string(base["skipped_pairs"]),
    ))

    summary = payload["summary_seconds"]
    key_stages = [
        "total_seconds",
        "operator_total_assembly",
        "regular_operator_assembly",
        "singular_corrections",
        "operator_allocation_overhead",
        "lhs_rhs_build",
        "linear_solve",
        "solve_total",
        "field_evaluation",
    ]

    println("Stage medians:")
    for key in key_stages
        haskey(summary, key) || continue
        println(@sprintf("  %-32s %.6f s", key, summary[key]["median"]))
    end

    if get(config, "verbose", false)
        println("Detailed medians:")
        for key in sort(collect(keys(summary)))
            key in key_stages && continue
            println(@sprintf("  %-32s %.6f s", key, summary[key]["median"]))
        end
    end
end

function benchmark_payload(config::CpuBenchmarkConfig)
    T = precision_type(config.precision_name)
    BLAS.set_num_threads(config.blas_threads)
    normalized_config = Dict{String,Any}(
        "mesh" => abspath(config.mesh),
        "frequency_hz" => config.frequency,
        "precision" => string(T),
        "backend" => "cpu",
        "quadrature_order" => config.quadrature_order,
        "singular_order" => config.singular_order,
        "eval_points" => config.eval_points,
        "subset_faces" => config.subset_faces,
        "symmetry" => config.symmetry,
        "repetitions" => config.repetitions,
        "warmups" => config.warmups,
        "blas_threads" => config.blas_threads,
        "skip_solve" => config.skip_solve,
        "skip_field" => config.skip_field,
        "threaded_assembly" => config.threaded_assembly,
        "profile" => config.profile,
        "regular_assembly_mode" => config.threaded_assembly ? "cpu_colored_threads" : "cpu_serial",
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
