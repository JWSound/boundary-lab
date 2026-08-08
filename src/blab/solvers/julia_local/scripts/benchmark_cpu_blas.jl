using Dates
using JSON
using LinearAlgebra
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

Base.@kwdef mutable struct CpuBlasBenchmarkConfig
    sizes::Vector{Int} = [441, 1390, 3502]
    mesh::String = ""
    frequency::Float64 = 1000.0
    quadrature_order::Int = 2
    singular_order::Int = 4
    symmetry::String = "off"
    scale_factor::Float64 = 0.001
    thread_counts::Vector{Int} = [1, 2, 4, 8, 16]
    warmups::Int = 2
    repetitions::Int = 5
    output::String = joinpath(@__DIR__, "..", "results", "benchmark_cpu_blas.json")
end

function parse_int_list(value::String, option::String)
    values = [parse(Int, strip(item)) for item in split(value, ",") if !isempty(strip(item))]
    isempty(values) && error("$option must contain at least one integer.")
    all(>(0), values) || error("$option values must be positive.")
    return values
end

function parse_args(args)
    config = CpuBlasBenchmarkConfig()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--sizes"
            i += 1
            config.sizes = parse_int_list(args[i], "--sizes")
        elseif arg == "--mesh"
            i += 1
            config.mesh = args[i]
        elseif arg == "--freq"
            i += 1
            config.frequency = parse(Float64, args[i])
        elseif arg == "--quadrature-order"
            i += 1
            config.quadrature_order = parse(Int, args[i])
        elseif arg == "--singular-order"
            i += 1
            config.singular_order = parse(Int, args[i])
        elseif arg == "--symmetry"
            i += 1
            config.symmetry = lowercase(strip(args[i]))
        elseif arg == "--scale"
            i += 1
            config.scale_factor = parse(Float64, args[i])
        elseif arg == "--thread-counts"
            i += 1
            config.thread_counts = parse_int_list(args[i], "--thread-counts")
        elseif arg == "--warmups"
            i += 1
            config.warmups = parse(Int, args[i])
        elseif arg == "--repetitions"
            i += 1
            config.repetitions = parse(Int, args[i])
        elseif arg == "--json"
            i += 1
            config.output = args[i]
        elseif arg == "--help" || arg == "-h"
            println("""
            Usage:
              julia scripts/benchmark_cpu_blas.jl [options]

            Options:
              --sizes CSV           Dense matrix sizes. Default: 441,1390,3502
              --mesh PATH           Assemble and benchmark a real BEAT CPU system instead.
              --freq HZ             Real-system frequency. Default: 1000
              --quadrature-order N  Real-system regular order. Default: 2
              --singular-order N    Real-system singular order. Default: 4
              --symmetry off|x|xy   Real-system symmetry mode. Default: off
              --scale FACTOR        Real-system mesh scale. Default: 0.001
              --thread-counts CSV   BLAS thread counts. Default: 1,2,4,8,16
              --warmups N           Warmups per size/thread count (1-3). Default: 2
              --repetitions N       Measured repetitions. Default: 5
              --json PATH           Output JSON path.
            """)
            exit()
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end
    1 <= config.warmups <= 3 || error("--warmups must be between 1 and 3.")
    config.repetitions >= 1 || error("--repetitions must be at least 1.")
    config.symmetry in ("off", "x", "xy") || error("--symmetry must be off, x, or xy.")
    return config
end

function representative_system(size::Int)
    rng = Xoshiro(0xBEA7 + size)
    matrix = randn(rng, ComplexF32, size, size)
    @inbounds for index in 1:size
        matrix[index, index] += ComplexF32(size, 0)
    end
    rhs = randn(rng, ComplexF32, size)
    return matrix, rhs
end

function real_beat_system(config::CpuBlasBenchmarkConfig)
    T = Float32
    mesh = load_gmsh22_with_tags(config.mesh, T(config.scale_factor))
    symmetry_mode = Symbol(config.symmetry)
    validate_symmetry_fundamental_domain!(mesh, symmetry_mode)
    p1_space = build_p1_space(mesh)
    dp0_space = build_dp0_space(mesh)
    rule = triangle_rule(T, config.quadrature_order)
    singular_cache = build_singular_correction_cache(mesh, config.singular_order)
    k = T(2pi * config.frequency / 343.0)
    println("Assembling real BEAT operators for $(length(mesh.faces)) elements and $(p1_space.global_dof_count) P1 DOFs")
    operators = assemble_regular_galerkin_operators(
        mesh,
        p1_space,
        dp0_space,
        k,
        rule;
        skip_singular=false,
        singular_order=config.singular_order,
        backend=:cpu,
        singular_cache=singular_cache,
        symmetry_mode=symmetry_mode,
    )
    identity_p1_p1 = assemble_l2_identity_matrix(
        mesh,
        p1_space,
        dp0_space,
        rule,
        :p1,
        :p1;
        symmetry_mode=symmetry_mode,
    )
    coupling = Complex{T}(0, 1) / k
    matrix = Complex{T}(0.5) .* Complex{T}.(identity_p1_p1) .-
        operators.double_layer .+
        coupling .* operators.hypersingular
    rng = Xoshiro(0xBEA7 + p1_space.global_dof_count)
    rhs = randn(rng, Complex{T}, p1_space.global_dof_count)
    return matrix, rhs, p1_space.global_dof_count
end

function factor_and_solve(matrix, rhs)
    work = copy(matrix)
    factorization = nothing
    factor_seconds = @elapsed factorization = lu!(work)
    solution = nothing
    solve_seconds = @elapsed solution = factorization \ rhs
    return factor_seconds, solve_seconds, Float64(norm(solution))
end

function benchmark_case(matrix, rhs, thread_count::Int, warmups::Int, repetitions::Int)
    BLAS.set_num_threads(thread_count)
    for _ in 1:warmups
        factor_and_solve(matrix, rhs)
        GC.gc()
    end

    factor_times = Float64[]
    solve_times = Float64[]
    solution_norm = 0.0
    for _ in 1:repetitions
        factor_seconds, solve_seconds, solution_norm = factor_and_solve(matrix, rhs)
        push!(factor_times, factor_seconds)
        push!(solve_times, solve_seconds)
        GC.gc()
    end
    return Dict(
        "blas_threads" => BLAS.get_num_threads(),
        "factor_seconds" => Dict(
            "min" => minimum(factor_times),
            "median" => median(factor_times),
            "max" => maximum(factor_times),
        ),
        "triangular_solve_seconds" => Dict(
            "min" => minimum(solve_times),
            "median" => median(solve_times),
            "max" => maximum(solve_times),
        ),
        "solution_norm" => solution_norm,
    )
end

function main(args=ARGS)
    config = parse_args(args)
    systems = if isempty(config.mesh)
        [(size, representative_system(size)...) for size in config.sizes]
    else
        matrix, rhs, size = real_beat_system(config)
        [(size, matrix, rhs)]
    end
    cases = Dict{String,Any}()
    for (size, matrix, rhs) in systems
        isempty(config.mesh) && println("Building representative ComplexF32 system of size $size")
        size_cases = Dict{String,Any}()
        for thread_count in config.thread_counts
            println("  BLAS threads $thread_count")
            result = benchmark_case(matrix, rhs, thread_count, config.warmups, config.repetitions)
            size_cases[string(thread_count)] = result
            println(@sprintf(
                "    LU median %.6f s | triangular solve median %.6f s",
                result["factor_seconds"]["median"],
                result["triangular_solve_seconds"]["median"],
            ))
        end
        cases[string(size)] = size_cases
    end

    payload = Dict(
        "timestamp" => string(now()),
        "julia_version" => string(VERSION),
        "cpu_name" => Sys.CPU_NAME,
        "cpu_threads" => Sys.CPU_THREADS,
        "blas_config" => string(BLAS.get_config()),
        "warmups" => config.warmups,
        "repetitions" => config.repetitions,
        "sizes" => config.sizes,
        "mesh" => isempty(config.mesh) ? nothing : abspath(config.mesh),
        "frequency_hz" => config.frequency,
        "quadrature_order" => config.quadrature_order,
        "singular_order" => config.singular_order,
        "symmetry" => config.symmetry,
        "thread_counts" => config.thread_counts,
        "cases" => cases,
    )
    mkpath(dirname(config.output))
    open(config.output, "w") do io
        JSON.print(io, payload, 2)
        println(io)
    end
    println("Wrote $(config.output)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
