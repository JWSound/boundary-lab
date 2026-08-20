include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))

using Dates
using LinearAlgebra
using Printf
using Statistics
using .BeatEngineCore

Base.@kwdef mutable struct RocmBenchmarkConfig
    mesh::String = joinpath(@__DIR__, "..", "test_meshes", "sample.msh")
    frequency::Float64 = 1000.0
    quadrature_order::Int = 4
    singular_order::Int = 4
    eval_points::Int = 144
    warmups::Int = 2
    repetitions::Int = 5
    scale_factor::Float64 = 0.001
    sound_speed::Float64 = 343.0
    rho::Float64 = 1.21
    tag_throat::Int = 2
    distance::Float64 = 2.0
    skip_solve::Bool = false
    skip_field::Bool = false
    output::String = joinpath(@__DIR__, "..", "results", "benchmark_rocm_warm.json")
    verbose::Bool = false
end

function parse_args(args)
    config = RocmBenchmarkConfig()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--mesh"
            i += 1; config.mesh = args[i]
        elseif arg == "--freq"
            i += 1; config.frequency = parse(Float64, args[i])
        elseif arg == "--quadrature-order"
            i += 1; config.quadrature_order = parse(Int, args[i])
        elseif arg == "--singular-order"
            i += 1; config.singular_order = parse(Int, args[i])
        elseif arg == "--eval-points"
            i += 1; config.eval_points = parse(Int, args[i])
        elseif arg == "--warmups"
            i += 1; config.warmups = parse(Int, args[i])
        elseif arg == "--repetitions"
            i += 1; config.repetitions = parse(Int, args[i])
        elseif arg == "--skip-solve"
            config.skip_solve = true
        elseif arg == "--skip-field"
            config.skip_field = true
        elseif arg == "--json"
            i += 1; config.output = args[i]
        elseif arg == "--verbose"
            config.verbose = true
        elseif arg == "--help" || arg == "-h"
            println("""
            Usage: benchmark_rocm.jl [options]
              --mesh PATH --freq HZ --quadrature-order N --singular-order N
              --eval-points N --warmups N --repetitions N
              --skip-solve --skip-field --json PATH --verbose
            """)
            exit()
        else
            error("Unknown argument: $(arg)")
        end
        i += 1
    end
    config.warmups >= 1 || error("--warmups must be at least 1.")
    config.repetitions >= 1 || error("--repetitions must be at least 1.")
    return config
end

function synchronized_elapsed(thunk, amdgpu)
    value = nothing
    amdgpu.synchronize()
    elapsed = @elapsed begin
        value = thunk()
        amdgpu.synchronize()
    end
    return value, elapsed
end

function build_setup(config::RocmBenchmarkConfig)
    amdgpu = BeatEngineCore.AMDGPU_MODULE
    amdgpu === nothing && error("AMDGPU.jl did not load.")
    BeatEngineCore._require_rocm!(rocsolver=!config.skip_solve)
    setup_timings = Dict{String,Float64}()
    mesh = nothing
    setup_timings["mesh_load"] = @elapsed mesh = load_gmsh22_with_tags(config.mesh, Float32(config.scale_factor))
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, config.quadrature_order)
    singular_cache = nothing
    setup_timings["singular_host_cache"] = @elapsed singular_cache =
        build_singular_correction_cache(mesh, config.singular_order)
    regular_cache = nothing
    setup_timings["regular_device_cache"] = @elapsed regular_cache =
        build_rocm_regular_assembly_cache(
            mesh, p1, dp0, rule;
            singular_order=config.singular_order,
            symmetry_mode=:off,
        )
    singular_device_cache = nothing
    setup_timings["singular_device_cache"] = @elapsed singular_device_cache =
        build_rocm_singular_correction_cache(singular_cache)

    identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1)
    identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0)
    identity_device_cache = config.skip_solve ? nothing :
        build_rocm_burton_miller_identity_cache(identity_p1_p1, identity_p1_dp0, Float32)

    field_cache = nothing
    eval_points = typeof(mesh.vertices[1])[]
    if !config.skip_field && config.eval_points > 0
        cpu_field_cache = build_field_evaluation_cache(mesh, rule)
        setup_timings["field_device_cache"] = @elapsed field_cache =
            build_rocm_field_evaluation_cache(cpu_field_cache)
        eval_points = fibonacci_sphere(config.eval_points, Float32(config.distance))
    end

    q_neumann = zeros(ComplexF32, length(mesh.faces))
    omega = Float32(2pi * config.frequency)
    throat_indices = findall(==(config.tag_throat), mesh.physical_tags)
    q_neumann[throat_indices] .= ComplexF32(0, Float32(config.rho) * omega)
    return (
        amdgpu=amdgpu,
        mesh=mesh,
        p1=p1,
        dp0=dp0,
        rule=rule,
        singular_cache=singular_cache,
        regular_cache=regular_cache,
        singular_device_cache=singular_device_cache,
        identity_device_cache=identity_device_cache,
        field_cache=field_cache,
        eval_points=eval_points,
        q_neumann=q_neumann,
        throat_count=length(throat_indices),
        k=Float32(2pi * config.frequency / config.sound_speed),
        setup_timings=setup_timings,
    )
end

function release_setup!(setup)
    setup.field_cache === nothing || release_rocm_field_evaluation_cache!(setup.field_cache)
    setup.identity_device_cache === nothing ||
        release_rocm_burton_miller_identity_cache!(setup.identity_device_cache)
    release_rocm_singular_correction_cache!(setup.singular_device_cache)
    release_rocm_regular_assembly_cache!(setup.regular_cache)
    return nothing
end

function run_iteration(config::RocmBenchmarkConfig, setup)
    timings = Dict{String,Float64}()
    operators, timings["assembly_total"] = synchronized_elapsed(setup.amdgpu) do
        assemble_regular_galerkin_operators(
            setup.mesh, setup.p1, setup.dp0, setup.k, setup.rule;
            skip_singular=false,
            singular_order=config.singular_order,
            backend=:rocm,
            device_cache=setup.regular_cache,
            singular_cache=setup.singular_cache,
            device_singular_cache=setup.singular_device_cache,
            rocm_assembly_mode=:native,
            timing=timings,
            symmetry_mode=:off,
        )
    end

    pressure = nothing
    if config.skip_solve
        timings["solve_total"] = 0.0
    else
        pressure, timings["solve_total"] = synchronized_elapsed(setup.amdgpu) do
            solve_burton_miller_neumann(
                operators,
                setup.identity_device_cache,
                setup.q_neumann,
                setup.k,
            )
        end
    end

    field = nothing
    if config.skip_field || config.eval_points == 0 || pressure === nothing
        timings["field_total"] = 0.0
    else
        field, timings["field_total"] = synchronized_elapsed(setup.amdgpu) do
            evaluate_galerkin_field_rocm(
                setup.eval_points,
                setup.mesh,
                pressure,
                setup.q_neumann,
                setup.k,
                setup.field_cache,
            )
        end
    end
    pressure_norm = pressure === nothing ? nothing : Float64(norm(pressure))
    field_norm = field === nothing ? nothing : Float64(norm(field))
    release_operator_storage!(operators)
    setup.amdgpu.synchronize()
    return Dict{String,Any}(
        "timings_seconds" => timings,
        "pressure_norm" => pressure_norm,
        "field_norm" => field_norm,
        "regular_pairs" => operators.regular_pairs,
        "singular_pairs" => operators.singular_pairs,
        "regular_kernel_mode" => operators.regular_kernel_mode,
        "regular_kernel_operator_mode" => get(operators, :regular_kernel_operator_mode, nothing),
        "regular_kernel_color_count" => get(operators, :regular_kernel_color_count, nothing),
        "regular_kernel_launches" => get(operators, :regular_kernel_launches, nothing),
    )
end

function summarize_runs(runs)
    timing_keys = sort(collect(keys(runs[1]["timings_seconds"])))
    summary = Dict{String,Any}()
    for key in timing_keys
        values = [run["timings_seconds"][key] for run in runs]
        summary[key] = Dict(
            "min" => minimum(values),
            "median" => median(values),
            "mean" => mean(values),
            "max" => maximum(values),
        )
    end
    return summary
end

json_escape(value::AbstractString) = replace(replace(replace(value, "\\" => "\\\\"), "\"" => "\\\""), "\n" => "\\n")

function write_json_value(io, value)
    if value === nothing
        print(io, "null")
    elseif value isa Bool
        print(io, value ? "true" : "false")
    elseif value isa Number
        print(io, value)
    elseif value isa AbstractString
        print(io, "\"", json_escape(value), "\"")
    elseif value isa AbstractVector || value isa Tuple
        print(io, "[")
        for (index, item) in enumerate(value)
            index == 1 || print(io, ",")
            write_json_value(io, item)
        end
        print(io, "]")
    elseif value isa AbstractDict
        print(io, "{")
        for (index, key) in enumerate(sort(collect(keys(value)); by=string))
            index == 1 || print(io, ",")
            write_json_value(io, string(key))
            print(io, ":")
            write_json_value(io, value[key])
        end
        print(io, "}")
    else
        write_json_value(io, string(value))
    end
end

function write_json(path, payload)
    mkpath(dirname(path))
    open(path, "w") do io
        write_json_value(io, payload)
        println(io)
    end
end

function print_summary(payload, config)
    metadata = payload["metadata"]
    println(@sprintf(
        "ROCm warm benchmark | %s | %d faces / %d P1 | q%d/s%d | eval %d",
        basename(config.mesh),
        metadata["mesh_faces"],
        metadata["p1_dofs"],
        config.quadrature_order,
        config.singular_order,
        config.eval_points,
    ))
    for key in (
        "assembly_total",
        "rocm_native_operator_alloc",
        "rocm_native_regular_kernel",
        "rocm_native_singular_kernel",
        "solve_total",
        "field_total",
    )
        haskey(payload["summary_seconds"], key) || continue
        println(@sprintf("  %-36s %.6f s", key, payload["summary_seconds"][key]["median"]))
    end
    if config.verbose
        println("All warmed stage medians:")
        for key in sort(collect(keys(payload["summary_seconds"])))
            println(@sprintf("  %-36s %.6f s", key, payload["summary_seconds"][key]["median"]))
        end
    end
end

function main(args=ARGS)
    config = parse_args(args)
    setup = build_setup(config)
    runs = Dict{String,Any}[]
    try
        println("device=$(setup.amdgpu.device())")
        println("Warming persistent caches and compiled kernels...")
        for index in 1:config.warmups
            println("warmup=$(index)/$(config.warmups)")
            run_iteration(config, setup)
            GC.gc(false)
        end
        for index in 1:config.repetitions
            println("measured=$(index)/$(config.repetitions)")
            push!(runs, run_iteration(config, setup))
            GC.gc(false)
        end
        payload = Dict{String,Any}(
            "timestamp" => string(now()),
            "config" => Dict(
                "mesh" => abspath(config.mesh),
                "frequency_hz" => config.frequency,
                "quadrature_order" => config.quadrature_order,
                "singular_order" => config.singular_order,
                "eval_points" => config.eval_points,
                "warmups" => config.warmups,
                "repetitions" => config.repetitions,
                "regular_kernel_mode" => string(BeatEngineCore._normalized_rocm_regular_kernel_mode()),
                "pair_operator_mode" => "partial_fused",
            ),
            "metadata" => Dict(
                "device" => string(setup.amdgpu.device()),
                "julia_version" => string(VERSION),
                "mesh_faces" => length(setup.mesh.faces),
                "mesh_vertices" => length(setup.mesh.vertices),
                "p1_dofs" => setup.p1.global_dof_count,
                "dp0_dofs" => setup.dp0.global_dof_count,
                "throat_elements" => setup.throat_count,
                "setup_timings_seconds" => setup.setup_timings,
            ),
            "runs" => runs,
            "summary_seconds" => summarize_runs(runs),
        )
        write_json(config.output, payload)
        print_summary(payload, config)
        println("Wrote $(config.output)")
    finally
        release_setup!(setup)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
