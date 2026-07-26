include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled

using LinearAlgebra
using Printf
using Statistics
using StaticArrays

Base.@kwdef mutable struct CoupledPrecisionConfig
    fem_mesh::String = normpath(
        joinpath(@__DIR__, "..", "..", "..", "..", "..", "tests", "fixtures", "femvolume.msh"),
    )
    bem_mesh::String = normpath(
        joinpath(
            @__DIR__,
            "..",
            "..",
            "..",
            "..",
            "..",
            "tests",
            "fixtures",
            "exterior_conforming.msh",
        ),
    )
    frequencies::Vector{Float64} = [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
    scale_factor::Float64 = 0.001
    sound_speed::Float64 = 343.0
    density::Float64 = 1.21
    radiator_name::String = "Radiator"
    fem_interface_name::String = "Interface"
    bem_interface_tag::Int = 2
    quadrature_order::Int = 2
    singular_order::Int = 2
    field_points::Int = 128
    field_radius_m::Float64 = 2.0
    response_floor_db::Float64 = -80.0
end

function print_usage()
    println("""
    Compare the directly coupled FEM-BEM solver in Float64 and Float32.

    Usage:
      julia -t 4 --project=src/blab/solvers/julia_local \\
        src/blab/solvers/julia_local/scripts/compare_coupled_precision.jl [options]

    Options:
      --fem PATH                 Gmsh 4.1 tetrahedral FEM mesh.
      --bem PATH                 Gmsh 2.2 triangular BEM mesh.
      --frequencies CSV          Frequencies in Hz.
      --scale VALUE              Mesh-to-metre scale. Default: 0.001.
      --sound-speed VALUE        Sound speed in m/s. Default: 343.
      --density VALUE            Density in kg/m^3. Default: 1.21.
      --radiator-name NAME       FEM moving-boundary physical name. Default: Radiator.
      --fem-interface-name NAME  FEM interface physical name. Default: Interface.
      --bem-interface-tag TAG    BEM interface physical tag. Default: 2.
      --quadrature-order N       Regular quadrature order. Default: 2.
      --singular-order N         Singular quadrature order. Default: 2.
      --field-points N           Fibonacci-sphere observation count. Default: 128.
      --field-radius VALUE       Observation radius in metres. Default: 2.
      --response-floor-db VALUE  Ignore values below this level relative to each
                                 quantity peak for dB/phase statistics. Default: -80.
      --help                     Show this help.
    """)
end

function parse_frequencies(text::String)
    values = Float64[]
    for value in split(text, ',')
        stripped = strip(value)
        isempty(stripped) || push!(values, parse(Float64, stripped))
    end
    isempty(values) && error("--frequencies must contain at least one value.")
    all(>(0.0), values) || error("--frequencies values must be positive.")
    return values
end

function parse_args(args)
    config = CoupledPrecisionConfig()
    index = 1
    while index <= length(args)
        arg = args[index]
        if arg in ("--help", "-h")
            print_usage()
            exit()
        elseif arg == "--fem"
            index += 1
            config.fem_mesh = args[index]
        elseif arg == "--bem"
            index += 1
            config.bem_mesh = args[index]
        elseif arg == "--frequencies"
            index += 1
            config.frequencies = parse_frequencies(args[index])
        elseif arg == "--scale"
            index += 1
            config.scale_factor = parse(Float64, args[index])
        elseif arg == "--sound-speed"
            index += 1
            config.sound_speed = parse(Float64, args[index])
        elseif arg == "--density"
            index += 1
            config.density = parse(Float64, args[index])
        elseif arg == "--radiator-name"
            index += 1
            config.radiator_name = args[index]
        elseif arg == "--fem-interface-name"
            index += 1
            config.fem_interface_name = args[index]
        elseif arg == "--bem-interface-tag"
            index += 1
            config.bem_interface_tag = parse(Int, args[index])
        elseif arg == "--quadrature-order"
            index += 1
            config.quadrature_order = parse(Int, args[index])
        elseif arg == "--singular-order"
            index += 1
            config.singular_order = parse(Int, args[index])
        elseif arg == "--field-points"
            index += 1
            config.field_points = parse(Int, args[index])
        elseif arg == "--field-radius"
            index += 1
            config.field_radius_m = parse(Float64, args[index])
        elseif arg == "--response-floor-db"
            index += 1
            config.response_floor_db = parse(Float64, args[index])
        else
            error("Unknown argument: $arg")
        end
        index += 1
    end
    isfile(config.fem_mesh) || error("FEM mesh does not exist: $(config.fem_mesh)")
    isfile(config.bem_mesh) || error("BEM mesh does not exist: $(config.bem_mesh)")
    config.scale_factor > 0.0 || error("--scale must be positive.")
    config.sound_speed > 0.0 || error("--sound-speed must be positive.")
    config.density > 0.0 || error("--density must be positive.")
    config.quadrature_order > 0 || error("--quadrature-order must be positive.")
    config.singular_order > 0 || error("--singular-order must be positive.")
    config.field_points > 0 || error("--field-points must be positive.")
    config.field_radius_m > 0.0 || error("--field-radius must be positive.")
    config.response_floor_db <= 0.0 || error("--response-floor-db must not be positive.")
    return config
end

function prepare_precision_state(::Type{T}, config::CoupledPrecisionConfig, reference_points) where {T}
    fem_mesh = load_gmsh41_volume(config.fem_mesh, T(config.scale_factor))
    bem_mesh = load_gmsh22_with_tags(config.bem_mesh, T(config.scale_factor))
    interface_map = build_conforming_interface_map(
        fem_mesh,
        bem_mesh,
        physical_tag(fem_mesh, 2, config.fem_interface_name),
        config.bem_interface_tag,
    )
    cache = prepare_coupled_reference_cache(
        fem_mesh,
        bem_mesh,
        interface_map;
        quadrature_order=config.quadrature_order,
        singular_order=config.singular_order,
    )
    points = [SVector{3,T}(point) for point in reference_points]
    return (
        fem_mesh=fem_mesh,
        bem_mesh=bem_mesh,
        interface_map=interface_map,
        cache=cache,
        field_points=points,
        radiator_tag=physical_tag(fem_mesh, 2, config.radiator_name),
    )
end

function solve_precision(state, ::Type{T}, frequency_hz::Float64, config::CoupledPrecisionConfig) where {T}
    system = build_coupled_reference_system(
        state.fem_mesh,
        state.bem_mesh,
        state.interface_map,
        T(frequency_hz),
        T(config.sound_speed),
        T(config.density);
        quadrature_order=config.quadrature_order,
        singular_order=config.singular_order,
        cache=state.cache,
        validation_diagnostics=true,
    )
    solution = solve_coupled_reference_system(system, state.radiator_tag)
    field = evaluate_galerkin_field_cpu(
        state.field_points,
        state.bem_mesh,
        solution.bem_pressure,
        solution.bem_neumann,
        system.wavenumber,
        system.field_cache,
    )
    return (
        fem_pressure=ComplexF64.(solution.fem_pressure),
        bem_pressure=ComplexF64.(solution.bem_pressure),
        interface_flux=ComplexF64.(solution.interface_flux),
        exterior_pressure=ComplexF64.(field),
        relative_residual=Float64(solution.relative_residual),
        pressure_continuity_error=Float64(solution.pressure_continuity_error),
        flux_conservation_error=Float64(solution.flux_conservation_error),
        all_bem_replay_error=Float64(solution.all_bem_replay_error),
    )
end

function percentile(values, probability)
    isempty(values) && return NaN
    return Float64(quantile(values, probability))
end

function comparison_metrics(reference, candidate, response_floor_db::Float64)
    length(reference) == length(candidate) || error("Precision results have different vector lengths.")
    difference = candidate .- reference
    reference_norm = norm(reference)
    reference_peak = maximum(abs.(reference))
    floor_ratio = 10.0^(response_floor_db / 20.0)
    threshold = reference_peak * floor_ratio
    selected = findall(value -> abs(value) >= threshold && abs(value) > 0.0, reference)
    magnitude_delta_db = Float64[]
    phase_delta_deg = Float64[]
    for index in selected
        push!(magnitude_delta_db, abs(20.0 * log10(abs(candidate[index]) / abs(reference[index]))))
        push!(
            phase_delta_deg,
            abs(rad2deg(angle(candidate[index] * conj(reference[index])))),
        )
    end
    return (
        relative_l2=Float64(norm(difference) / max(reference_norm, eps(Float64))),
        relative_max_abs=Float64(maximum(abs.(difference)) / max(reference_peak, eps(Float64))),
        rms_db=isempty(magnitude_delta_db) ? NaN : sqrt(mean(abs2, magnitude_delta_db)),
        p95_db=percentile(magnitude_delta_db, 0.95),
        max_db=isempty(magnitude_delta_db) ? NaN : maximum(magnitude_delta_db),
        p95_phase_deg=percentile(phase_delta_deg, 0.95),
        max_phase_deg=isempty(phase_delta_deg) ? NaN : maximum(phase_delta_deg),
        selected_count=length(selected),
        total_count=length(reference),
    )
end

function print_metric(frequency_hz, name, metrics)
    @printf(
        "%8.1f  %-10s  %10.3e  %10.3e  %9.4f  %9.4f  %9.4f  %9.4f  %5d/%-5d\n",
        frequency_hz,
        name,
        metrics.relative_l2,
        metrics.relative_max_abs,
        metrics.rms_db,
        metrics.max_db,
        metrics.p95_phase_deg,
        metrics.max_phase_deg,
        metrics.selected_count,
        metrics.total_count,
    )
end

function run_comparison(config::CoupledPrecisionConfig)
    reference_points = fibonacci_sphere(config.field_points, config.field_radius_m)
    println("Preparing Float64 coupled cache...")
    state64 = prepare_precision_state(Float64, config, reference_points)
    println("Preparing Float32 coupled cache...")
    state32 = prepare_precision_state(Float32, config, reference_points)
    state64.interface_map.fem_face_indices == state32.interface_map.fem_face_indices ||
        error("Float64 and Float32 FEM interface face mappings differ.")
    state64.interface_map.bem_face_indices == state32.interface_map.bem_face_indices ||
        error("Float64 and Float32 BEM interface face mappings differ.")
    state64.interface_map.normal_sign == state32.interface_map.normal_sign ||
        error("Float64 and Float32 interface orientations differ.")

    names = (:fem_pressure, :bem_pressure, :interface_flux, :exterior_pressure)
    maxima = Dict(
        name => Dict(
            :relative_l2 => 0.0,
            :relative_max_abs => 0.0,
            :max_db => 0.0,
            :max_phase_deg => 0.0,
        )
        for name in names
    )
    diagnostic_maxima = Dict(
        :float64_residual => 0.0,
        :float32_residual => 0.0,
        :float64_continuity => 0.0,
        :float32_continuity => 0.0,
        :float64_flux => 0.0,
        :float32_flux => 0.0,
        :float64_replay => 0.0,
        :float32_replay => 0.0,
    )

    println()
    println(
        "    Hz    quantity         rel L2     rel max     RMS dB     max dB" *
        "   p95 phase  max phase       used",
    )
    println("-"^119)
    for frequency_hz in config.frequencies
        result64 = solve_precision(state64, Float64, frequency_hz, config)
        GC.gc()
        result32 = solve_precision(state32, Float32, frequency_hz, config)
        GC.gc()
        for name in names
            metrics = comparison_metrics(
                getproperty(result64, name),
                getproperty(result32, name),
                config.response_floor_db,
            )
            print_metric(frequency_hz, String(name), metrics)
            maxima[name][:relative_l2] = max(maxima[name][:relative_l2], metrics.relative_l2)
            maxima[name][:relative_max_abs] = max(
                maxima[name][:relative_max_abs],
                metrics.relative_max_abs,
            )
            maxima[name][:max_db] = max(maxima[name][:max_db], metrics.max_db)
            maxima[name][:max_phase_deg] = max(
                maxima[name][:max_phase_deg],
                metrics.max_phase_deg,
            )
        end
        diagnostic_maxima[:float64_residual] = max(
            diagnostic_maxima[:float64_residual],
            result64.relative_residual,
        )
        diagnostic_maxima[:float32_residual] = max(
            diagnostic_maxima[:float32_residual],
            result32.relative_residual,
        )
        diagnostic_maxima[:float64_continuity] = max(
            diagnostic_maxima[:float64_continuity],
            result64.pressure_continuity_error,
        )
        diagnostic_maxima[:float32_continuity] = max(
            diagnostic_maxima[:float32_continuity],
            result32.pressure_continuity_error,
        )
        diagnostic_maxima[:float64_flux] = max(
            diagnostic_maxima[:float64_flux],
            result64.flux_conservation_error,
        )
        diagnostic_maxima[:float32_flux] = max(
            diagnostic_maxima[:float32_flux],
            result32.flux_conservation_error,
        )
        diagnostic_maxima[:float64_replay] = max(
            diagnostic_maxima[:float64_replay],
            result64.all_bem_replay_error,
        )
        diagnostic_maxima[:float32_replay] = max(
            diagnostic_maxima[:float32_replay],
            result32.all_bem_replay_error,
        )
    end

    println()
    println("Worst case across $(length(config.frequencies)) frequencies:")
    for name in names
        values = maxima[name]
        @printf(
            "  %-17s rel L2 %.3e, rel max %.3e, max |dB| %.4f, max |phase| %.4f deg\n",
            String(name),
            values[:relative_l2],
            values[:relative_max_abs],
            values[:max_db],
            values[:max_phase_deg],
        )
    end
    @printf(
        "  residual            Float64 %.3e, Float32 %.3e\n",
        diagnostic_maxima[:float64_residual],
        diagnostic_maxima[:float32_residual],
    )
    @printf(
        "  continuity          Float64 %.3e, Float32 %.3e\n",
        diagnostic_maxima[:float64_continuity],
        diagnostic_maxima[:float32_continuity],
    )
    @printf(
        "  flux conservation   Float64 %.3e, Float32 %.3e\n",
        diagnostic_maxima[:float64_flux],
        diagnostic_maxima[:float32_flux],
    )
    @printf(
        "  all-BEM replay      Float64 %.3e, Float32 %.3e\n",
        diagnostic_maxima[:float64_replay],
        diagnostic_maxima[:float32_replay],
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_comparison(parse_args(ARGS))
end
