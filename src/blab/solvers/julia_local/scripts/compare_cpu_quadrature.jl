include(joinpath(@__DIR__, "benchmark_cpu.jl"))

using Dates
using LinearAlgebra
using Printf
using Statistics

Base.@kwdef mutable struct CpuQuadratureCompareConfig
    mesh::String = joinpath(@__DIR__, "..", "test_meshes", "sample.msh")
    frequencies::Vector{Float64} = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0]
    precision_name::String = "Float32"
    reference_order::Int = 4
    candidate_order::Int = 2
    candidate_mode::String = "wavelength"
    singular_order::Int = 2
    output_points::Int = 72
    subset_faces::Int = 128
    symmetry::String = "off"
    threaded_assembly::Bool = true
    scale_factor::Float64 = 0.001
    sound_speed::Float64 = 343.0
    rho::Float64 = 1.21
    tag_throat::Int = 2
    distance::Float64 = 2.0
    wavelength_mesh_stat::String = "p90"
    wavelength_kh_q1_max::Float64 = 0.0
    wavelength_kh_q2_max::Float64 = 2.0
    output::String = joinpath(@__DIR__, "..", "results", "cpu_quadrature_compare.json")
end

function print_compare_usage()
    println("""
    Usage:
      julia scripts/compare_cpu_quadrature.jl [options]

    Options:
      --mesh PATH                    Mesh path. Default: test_meshes/sample.msh
      --frequencies LIST             Comma-separated Hz list. Default: 20,...,20000
      --precision Float32|Float64    Numeric precision. Default: Float32
      --reference-order N            Fixed reference regular quadrature order. Default: 4
      --candidate-order N            Fixed candidate order when mode=fixed. Default: 2
      --candidate-mode fixed|wavelength
                                      Candidate quadrature mode. Default: wavelength
      --singular-order N             Singular quadrature order for both runs. Default: 2
      --output-points N              Fibonacci sphere points for field/SPL comparison. Default: 72. 0 disables output checks.
      --subset-faces N               Use first N faces. Default: 128. 0 means full mesh.
      --symmetry off|x|xy            Symmetry mode. Default: off
      --serial-assembly              Disable colored threaded CPU operator assembly.
      --scale FACTOR                 Mesh scale factor. Default: 0.001
      --sound-speed VALUE            Sound speed in m/s. Default: 343
      --rho VALUE                    Air density for throat RHS. Default: 1.21
      --tag-throat N                 Physical tag used for Neumann throat RHS. Default: 2
      --distance VALUE               Field comparison radius in meters. Default: 2
      --wavelength-mesh-stat median|p75|p90|max
                                      Area statistic used for h=sqrt(area). Default: p90
      --wavelength-kh-q1-max VALUE    q1 cutoff for k*h. Default: 0.0
      --wavelength-kh-q2-max VALUE    q2 cutoff for k*h. Default: 2.0
      --json PATH                    Write JSON results. Default: results/cpu_quadrature_compare.json
      --help                         Print this message.
    """)
end

function parse_frequency_list(value::String)
    freqs = Float64[]
    for item in split(value, ",")
        stripped = strip(item)
        isempty(stripped) && continue
        push!(freqs, parse(Float64, stripped))
    end
    isempty(freqs) && error("--frequencies must contain at least one value.")
    return freqs
end

function parse_compare_args(args)
    config = CpuQuadratureCompareConfig()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            print_compare_usage()
            exit()
        elseif arg == "--mesh"
            i += 1; config.mesh = args[i]
        elseif arg == "--frequencies"
            i += 1; config.frequencies = parse_frequency_list(args[i])
        elseif arg == "--precision"
            i += 1; config.precision_name = args[i]
        elseif arg == "--reference-order"
            i += 1; config.reference_order = parse(Int, args[i])
        elseif arg == "--candidate-order"
            i += 1; config.candidate_order = parse(Int, args[i])
        elseif arg == "--candidate-mode"
            i += 1; config.candidate_mode = lowercase(strip(args[i]))
        elseif arg == "--singular-order"
            i += 1; config.singular_order = parse(Int, args[i])
        elseif arg == "--output-points"
            i += 1; config.output_points = parse(Int, args[i])
        elseif arg == "--subset-faces"
            i += 1; config.subset_faces = parse(Int, args[i])
        elseif arg == "--symmetry"
            i += 1; config.symmetry = lowercase(strip(args[i]))
        elseif arg == "--serial-assembly"
            config.threaded_assembly = false
        elseif arg == "--scale"
            i += 1; config.scale_factor = parse(Float64, args[i])
        elseif arg == "--sound-speed"
            i += 1; config.sound_speed = parse(Float64, args[i])
        elseif arg == "--rho"
            i += 1; config.rho = parse(Float64, args[i])
        elseif arg == "--tag-throat"
            i += 1; config.tag_throat = parse(Int, args[i])
        elseif arg == "--distance"
            i += 1; config.distance = parse(Float64, args[i])
        elseif arg == "--wavelength-mesh-stat"
            i += 1; config.wavelength_mesh_stat = lowercase(strip(args[i]))
        elseif arg == "--wavelength-kh-q1-max"
            i += 1; config.wavelength_kh_q1_max = parse(Float64, args[i])
        elseif arg == "--wavelength-kh-q2-max"
            i += 1; config.wavelength_kh_q2_max = parse(Float64, args[i])
        elseif arg == "--json"
            i += 1; config.output = args[i]
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end

    config.candidate_mode in ("fixed", "wavelength") || error("Unsupported candidate mode: $(config.candidate_mode)")
    config.wavelength_mesh_stat in ("median", "p75", "p90", "max") || error("Unsupported wavelength mesh stat: $(config.wavelength_mesh_stat)")
    config.wavelength_kh_q1_max >= 0.0 || error("--wavelength-kh-q1-max must be non-negative.")
    config.wavelength_kh_q2_max > config.wavelength_kh_q1_max || error("--wavelength-kh-q2-max must be greater than --wavelength-kh-q1-max.")
    config.output_points >= 0 || error("--output-points must be non-negative.")
    isempty(config.frequencies) && error("--frequencies must contain at least one value.")
    return config
end

function compare_operator(reference, candidate)
    diff = candidate .- reference
    ref_norm = Float64(norm(reference))
    diff_norm = Float64(norm(diff))
    ref_max = Float64(maximum(abs.(reference)))
    max_abs = Float64(maximum(abs.(diff)))
    return Dict{String,Any}(
        "reference_norm" => ref_norm,
        "candidate_norm" => Float64(norm(candidate)),
        "diff_norm" => diff_norm,
        "relative_frobenius" => ref_norm == 0.0 ? nothing : diff_norm / ref_norm,
        "max_abs" => max_abs,
        "relative_max_abs" => ref_max == 0.0 ? nothing : max_abs / ref_max,
    )
end

function operator_errors(reference, candidate)
    return Dict{String,Any}(
        "single_layer" => compare_operator(reference.single_layer, candidate.single_layer),
        "double_layer" => compare_operator(reference.double_layer, candidate.double_layer),
        "adjoint_double_layer" => compare_operator(reference.adjoint_double_layer, candidate.adjoint_double_layer),
        "hypersingular" => compare_operator(reference.hypersingular, candidate.hypersingular),
    )
end

function complex_vector_errors(reference, candidate)
    diff = candidate .- reference
    ref_norm = Float64(norm(reference))
    diff_norm = Float64(norm(diff))
    ref_max = Float64(maximum(abs.(reference)))
    max_abs = Float64(maximum(abs.(diff)))
    return Dict{String,Any}(
        "reference_norm" => ref_norm,
        "candidate_norm" => Float64(norm(candidate)),
        "diff_norm" => diff_norm,
        "relative_l2" => ref_norm == 0.0 ? nothing : diff_norm / ref_norm,
        "max_abs" => max_abs,
        "relative_max_abs" => ref_max == 0.0 ? nothing : max_abs / ref_max,
    )
end

function spl_db(values)
    return 20.0 .* log10.(max.(abs.(values), eps(Float64)) ./ 20e-6)
end

function spl_delta_errors(reference, candidate)
    delta = abs.(spl_db(reference) .- spl_db(candidate))
    return Dict{String,Any}(
        "rms_db" => Float64(sqrt(mean(delta .^ 2))),
        "p95_db" => Float64(quantile(delta, 0.95)),
        "max_db" => Float64(maximum(delta)),
    )
end

function max_metric(results, operator_name::String, metric::String)
    values = Float64[]
    for result in results
        value = result["operator_errors"][operator_name][metric]
        value === nothing && continue
        push!(values, Float64(value))
    end
    return isempty(values) ? nothing : maximum(values)
end

function max_result_metric(results, section::String, metric::String)
    values = Float64[]
    for result in results
        section_value = result[section]
        section_value === nothing && continue
        value = section_value[metric]
        value === nothing && continue
        push!(values, Float64(value))
    end
    return isempty(values) ? nothing : maximum(values)
end

function assemble_cpu_operators(mesh, p1_space, dp0_space, rule, k, singular_order, element_indices, singular_cache, symmetry_mode, threaded)
    timings = Dict{String,Float64}()
    operators = timed_stage!(timings, "operator_total_assembly") do
        assemble_regular_galerkin_operators(
            mesh,
            p1_space,
            dp0_space,
            k,
            rule;
            skip_singular=false,
            singular_order=singular_order,
            element_indices=element_indices,
            threaded=threaded,
            backend=:cpu,
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
    return operators, timings
end

function identity_pair(mesh, p1_space, dp0_space, rule, symmetry_mode)
    identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1_space, dp0_space, rule, :p1, :p1; symmetry_mode=symmetry_mode)
    identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1_space, dp0_space, rule, :p1, :dp0; symmetry_mode=symmetry_mode)
    return identity_p1_p1, identity_p1_dp0
end

function get_cached!(cache::Dict{Int,Any}, order::Int, builder)
    return get!(cache, order) do
        builder()
    end
end

get_cached!(builder, cache::Dict{Int,Any}, order::Int) = get_cached!(cache, order, builder)

function compare_payload(config::CpuQuadratureCompareConfig)
    T = precision_type(config.precision_name)
    mesh = load_gmsh22_with_tags(config.mesh, T(config.scale_factor))
    symmetry_mode = Symbol(config.symmetry)
    validate_symmetry_fundamental_domain!(mesh, symmetry_mode)
    p1_space = build_p1_space(mesh)
    dp0_space = build_dp0_space(mesh)
    element_count = config.subset_faces > 0 ? min(config.subset_faces, length(mesh.faces)) : length(mesh.faces)
    element_indices = 1:element_count
    singular_cache = build_singular_correction_cache(mesh, config.singular_order, element_indices)
    reference_rule = triangle_rule(T, config.reference_order)
    reference_identity = identity_pair(mesh, p1_space, dp0_space, reference_rule, symmetry_mode)
    reference_field_cache = config.output_points > 0 ? build_field_evaluation_cache(mesh, reference_rule; symmetry_mode=symmetry_mode) : nothing
    candidate_identity_cache = Dict{Int,Any}()
    candidate_field_cache_by_order = Dict{Int,Any}()
    field_points = config.output_points > 0 ? fibonacci_sphere(config.output_points, T(config.distance)) : nothing

    results = Dict{String,Any}[]
    for frequency in config.frequencies
        println(@sprintf("Comparing %.1f Hz", frequency))
        k = T(2pi * frequency / config.sound_speed)
        reference_operators, reference_timings = assemble_cpu_operators(
            mesh,
            p1_space,
            dp0_space,
            reference_rule,
            k,
            config.singular_order,
            element_indices,
            singular_cache,
            symmetry_mode,
            config.threaded_assembly,
        )

        selector_config = CpuBenchmarkConfig(
            mesh=config.mesh,
            frequency=frequency,
            precision_name=config.precision_name,
            quadrature_order=config.candidate_order,
            quadrature_mode=config.candidate_mode,
            wavelength_mesh_stat=config.wavelength_mesh_stat,
            wavelength_kh_q1_max=config.wavelength_kh_q1_max,
            wavelength_kh_q2_max=config.wavelength_kh_q2_max,
            singular_order=config.singular_order,
            subset_faces=config.subset_faces,
            symmetry=config.symmetry,
            threaded_assembly=config.threaded_assembly,
            scale_factor=config.scale_factor,
            sound_speed=config.sound_speed,
            rho=config.rho,
            tag_throat=config.tag_throat,
            distance=config.distance,
        )
        selection = selected_regular_quadrature_order(selector_config, mesh, element_indices, T)
        candidate_rule = triangle_rule(T, selection.order)
        candidate_identity = get_cached!(candidate_identity_cache, selection.order) do
            identity_pair(mesh, p1_space, dp0_space, candidate_rule, symmetry_mode)
        end
        candidate_field_cache = config.output_points > 0 ? get_cached!(candidate_field_cache_by_order, selection.order) do
            build_field_evaluation_cache(mesh, candidate_rule; symmetry_mode=symmetry_mode)
        end : nothing
        candidate_operators, candidate_timings = assemble_cpu_operators(
            mesh,
            p1_space,
            dp0_space,
            candidate_rule,
            k,
            config.singular_order,
            element_indices,
            singular_cache,
            symmetry_mode,
            config.threaded_assembly,
        )

        speedup = candidate_timings["operator_total_assembly"] == 0.0 ? nothing :
            reference_timings["operator_total_assembly"] / candidate_timings["operator_total_assembly"]
        q_neumann, throat_indices = throat_rhs(mesh, selector_config, T)
        reference_pressure = solve_cpu_timed!(reference_timings, reference_operators, reference_identity[1], reference_identity[2], q_neumann, k)
        candidate_pressure = solve_cpu_timed!(candidate_timings, candidate_operators, candidate_identity[1], candidate_identity[2], q_neumann, k)
        pressure_errors = complex_vector_errors(reference_pressure, candidate_pressure)
        field_errors = nothing
        spl_errors = nothing
        if config.output_points > 0
            reference_field = timed_stage!(reference_timings, "field_evaluation") do
                evaluate_galerkin_field_cpu(field_points, mesh, reference_pressure, q_neumann, k, reference_field_cache)
            end
            candidate_field = timed_stage!(candidate_timings, "field_evaluation") do
                evaluate_galerkin_field_cpu(field_points, mesh, candidate_pressure, q_neumann, k, candidate_field_cache)
            end
            field_errors = complex_vector_errors(reference_field, candidate_field)
            spl_errors = spl_delta_errors(reference_field, candidate_field)
        else
            reference_timings["field_evaluation"] = 0.0
            candidate_timings["field_evaluation"] = 0.0
        end
        push!(results, Dict{String,Any}(
            "frequency_hz" => frequency,
            "reference_quadrature_order" => config.reference_order,
            "candidate_mode" => config.candidate_mode,
            "candidate_requested_order" => config.candidate_order,
            "candidate_selected_order" => selection.order,
            "wavelength_mesh_stat" => config.wavelength_mesh_stat,
            "wavelength_mesh_area_stat_m2" => selection.mesh_area_stat,
            "wavelength_element_length_m" => selection.element_length_m,
            "wavelength_kh" => selection.kh,
            "wavelength_kh_q1_max" => config.wavelength_kh_q1_max,
            "wavelength_kh_q2_max" => config.wavelength_kh_q2_max,
            "reference_timings_seconds" => reference_timings,
            "candidate_timings_seconds" => candidate_timings,
            "operator_assembly_speedup" => speedup,
            "reference_regular_pairs" => get(reference_operators, :regular_pairs, nothing),
            "candidate_regular_pairs" => get(candidate_operators, :regular_pairs, nothing),
            "singular_pairs" => get(candidate_operators, :singular_pairs, nothing),
            "skipped_pairs" => get(candidate_operators, :skipped_pairs, nothing),
            "throat_elements" => length(throat_indices),
            "operator_errors" => operator_errors(reference_operators, candidate_operators),
            "pressure_errors" => pressure_errors,
            "field_errors" => field_errors,
            "spl_delta" => spl_errors,
        ))
        GC.gc()
    end

    speedups = [Float64(result["operator_assembly_speedup"]) for result in results if result["operator_assembly_speedup"] !== nothing]
    summary = Dict{String,Any}(
        "frequency_count" => length(results),
        "median_operator_assembly_speedup" => isempty(speedups) ? nothing : median(speedups),
        "max_single_layer_relative_frobenius" => max_metric(results, "single_layer", "relative_frobenius"),
        "max_double_layer_relative_frobenius" => max_metric(results, "double_layer", "relative_frobenius"),
        "max_adjoint_double_layer_relative_frobenius" => max_metric(results, "adjoint_double_layer", "relative_frobenius"),
        "max_hypersingular_relative_frobenius" => max_metric(results, "hypersingular", "relative_frobenius"),
        "max_pressure_relative_l2" => max_result_metric(results, "pressure_errors", "relative_l2"),
        "max_pressure_relative_max_abs" => max_result_metric(results, "pressure_errors", "relative_max_abs"),
        "max_field_relative_l2" => max_result_metric(results, "field_errors", "relative_l2"),
        "max_spl_rms_db" => max_result_metric(results, "spl_delta", "rms_db"),
        "max_spl_p95_db" => max_result_metric(results, "spl_delta", "p95_db"),
        "max_spl_max_db" => max_result_metric(results, "spl_delta", "max_db"),
    )

    return Dict{String,Any}(
        "config" => Dict{String,Any}(
            "mesh" => abspath(config.mesh),
            "frequencies_hz" => config.frequencies,
            "precision" => string(T),
            "backend" => "cpu",
            "reference_order" => config.reference_order,
            "candidate_order" => config.candidate_order,
            "candidate_mode" => config.candidate_mode,
            "singular_order" => config.singular_order,
            "output_points" => config.output_points,
            "subset_faces" => config.subset_faces,
            "symmetry" => config.symmetry,
            "threaded_assembly" => config.threaded_assembly,
            "scale_factor" => config.scale_factor,
            "sound_speed" => config.sound_speed,
            "rho" => config.rho,
            "tag_throat" => config.tag_throat,
            "distance" => config.distance,
            "wavelength_mesh_stat" => config.wavelength_mesh_stat,
            "wavelength_kh_q1_max" => config.wavelength_kh_q1_max,
            "wavelength_kh_q2_max" => config.wavelength_kh_q2_max,
        ),
        "metadata" => Dict{String,Any}(
            "timestamp" => string(now()),
            "git_commit" => git_commit(),
            "julia_version" => string(VERSION),
            "threads" => Threads.nthreads(),
            "mesh_faces" => length(mesh.faces),
            "mesh_vertices" => length(mesh.vertices),
            "p1_dofs" => p1_space.global_dof_count,
            "dp0_dofs" => dp0_space.global_dof_count,
            "element_count" => element_count,
            "subset_run" => element_count != length(mesh.faces),
            "singular_cache_pairs" => singular_cache.pair_count,
        ),
        "summary" => summary,
        "results" => results,
    )
end

function print_compare_summary(payload)
    summary = payload["summary"]
    metadata = payload["metadata"]
    println(@sprintf(
        "Compared %d frequencies on %d/%d faces",
        summary["frequency_count"],
        metadata["element_count"],
        metadata["mesh_faces"],
    ))
    speedup = summary["median_operator_assembly_speedup"]
    speedup !== nothing && println(@sprintf("Median operator assembly speedup: %.3fx", speedup))
    for key in (
        "max_single_layer_relative_frobenius",
        "max_double_layer_relative_frobenius",
        "max_adjoint_double_layer_relative_frobenius",
        "max_hypersingular_relative_frobenius",
        "max_pressure_relative_l2",
        "max_field_relative_l2",
        "max_spl_rms_db",
        "max_spl_p95_db",
        "max_spl_max_db",
    )
        value = summary[key]
        value === nothing && continue
        println(@sprintf("%s: %.6g", key, value))
    end
end

function main(args=ARGS)
    config = parse_compare_args(args)
    payload = compare_payload(config)
    write_json(config.output, payload)
    print_compare_summary(payload)
    println("Wrote $(config.output)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
