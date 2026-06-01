include(joinpath(@__DIR__, "benchmark_solver.jl"))

function main(args=ARGS)
    profile_args = isempty(args) ? [
        "--subset-faces", "16",
        "--quadrature-order", "2",
        "--singular-order", "2",
        "--skip-solve",
        "--skip-field",
        "--profile", "cpu",
        "--json", joinpath(@__DIR__, "..", "results", "profile_sample.json"),
    ] : args

    if !("--profile" in profile_args)
        profile_args = vcat(profile_args, ["--profile", "cpu"])
    end

    config = parse_args(profile_args)
    payload = benchmark_payload(config)
    write_json(config.output, payload)
    print_summary(payload)
    println("Wrote $(config.output)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
