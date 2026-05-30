include(joinpath(@__DIR__, "..", "solver.jl"))

config = SimulationConfig(single_frequency=1000.0, eval_point_count=12)
freqs = build_frequency_vector(config, Float64)

println((
    frequencies=freqs,
    eval_point_count=config.eval_point_count,
    precision=config.precision,
))
