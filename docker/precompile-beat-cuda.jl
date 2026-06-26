using CUDA
using JSON
using StaticArrays

include(joinpath(@__DIR__, "..", "src", "blab", "solvers", "julia_local", "solver.jl"))

request = Dict{String,Any}(
    "beat_engine_backend" => "cuda",
    "frequencies_hz" => [100.0f0],
    "config" => Dict{String,Any}(
        "mesh_file" => "warmup.msh",
        "scale_factor" => 1.0,
        "distance" => 1.0,
        "step_size" => 90.0,
        "min_angle" => 0.0,
        "max_angle" => 0.0,
        "tag_throat" => 2,
        "radiators" => [Dict{String,Any}(
            "name" => "warmup",
            "tag" => 2,
            "channel" => "main",
            "hpf" => Dict{String,Any}("type" => "none"),
            "lpf" => Dict{String,Any}("type" => "none"),
        )],
        "channels" => [Dict{String,Any}(
            "name" => "main",
            "hpf" => Dict{String,Any}("type" => "none"),
            "lpf" => Dict{String,Any}("type" => "none"),
        )],
    ),
)

beat_backend_from_request(request)
mesh_inputs = mesh_inputs_from_config(request["config"])
radiator_inputs_from_config(request["config"], mesh_inputs)
channel_inputs_from_config(request["config"])
polar_observation_points(request["config"], Float32)
spherical_observation(request["config"], Float32)
