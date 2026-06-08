using JSON
using LinearAlgebra
using Printf
using Statistics
using StaticArrays

include(joinpath(@__DIR__, "src", "BeatEngineCore.jl"))
using .BeatEngineCore

LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())

function emit_event(event_type::String; kwargs...)
    payload = Dict{String,Any}("type" => event_type)
    for (key, value) in kwargs
        payload[String(key)] = value
    end
    println(JSON.json(payload))
    flush(stdout)
end

function fail!(message::AbstractString)
    emit_event("failed"; error=String(message))
    exit(1)
end

function parse_args(args)
    request_path = nothing
    worker_mode = false
    i = 1
    while i <= length(args)
        if args[i] == "--request" && i < length(args)
            request_path = args[i + 1]
            i += 2
        elseif args[i] == "--worker"
            worker_mode = true
            i += 1
        else
            fail!("Unknown argument: $(args[i])")
        end
    end
    return request_path, worker_mode
end

get_value(raw, key::String, default=nothing) = haskey(raw, key) ? raw[key] : default

function as_float_vector(raw)
    return [Float64(value) for value in raw]
end

function validate_crossover_config(owner_name::String, crossover)
    crossover === nothing && return
    crossover_type = lowercase(String(get_value(crossover, "type", "none")))
    crossover_type in ("none", "lowpass", "highpass") || error("$(owner_name) crossover type must be none, lowpass, or highpass.")
    crossover_type == "none" && return

    frequency = get_value(crossover, "frequency_hz", nothing)
    frequency === nothing && error("$(owner_name) crossover frequency_hz must be > 0.")
    Float64(frequency) > 0.0 || error("$(owner_name) crossover frequency_hz must be > 0.")

    filter_name = lowercase(String(get_value(crossover, "filter", "butterworth")))
    filter_name in ("butterworth", "linkwitz_riley") || error("$(owner_name) crossover filter must be butterworth or linkwitz_riley.")

    order = Int(get_value(crossover, "order", 1))
    order in (1, 2, 4, 6) || error("$(owner_name) crossover order must be 1, 2, 4, or 6.")
    if filter_name == "linkwitz_riley" && !(order in (2, 4, 6))
        error("$(owner_name) Linkwitz-Riley order must be 2, 4, or 6.")
    end
end

function symmetry_mode_from_config(config)
    mode = lowercase(strip(String(get_value(config, "symmetry", "off"))))
    mode in ("off", "x", "xy") || error("Unsupported symmetry mode: $(mode). Expected off, x, or xy.")
    return mode
end

function beat_backend_from_request(request)
    backend = lowercase(strip(String(get_value(request, "beat_engine_backend", "cuda"))))
    aliases = Dict(
        "beat_cuda" => "cuda",
        "gpu" => "cuda",
        "julia_local" => "cuda",
        "local_julia" => "cuda",
        "afterburner" => "cuda",
        "beat_cpu" => "cpu",
    )
    backend = get(aliases, backend, backend)
    backend in ("cuda", "cpu") || error("Unsupported BEAT Engine backend: $(backend). Expected cuda or cpu.")
    return Symbol(backend)
end

function regular_quadrature_mode_from_config(config, beat_backend::Symbol)
    default_mode = beat_backend == :cpu ? "wavelength" : "fixed"
    mode = lowercase(strip(String(get_value(config, "regular_quadrature_mode", get_value(config, "quadrature_mode", default_mode)))))
    mode in ("fixed", "wavelength") || error("Unsupported regular quadrature mode: $(mode). Expected fixed or wavelength.")
    if beat_backend != :cpu && mode == "wavelength"
        error("Wavelength-driven regular quadrature is currently implemented only for the BEAT CPU backend.")
    end
    return mode
end

function mesh_area_statistic(areas, stat::String)
    values = collect(Float64.(areas))
    isempty(values) && error("Cannot select wavelength quadrature order from an empty mesh.")
    if stat == "median"
        return median(values)
    elseif stat == "p75"
        return quantile(values, 0.75)
    elseif stat == "p90"
        return quantile(values, 0.90)
    elseif stat == "max"
        return maximum(values)
    end
    error("Unsupported wavelength mesh stat: $(stat). Expected median, p75, p90, or max.")
end

function regular_quadrature_selection(config, mesh::BoundaryMesh{T}, freq::T, sound_speed::T, base_order::Int, mode::String) where {T<:AbstractFloat}
    if mode == "fixed"
        return (
            order=base_order,
            mesh_area_stat=nothing,
            element_length_m=nothing,
            kh=nothing,
            q1_cutoff=nothing,
            q2_cutoff=nothing,
            mesh_stat=nothing,
        )
    end

    mesh_stat = lowercase(strip(String(get_value(config, "wavelength_mesh_stat", "p90"))))
    area_stat = mesh_area_statistic(mesh.areas, mesh_stat)
    element_length = sqrt(area_stat)
    k = Float64(2pi * freq / sound_speed)
    kh = k * element_length
    q1_cutoff = Float64(get_value(config, "wavelength_kh_q1_max", 0.0))
    q2_cutoff = Float64(get_value(config, "wavelength_kh_q2_max", 2.0))
    q1_cutoff >= 0.0 || error("wavelength_kh_q1_max must be non-negative.")
    q2_cutoff > q1_cutoff || error("wavelength_kh_q2_max must be greater than wavelength_kh_q1_max.")
    order = kh <= q1_cutoff ? 1 : kh <= q2_cutoff ? 2 : base_order
    return (
        order=order,
        mesh_area_stat=area_stat,
        element_length_m=element_length,
        kh=kh,
        q1_cutoff=q1_cutoff,
        q2_cutoff=q2_cutoff,
        mesh_stat=mesh_stat,
    )
end

function mesh_inputs_from_config(config)
    meshes = get_value(config, "meshes", Any[])
    if !isempty(meshes)
        inputs = NamedTuple[]
        seen_names = Set{String}()
        for (index, mesh) in enumerate(meshes)
            name = String(get_value(mesh, "name", "mesh_$(index)"))
            name in seen_names && error("Duplicate mesh name: $(name)")
            push!(seen_names, name)
            translation = get_value(mesh, "translation_m", [0.0, 0.0, 0.0])
            length(translation) == 3 || error("Mesh '$(name)' translation_m must contain three values.")
            push!(inputs, (
                path=String(mesh["file"]),
                scale=Float64(get_value(mesh, "scale_factor", get_value(config, "scale_factor", 0.001))),
                name=name,
                translation=(Float64(translation[1]), Float64(translation[2]), Float64(translation[3])),
            ))
        end
        return inputs
    end

    return [(
        path=String(config["mesh_file"]),
        scale=Float64(get_value(config, "scale_factor", 0.001)),
        name="mesh",
        translation=(0.0, 0.0, 0.0),
    )]
end

function mesh_name_to_id(mesh_inputs)
    return Dict(input.name => index for (index, input) in enumerate(mesh_inputs))
end

function radiator_inputs_from_config(config, mesh_inputs)
    raw_radiators = get_value(config, "radiators", Any[])
    mesh_lookup = mesh_name_to_id(mesh_inputs)
    if isempty(raw_radiators)
        first_mesh = mesh_inputs[1]
        return [Dict{String,Any}(
            "name" => "Radiator",
            "tag" => Int(get_value(config, "tag_throat", 2)),
            "mesh" => first_mesh.name,
            "mesh_id" => 1,
            "channel" => "main",
            "velocity_offset_db" => 0.0,
            "level_db" => 0.0,
            "polarity" => 1,
            "delay_ms" => 0.0,
            "hpf" => Dict("type" => "none"),
            "lpf" => Dict("type" => "none"),
        )]
    end

    radiators = Dict{String,Any}[]
    for radiator in raw_radiators
        radiator_name = String(get_value(radiator, "name", "Radiator"))
        raw_mesh = get_value(radiator, "mesh", nothing)
        if raw_mesh === nothing
            length(mesh_inputs) == 1 || error("Radiator '$(radiator_name)' must specify 'mesh' when multiple meshes are configured.")
            mesh_name = mesh_inputs[1].name
        else
            mesh_name = String(raw_mesh)
        end
        haskey(mesh_lookup, mesh_name) || error("Radiator '$(radiator_name)' references unknown mesh '$(mesh_name)'.")
        validate_crossover_config(radiator_name, get_value(radiator, "hpf", nothing))
        validate_crossover_config(radiator_name, get_value(radiator, "lpf", nothing))
        push!(radiators, Dict{String,Any}(
            "name" => radiator_name,
            "tag" => Int(radiator["tag"]),
            "mesh" => mesh_name,
            "mesh_id" => mesh_lookup[mesh_name],
            "channel" => String(get_value(radiator, "channel", "main")),
            "velocity_offset_db" => Float64(get_value(radiator, "velocity_offset_db", 0.0)),
            "level_db" => Float64(get_value(radiator, "level_db", 0.0)),
            "polarity" => Int(get_value(radiator, "polarity", 1)),
            "delay_ms" => Float64(get_value(radiator, "delay_ms", 0.0)),
            "hpf" => get_value(radiator, "hpf", Dict("type" => "none")),
            "lpf" => get_value(radiator, "lpf", Dict("type" => "none")),
        ))
    end
    return radiators
end

function channel_inputs_from_config(config)
    channels = Dict{String,Any}()
    for channel in get_value(config, "channels", Any[])
        name = String(channel["name"])
        validate_crossover_config("channel $(name)", get_value(channel, "hpf", nothing))
        validate_crossover_config("channel $(name)", get_value(channel, "lpf", nothing))
        channels[name] = Dict{String,Any}(
            "level_db" => Float64(get_value(channel, "level_db", 0.0)),
            "polarity" => Int(get_value(channel, "polarity", 1)),
            "delay_ms" => Float64(get_value(channel, "delay_ms", 0.0)),
            "hpf" => get_value(channel, "hpf", Dict("type" => "none")),
            "lpf" => get_value(channel, "lpf", Dict("type" => "none")),
        )
    end
    return channels
end

function load_combined_mesh(mesh_inputs, ::Type{T}) where {T<:AbstractFloat}
    vertices = SVector{3,T}[]
    faces = NTuple{3,Int}[]
    physical_tags = Int[]
    element_mesh_ids = Int[]

    for (mesh_id, input) in enumerate(mesh_inputs)
        mesh = load_gmsh22_with_tags(input.path, T(input.scale))
        translation = SVector{3,T}(T(input.translation[1]), T(input.translation[2]), T(input.translation[3]))
        vertex_offset = length(vertices)
        append!(vertices, [vertex + translation for vertex in mesh.vertices])
        append!(faces, [(face[1] + vertex_offset, face[2] + vertex_offset, face[3] + vertex_offset) for face in mesh.faces])
        append!(physical_tags, mesh.physical_tags)
        append!(element_mesh_ids, fill(mesh_id, length(mesh.faces)))
    end

    return BoundaryMesh(vertices, faces, physical_tags), element_mesh_ids
end

function butterworth_poles(order::Int, ::Type{T}) where {T<:AbstractFloat}
    return Complex{T}[
        exp(Complex{T}(0, T(pi) / T(2) + T(2 * k - 1) * T(pi) / T(2 * order)))
        for k in 1:order
    ]
end

function butterworth_response(crossover_type::String, order::Int, cutoff_hz, freq::T) where {T<:AbstractFloat}
    cutoff = T(cutoff_hz)
    omega = T(2pi) * freq
    omega_c = T(2pi) * cutoff
    s = Complex{T}(0, omega)
    response = one(Complex{T})

    for pole in butterworth_poles(order, T)
        scaled_pole = crossover_type == "lowpass" ? omega_c * pole : omega_c / pole
        response *= crossover_type == "lowpass" ? (-scaled_pole) / (s - scaled_pole) : s / (s - scaled_pole)
    end

    return response
end

function crossover_response(crossover, freq::T) where {T<:AbstractFloat}
    crossover === nothing && return one(Complex{T})
    crossover_type = lowercase(String(get_value(crossover, "type", "none")))
    crossover_type == "none" && return one(Complex{T})

    filter_name = lowercase(String(get_value(crossover, "filter", "butterworth")))
    order = Int(get_value(crossover, "order", 1))
    cutoff_hz = get_value(crossover, "frequency_hz", nothing)
    cutoff_hz === nothing && error("Crossover frequency_hz must be set for $(crossover_type).")

    if filter_name == "linkwitz_riley"
        section = butterworth_response(crossover_type, div(order, 2), cutoff_hz, freq)
        return section * section
    end

    return butterworth_response(crossover_type, order, cutoff_hz, freq)
end

function channel_drive(channel, freq::T) where {T<:AbstractFloat}
    omega = T(2pi) * freq
    level = T(10.0) ^ (T(channel["level_db"]) / T(20.0))
    delay = exp(Complex{T}(0, -omega * T(channel["delay_ms"]) / T(1000.0)))
    crossover = crossover_response(get_value(channel, "hpf", nothing), freq) *
        crossover_response(get_value(channel, "lpf", nothing), freq)
    return Complex{T}(T(channel["polarity"]) * level) * delay * crossover
end

function polar_observation_points(config, ::Type{T}) where {T<:AbstractFloat}
    step = T(get_value(config, "step_size", 5.0))
    angle_min = T(get_value(config, "min_angle", -180.0))
    angle_max = T(get_value(config, "max_angle", 180.0))
    step <= 0 && error("step_size must be positive.")
    angle_min < -180 && error("polar angle range must stay within [-180, 180] degrees.")
    angle_max > 180 && error("polar angle range must stay within [-180, 180] degrees.")
    angle_max < angle_min && error("max_angle must be >= min_angle.")
    !(angle_min <= 0 <= angle_max) && error("polar angle range must include 0 degrees.")

    angles = collect(Float32.(range(Float64(angle_min), stop=Float64(angle_max), step=Float64(step))))
    if isempty(angles) || angles[end] < Float32(angle_max)
        push!(angles, Float32(angle_max))
    end
    angles = Float32.(clamp.(angles, Float32(angle_min), Float32(angle_max)))

    distance = T(get_value(config, "distance", 2.0))
    distance <= 0 && error("distance must be positive.")
    axial_offset = T(get_value(config, "axial_offset", 0.0))
    horizontal = SVector{3,T}[]
    vertical = SVector{3,T}[]
    for angle_deg in angles
        angle = T(pi) * T(angle_deg) / T(180.0)
        push!(horizontal, SVector{3,T}(distance * sin(angle), T(0.0), distance * cos(angle) + axial_offset))
        push!(vertical, SVector{3,T}(T(0.0), distance * sin(angle), distance * cos(angle) + axial_offset))
    end
    on_axis_idx = argmin(abs.(Float64.(angles)))
    return angles, horizontal, vertical, on_axis_idx
end

function spherical_observation(config, ::Type{T}) where {T<:AbstractFloat}
    enabled = Bool(get_value(config, "spherical_sampling_enabled", false))
    if !enabled
        return nothing
    end

    point_count = Int(get_value(config, "spherical_sampling_points", 6000))
    point_count <= 0 && error("spherical_sampling_points must be positive.")
    distance = T(get_value(config, "distance", 2.0))
    distance <= 0 && error("distance must be positive.")
    axial_offset = T(get_value(config, "axial_offset", 0.0))
    golden_angle = T(pi * (3.0 - sqrt(5.0)))
    points = Vector{SVector{3,T}}(undef, point_count)
    theta = Vector{Float32}(undef, point_count)
    phi = Vector{Float32}(undef, point_count)
    r_distance = fill(Float32(distance), point_count)

    for i in 0:(point_count - 1)
        z_unit = T(1.0 - (2.0 * i + 1.0) / point_count)
        xy_radius = sqrt(max(T(1.0) - z_unit * z_unit, T(0.0)))
        azimuth = T(i) * golden_angle
        x = distance * xy_radius * cos(azimuth)
        y = distance * xy_radius * sin(azimuth)
        z = distance * z_unit + axial_offset
        storage_index = i + 1
        points[storage_index] = SVector{3,T}(x, y, z)
        theta[storage_index] = Float32(acos(clamp(z_unit, T(-1.0), T(1.0))))
        phi[storage_index] = Float32(mod(atan(y, x), T(2pi)))
    end

    return (
        points=points,
        metadata=Dict(
            "r_distance_m" => r_distance,
            "theta_polar_rad" => theta,
            "phi_azimuth_rad" => phi,
        ),
    )
end

function drive_for_radiator(radiator, channels, freq::T) where {T<:AbstractFloat}
    omega = T(2pi) * freq
    channel_name = String(get_value(radiator, "channel", "main"))
    channel = get(channels, channel_name, nothing)
    if channel !== nothing
        return channel_drive(channel, freq) * T(10.0) ^ (T(radiator["velocity_offset_db"]) / T(20.0))
    end

    level_db = T(radiator["level_db"] + radiator["velocity_offset_db"])
    polarity = T(radiator["polarity"])
    delay_ms = T(radiator["delay_ms"])
    level = T(10.0) ^ (level_db / T(20.0))
    delay = exp(Complex{T}(0, -omega * delay_ms / T(1000.0)))
    crossover = crossover_response(get_value(radiator, "hpf", nothing), freq) *
        crossover_response(get_value(radiator, "lpf", nothing), freq)
    return Complex{T}(polarity * level) * delay * crossover
end

function radiator_owns_element(radiator, element_mesh_ids, element_index::Int)
    return element_mesh_ids[element_index] == Int(radiator["mesh_id"])
end

function validate_radiator_channels(radiators, channels)
    isempty(channels) && return
    for radiator in radiators
        channel_name = String(get_value(radiator, "channel", "main"))
        haskey(channels, channel_name) || error("Radiator '$(radiator["name"])' references unknown channel '$(channel_name)'.")
    end
end

function validate_radiator_elements(mesh, element_mesh_ids, radiators)
    for radiator in radiators
        tag = Int(radiator["tag"])
        found = false
        for element_index in eachindex(mesh.physical_tags)
            if mesh.physical_tags[element_index] == tag && radiator_owns_element(radiator, element_mesh_ids, element_index)
                found = true
                break
            end
        end
        found || error("No elements found for radiator '$(radiator["name"])' tag=$(tag) on mesh '$(radiator["mesh"])'.")
    end
end

function pressure_for_drives(mesh, element_mesh_ids, operators, identity_p1_p1, identity_p1_dp0, radiators, drives, rho, omega, k; cpu_solve_system=nothing)
    ComplexType = eltype(drives)
    q_neumann = zeros(ComplexType, length(mesh.faces))
    for (radiator_index, radiator) in enumerate(radiators)
        tag = Int(radiator["tag"])
        drive = drives[radiator_index]
        for element_index in eachindex(mesh.physical_tags)
            if mesh.physical_tags[element_index] == tag && radiator_owns_element(radiator, element_mesh_ids, element_index)
                q_neumann[element_index] = ComplexType(0, rho * omega) * drive
            end
        end
    end
    pressure = cpu_solve_system === nothing ?
        solve_burton_miller_neumann(operators, identity_p1_p1, identity_p1_dp0, q_neumann, k) :
        solve_burton_miller_neumann_cpu_system(cpu_solve_system, q_neumann, typeof(k))
    return pressure, q_neumann
end

function field_for_points(points, mesh, pressure, q_neumann, k, field_cache, beat_backend::Symbol)
    if beat_backend == :cuda
        return evaluate_galerkin_field_cuda(points, mesh, pressure, q_neumann, k, field_cache)
    elseif beat_backend == :cpu
        return evaluate_galerkin_field_cpu(points, mesh, pressure, q_neumann, k, field_cache)
    end
    error("Unsupported BEAT Engine backend: $(beat_backend).")
end

function spl_for_points(points, mesh, pressure, q_neumann, k, field_cache, beat_backend::Symbol, ::Type{T}) where {T<:AbstractFloat}
    pot = field_for_points(points, mesh, pressure, q_neumann, k, field_cache, beat_backend)
    return Float32.(T(20.0) .* log10.(abs.(pot) ./ T(20e-6)))
end

function pressure_to_spl(pressure, ::Type{T}) where {T<:AbstractFloat}
    return Float32.(T(20.0) .* log10.(abs.(pressure) ./ T(20e-6)))
end

function complex_rows_to_wire(rows)
    return Dict(
        "real" => [Float32.(real.(row)) for row in rows],
        "imag" => [Float32.(imag.(row)) for row in rows],
    )
end

function interpolate_complex_reference(values, angles_deg, reference_angle_deg, ::Type{T}) where {T<:AbstractFloat}
    isempty(values) && return Complex{T}(0, 0)
    length(values) == 1 && return Complex{T}(values[1])

    reference_wrapped = mod(T(reference_angle_deg) + T(180.0), T(360.0)) - T(180.0)
    angle_values = T.(angles_deg)
    value_values = Complex{T}.(values)
    if isapprox(angle_values[1], T(-180.0)) && isapprox(angle_values[end], T(180.0))
        angle_values = angle_values[1:(end - 1)]
        value_values = value_values[1:(end - 1)]
    end

    extended_angles = vcat(angle_values .- T(360.0), angle_values, angle_values .+ T(360.0))
    extended_values = vcat(value_values, value_values, value_values)
    for idx in 1:(length(extended_angles) - 1)
        left = extended_angles[idx]
        right = extended_angles[idx + 1]
        if reference_wrapped >= left && reference_wrapped <= right
            if isapprox(left, right)
                return extended_values[idx]
            end
            fraction = (reference_wrapped - left) / (right - left)
            return extended_values[idx] * (one(T) - fraction) + extended_values[idx + 1] * fraction
        end
    end
    return extended_values[argmin(abs.(extended_angles .- reference_wrapped))]
end

function flat_target_corrections(channel_names, horizontal_pressure_rows, angles_deg, reference_angle_deg, flat_target::Bool, ::Type{T}) where {T<:AbstractFloat}
    corrections = Dict{String,T}()
    for (channel_index, channel_name) in enumerate(channel_names)
        if !flat_target
            corrections[channel_name] = one(T)
            continue
        end
        reference_pressure = interpolate_complex_reference(
            horizontal_pressure_rows[channel_index],
            angles_deg,
            reference_angle_deg,
            T,
        )
        magnitude = abs(reference_pressure)
        corrections[channel_name] = magnitude <= T(1e-12) ? one(T) : one(T) / magnitude
    end
    return corrections
end

function synthesize_channel_basis(channel_names, horizontal_pressure_rows, vertical_pressure_rows, sphere_pressure_rows, channels, freq, angles_deg, reference_angle_deg, flat_target::Bool, ::Type{T}) where {T<:AbstractFloat}
    corrections = flat_target_corrections(channel_names, horizontal_pressure_rows, angles_deg, reference_angle_deg, flat_target, T)
    weights = Complex{T}[
        channel_drive(get(channels, channel_name, Dict(
            "level_db" => 0.0,
            "polarity" => 1,
            "delay_ms" => 0.0,
            "hpf" => Dict("type" => "none"),
            "lpf" => Dict("type" => "none"),
        )), freq) * corrections[channel_name]
        for channel_name in channel_names
    ]

    horizontal_pressure = zero.(horizontal_pressure_rows[1])
    vertical_pressure = zero.(vertical_pressure_rows[1])
    for channel_index in eachindex(channel_names)
        horizontal_pressure .+= horizontal_pressure_rows[channel_index] .* weights[channel_index]
        vertical_pressure .+= vertical_pressure_rows[channel_index] .* weights[channel_index]
    end

    horizontal_spl = pressure_to_spl(horizontal_pressure, T)
    vertical_spl = pressure_to_spl(vertical_pressure, T)
    on_axis_idx = argmin(abs.(Float64.(angles_deg)))
    reference = horizontal_spl[on_axis_idx]
    sphere_norm = nothing
    if sphere_pressure_rows !== nothing
        sphere_pressure = zero.(sphere_pressure_rows[1])
        for channel_index in eachindex(channel_names)
            sphere_pressure .+= sphere_pressure_rows[channel_index] .* weights[channel_index]
        end
        sphere_norm = Float32.(pressure_to_spl(sphere_pressure, T) .- reference)
    end

    return (
        horizontal_spl=horizontal_spl,
        vertical_spl=vertical_spl,
        horizontal_norm=Float32.(horizontal_spl .- reference),
        vertical_norm=Float32.(vertical_spl .- reference),
        sphere_norm=sphere_norm,
        corrections=corrections,
        weights=weights,
    )
end

function channel_unit_drives(radiators, channel_name::String, ::Type{T}) where {T<:AbstractFloat}
    return Complex{T}[
        String(get_value(radiator, "channel", "main")) == channel_name ?
        Complex{T}(T(10.0) ^ (T(radiator["velocity_offset_db"]) / T(20.0)), 0) :
        Complex{T}(0, 0)
        for radiator in radiators
    ]
end

function radiator_drives_from_channel_basis(radiators, channels, freq, corrections, ::Type{T}) where {T<:AbstractFloat}
    return Complex{T}[
        drive_for_radiator(radiator, channels, freq) *
        get(corrections, String(get_value(radiator, "channel", "main")), one(T))
        for radiator in radiators
    ]
end

function impedance_for_radiators(mesh, element_mesh_ids, pressure, radiators, drives, ::Type{T}; symmetry_mode::Symbol=:off) where {T<:AbstractFloat}
    force_scale = eltype(pressure)(symmetry_reduction_factor(symmetry_mode))
    impedance = Vector{Vector{Float32}}()
    for (radiator_index, radiator) in enumerate(radiators)
        drive = drives[radiator_index]
        if abs(drive) <= T(0.0)
            push!(impedance, [Float32(NaN), Float32(NaN)])
            continue
        end

        total_force = zero(eltype(pressure))
        tag = Int(radiator["tag"])
        for element_index in eachindex(mesh.physical_tags)
            mesh.physical_tags[element_index] == tag || continue
            radiator_owns_element(radiator, element_mesh_ids, element_index) || continue
            face = mesh.faces[element_index]
            p_avg = (pressure[face[1]] + pressure[face[2]] + pressure[face[3]]) / eltype(pressure)(3.0)
            total_force += p_avg * eltype(pressure)(mesh.areas[element_index]) * force_scale
        end
        total_force *= eltype(pressure)(10.0)
        z_complex = total_force / drive
        push!(impedance, [Float32(real(z_complex) / 2), Float32(-imag(z_complex) / 2)])
    end
    return impedance
end

function solve_request(request)
    try
        solve_request_impl(request)
    finally
        cleanup_cuda_after_solve!()
    end
end

function cleanup_cuda_after_solve!()
    cuda = BeatEngineCore.CUDA_MODULE
    if cuda !== nothing
        try
            cuda.functional() && cuda.synchronize()
        catch
        end
    end

    GC.gc(true)

    if cuda !== nothing
        try
            if isdefined(cuda, :reclaim)
                cuda.reclaim()
            end
        catch
        end
    end

    GC.gc(true)
    return nothing
end

function solve_request_impl(request)
    schema_version = Int(get_value(request, "schema_version", 1))
    schema_version in (1, 2) || error("Unsupported solve request schema_version $(schema_version).")

    config = request["config"]
    symmetry_mode = symmetry_mode_from_config(config)
    beat_backend = beat_backend_from_request(request)
    frequencies = Float32.(request["frequencies_hz"])
    isempty(frequencies) && error("frequencies_hz must contain at least one frequency.")
    cancel_path = get_value(request, "cancel_path", nothing)

    FloatType = Float32
    mesh_inputs = mesh_inputs_from_config(config)
    radiators = radiator_inputs_from_config(config, mesh_inputs)
    channels = channel_inputs_from_config(config)
    validate_radiator_channels(radiators, channels)
    polar_angles_deg, horizontal_points, vertical_points, on_axis_idx = polar_observation_points(config, FloatType)
    sphere = spherical_observation(config, FloatType)
    sphere_metadata = sphere === nothing ? nothing : sphere.metadata

    emit_event(
        "initialized";
        polar_angle_deg=polar_angles_deg,
        radiator_names=[radiator["name"] for radiator in radiators],
        sphere_metadata=sphere_metadata,
    )

    emit_event("status"; message=@sprintf(
        "BEAT Engine loading %d mesh%s with %d thread(s)",
        length(mesh_inputs),
        length(mesh_inputs) == 1 ? "" : "es",
        Threads.nthreads(),
    ))

    mesh, element_mesh_ids = load_combined_mesh(mesh_inputs, FloatType)
    validate_symmetry_fundamental_domain!(mesh, Symbol(symmetry_mode))
    validate_radiator_elements(mesh, element_mesh_ids, radiators)
    p1_space = build_p1_space(mesh)
    dp0_space = build_dp0_space(mesh)
    base_regular_order = Int(get_value(config, "quadrature_order", 4))
    regular_quadrature_mode = regular_quadrature_mode_from_config(config, beat_backend)
    rule = triangle_rule(FloatType, base_regular_order)
    cpu_field_cache = build_field_evaluation_cache(mesh, rule; symmetry_mode=Symbol(symmetry_mode))
    singular_order = Int(get_value(config, "singular_order", 4))
    identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1_space, dp0_space, rule, :p1, :p1; symmetry_mode=Symbol(symmetry_mode))
    identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1_space, dp0_space, rule, :p1, :dp0; symmetry_mode=Symbol(symmetry_mode))
    rho = FloatType(get_value(config, "rho", 1.21))
    sound_speed = FloatType(get_value(config, "sound_speed", 343.0))
    flat_target = Bool(get_value(config, "flat_target_normalization_enabled", true))
    flat_target_reference_angle_deg = FloatType(get_value(config, "flat_target_reference_angle_deg", 0.0))
    channel_names = sort(unique([String(get_value(radiator, "channel", "main")) for radiator in radiators]))
    singular_cache = build_singular_correction_cache(mesh, singular_order)
    cuda_cache = nothing
    cuda_singular_cache = nothing
    field_cache = cpu_field_cache
    regular_rule_cache = Dict{Int,Any}(base_regular_order => rule)
    identity_cache = Dict{Int,Any}(base_regular_order => (identity_p1_p1, identity_p1_dp0))
    cpu_field_cache_by_order = Dict{Int,Any}(base_regular_order => cpu_field_cache)
    if beat_backend == :cuda
        emit_event("status"; message="BEAT Engine using CUDA balanced split assembly, GPU dense solve, and GPU field evaluation")
        cuda_cache = build_cuda_regular_assembly_cache(mesh, rule)
        field_cache = build_cuda_field_evaluation_cache(cpu_field_cache)
        cuda_singular_cache = BeatEngineCore.build_cuda_singular_correction_cache(singular_cache, p1_space, dp0_space)
    else
        emit_event("status"; message="BEAT Engine using CPU assembly, OpenBLAS/LAPACK dense solve, CPU field evaluation, and $(regular_quadrature_mode) regular quadrature")
    end

    for (index, freq_raw) in enumerate(frequencies)
        if cancel_path !== nothing && isfile(String(cancel_path))
            emit_event("cancelled"; solved_count=index - 1)
            return
        end

        freq = FloatType(freq_raw)
        omega = FloatType(2pi) * freq
        k = omega / sound_speed
        quadrature_selection = regular_quadrature_selection(config, mesh, freq, sound_speed, base_regular_order, regular_quadrature_mode)
        selected_rule = if beat_backend == :cpu
            get!(regular_rule_cache, quadrature_selection.order) do
                triangle_rule(FloatType, quadrature_selection.order)
            end
        else
            rule
        end
        selected_identity_p1_p1 = identity_p1_p1
        selected_identity_p1_dp0 = identity_p1_dp0
        selected_field_cache = field_cache
        if beat_backend == :cpu
            selected_identity = get!(identity_cache, quadrature_selection.order) do
                (
                    assemble_l2_identity_matrix(mesh, p1_space, dp0_space, selected_rule, :p1, :p1; symmetry_mode=Symbol(symmetry_mode)),
                    assemble_l2_identity_matrix(mesh, p1_space, dp0_space, selected_rule, :p1, :dp0; symmetry_mode=Symbol(symmetry_mode)),
                )
            end
            selected_identity_p1_p1 = selected_identity[1]
            selected_identity_p1_dp0 = selected_identity[2]
            selected_field_cache = get!(cpu_field_cache_by_order, quadrature_selection.order) do
                build_field_evaluation_cache(mesh, selected_rule; symmetry_mode=Symbol(symmetry_mode))
            end
        end

        t_assembly = @elapsed begin
            operators = assemble_regular_galerkin_operators(
                mesh,
                p1_space,
                dp0_space,
                k,
                selected_rule;
                skip_singular=false,
                singular_order=singular_order,
                use_cuda_regular=beat_backend == :cuda,
                cuda_cache=cuda_cache,
                return_gpu=beat_backend == :cuda,
                parallel_quadrature=beat_backend == :cuda,
                singular_cache=singular_cache,
                cuda_singular_cache=cuda_singular_cache,
                regular_assembly_mode=:split_atomic_balanced,
                symmetry_mode=Symbol(symmetry_mode),
            )
        end

        t_solve = 0.0
        t_field = 0.0
        cpu_solve_system = nothing
        if beat_backend == :cpu
            t_solve += @elapsed begin
                cpu_solve_system = build_burton_miller_neumann_cpu_system(operators, selected_identity_p1_p1, selected_identity_p1_dp0, k)
            end
        end
        channel_boundary_pressures = Vector{Vector{Complex{FloatType}}}()
        horizontal_pressure_rows = Vector{Vector{Complex{FloatType}}}()
        vertical_pressure_rows = Vector{Vector{Complex{FloatType}}}()
        sphere_pressure_rows = sphere === nothing ? nothing : Vector{Vector{Complex{FloatType}}}()

        combined_points = sphere === nothing ? vcat(horizontal_points, vertical_points) : vcat(horizontal_points, vertical_points, sphere.points)
        horizontal_count = length(horizontal_points)
        vertical_count = length(vertical_points)

        for channel_name in channel_names
            unit_drives = channel_unit_drives(radiators, channel_name, FloatType)
            pressure = nothing
            q_neumann = nothing
            t_solve += @elapsed begin
                pressure, q_neumann = pressure_for_drives(
                    mesh,
                    element_mesh_ids,
                    operators,
                    selected_identity_p1_p1,
                    selected_identity_p1_dp0,
                    radiators,
                    unit_drives,
                    rho,
                    omega,
                    k,
                    cpu_solve_system=cpu_solve_system,
                )
            end
            t_field += @elapsed begin
                combined_pressure = field_for_points(combined_points, mesh, pressure, q_neumann, k, selected_field_cache, beat_backend)
                push!(horizontal_pressure_rows, Complex{FloatType}.(combined_pressure[1:horizontal_count]))
                push!(vertical_pressure_rows, Complex{FloatType}.(combined_pressure[(horizontal_count + 1):(horizontal_count + vertical_count)]))
                if sphere !== nothing
                    sphere_start = horizontal_count + vertical_count + 1
                    push!(sphere_pressure_rows, Complex{FloatType}.(combined_pressure[sphere_start:end]))
                end
                push!(channel_boundary_pressures, Complex{FloatType}.(pressure))
            end
        end

        horizontal_spl = Float32[]
        vertical_spl = Float32[]
        horizontal_norm = Float32[]
        vertical_norm = Float32[]
        impedance = Vector{Vector{Float32}}()
        sphere_norm = nothing
        synthesis = nothing
        drives = Complex{FloatType}[]
        mixed_boundary_pressure = zeros(Complex{FloatType}, length(mesh.vertices))
        t_field += @elapsed begin
            synthesis = synthesize_channel_basis(
                channel_names,
                horizontal_pressure_rows,
                vertical_pressure_rows,
                sphere_pressure_rows,
                channels,
                freq,
                polar_angles_deg,
                flat_target_reference_angle_deg,
                flat_target,
                FloatType,
            )
            horizontal_spl = synthesis.horizontal_spl
            vertical_spl = synthesis.vertical_spl
            horizontal_norm = synthesis.horizontal_norm
            vertical_norm = synthesis.vertical_norm
            sphere_norm = synthesis.sphere_norm
            drives = radiator_drives_from_channel_basis(radiators, channels, freq, synthesis.corrections, FloatType)
            for channel_index in eachindex(channel_names)
                mixed_boundary_pressure .+= channel_boundary_pressures[channel_index] .* synthesis.weights[channel_index]
            end
            impedance = impedance_for_radiators(mesh, element_mesh_ids, mixed_boundary_pressure, radiators, drives, FloatType; symmetry_mode=Symbol(symmetry_mode))
        end

        release_operator_storage!(operators)
        emit_event(
            "result";
            solved_count=index,
            total_count=length(frequencies),
            result=Dict(
                "freq_hz" => Float32(freq),
                "horizontal_spl_norm_db" => horizontal_norm,
                "vertical_spl_norm_db" => vertical_norm,
                "impedance" => impedance,
                "horizontal_spl_db" => horizontal_spl,
                "vertical_spl_db" => vertical_spl,
                "sphere_spl_norm_db" => sphere_norm,
                "channel_names" => channel_names,
                "horizontal_pressure" => complex_rows_to_wire(horizontal_pressure_rows),
                "vertical_pressure" => complex_rows_to_wire(vertical_pressure_rows),
                "sphere_pressure" => sphere_pressure_rows === nothing ? nothing : complex_rows_to_wire(sphere_pressure_rows),
                "timings" => Dict(
                    "assembly_s" => Float32(t_assembly),
                    "solve_s" => Float32(t_solve),
                    "field_s" => Float32(t_field),
                ),
                "diagnostics" => Dict(
                    "convergence_info" => 0,
                    "message" => "Julia direct dense solve",
                    "backend" => String(beat_backend),
                    "symmetry" => symmetry_mode,
                    "regular_quadrature_mode" => regular_quadrature_mode,
                    "regular_quadrature_order" => quadrature_selection.order,
                    "regular_quadrature_base_order" => base_regular_order,
                    "regular_quadrature_wavelength_mesh_stat" => quadrature_selection.mesh_stat,
                    "regular_quadrature_wavelength_mesh_area_stat_m2" => quadrature_selection.mesh_area_stat,
                    "regular_quadrature_wavelength_element_length_m" => quadrature_selection.element_length_m,
                    "regular_quadrature_wavelength_kh" => quadrature_selection.kh,
                    "regular_quadrature_wavelength_kh_q1_max" => quadrature_selection.q1_cutoff,
                    "regular_quadrature_wavelength_kh_q2_max" => quadrature_selection.q2_cutoff,
                ),
            ),
        )
    end

    emit_event("completed"; solved_count=length(frequencies))
end

function worker_loop()
    emit_event("ready"; protocol="boundary_lab_julia_worker", pid=getpid())
    for line in eachline(stdin)
        text = strip(line)
        isempty(text) && continue
        try
            message = JSON.parse(text)
            request_path = String(message["request"])
            request = JSON.parsefile(request_path)
            solve_request(request)
        catch exc
            emit_event("failed"; error=sprint(showerror, exc))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        request_path, worker_mode = parse_args(ARGS)
        if worker_mode
            worker_loop()
        else
            isnothing(request_path) && fail!("Missing --request path.")
            request = JSON.parsefile(request_path)
            solve_request(request)
        end
    catch exc
        fail!(sprint(showerror, exc))
    end
end
