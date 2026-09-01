const DEPLOY_BOUNDARY_STATE = Ref{Any}(nothing)
const DEPLOY_GEOMETRY_STATE = Ref{Any}(nothing)

function release_deploy_boundary_state!()
    state = DEPLOY_BOUNDARY_STATE[]
    state === nothing && return nothing
    if state.backend == :cuda
        release_cuda_weighted_field_sources!(state.weighted_sources)
        Bool(get(state, :shared_geometry, false)) || release_cuda_field_evaluation_cache!(state.field_cache)
        cuda = BeatEngineCore.CUDA_MODULE
        state.pressure isa cuda.CuArray && cuda.unsafe_free!(state.pressure)
        state.q_neumann isa cuda.CuArray && cuda.unsafe_free!(state.q_neumann)
    end
    DEPLOY_BOUNDARY_STATE[] = nothing
    return nothing
end

function release_deploy_geometry_state!()
    state = DEPLOY_GEOMETRY_STATE[]
    state === nothing && return nothing
    release_deploy_boundary_state!()
    if state.backend == :cuda
        cuda = BeatEngineCore.CUDA_MODULE
        release_cuda_field_evaluation_cache!(state.field_cache)
        state.cuda_identity_cache === nothing || release_cuda_burton_miller_identity_cache!(state.cuda_identity_cache)
        for cache in (
            state.device_cache,
            state.device_singular_cache,
            state.device_image_singular_cache,
            state.device_near_correction_cache,
            state.device_ground_near_correction_cache,
        )
            cache === nothing && continue
            for name in fieldnames(typeof(cache))
                value = getfield(cache, name)
                value isa cuda.CuArray && cuda.unsafe_free!(value)
            end
        end
    end
    DEPLOY_GEOMETRY_STATE[] = nothing
    return nothing
end

function deploy_source_transform(raw, ::Type{T}) where {T<:AbstractFloat}
    position = get_value(raw, "position_m", [0.0, 0.0, 0.0])
    length(position) == 3 || error("Deploy source position_m must contain three values.")
    roll = T(pi) * T(get_value(raw, "roll_deg", 0.0)) / T(180.0)
    pitch = T(pi) * T(get_value(raw, "pitch_deg", 0.0)) / T(180.0)
    yaw = T(pi) * T(get_value(raw, "yaw_deg", 0.0)) / T(180.0)
    return (
        position=SVector{3,T}(T(position[1]), T(position[2]), T(position[3])),
        roll_cosine=cos(roll),
        roll_sine=sin(roll),
        pitch_cosine=cos(pitch),
        pitch_sine=sin(pitch),
        yaw_cosine=cos(yaw),
        yaw_sine=sin(yaw),
    )
end

function deploy_source_transforms(request, ::Type{T}) where {T<:AbstractFloat}
    raw_transforms = get_value(request, "source_transforms", nothing)
    if raw_transforms === nothing
        raw_transforms = [get_value(request, "source_transform", Dict{String,Any}())]
    end
    isempty(raw_transforms) && error("Deploy solve requires at least one source transform.")
    return [deploy_source_transform(raw, T) for raw in raw_transforms]
end

function transform_deploy_mesh(mesh::BoundaryMesh{T}, transform) where {T<:AbstractFloat}
    vertices = SVector{3,T}[]
    sizehint!(vertices, length(mesh.vertices))
    for package_point in mesh.vertices
        # This is the same package-to-scene proper rotation used by the
        # Three.js renderer before applying scene yaw and translation.
        scene_x = package_point[1]
        scene_y = -package_point[3]
        scene_z = package_point[2]
        rolled_x = transform.roll_cosine * scene_x - transform.roll_sine * scene_y
        rolled_y = transform.roll_sine * scene_x + transform.roll_cosine * scene_y
        pitched_y = transform.pitch_cosine * rolled_y - transform.pitch_sine * scene_z
        pitched_z = transform.pitch_sine * rolled_y + transform.pitch_cosine * scene_z
        rotated_x = transform.yaw_cosine * rolled_x + transform.yaw_sine * pitched_z
        rotated_z = -transform.yaw_sine * rolled_x + transform.yaw_cosine * pitched_z
        push!(vertices, SVector{3,T}(rotated_x, pitched_y, rotated_z) + transform.position)
    end
    return BoundaryMesh(vertices, mesh.faces, mesh.physical_tags)
end

function deploy_complex_vector(raw, ::Type{T}) where {T<:AbstractFloat}
    raw isa AbstractDict || error("Deploy boundary_neumann must be an object.")
    real_values = get_value(raw, "real", Any[])
    imag_values = get_value(raw, "imag", Any[])
    length(real_values) == length(imag_values) || error("Deploy boundary trace real and imaginary arrays differ in length.")
    return Complex{T}[Complex{T}(T(real_values[index]), T(imag_values[index])) for index in eachindex(real_values)]
end

function deploy_observation_points(request, ::Type{T}) where {T<:AbstractFloat}
    raw_points = get_value(request, "observation_points_m", Any[])
    points = SVector{3,T}[]
    sizehint!(points, length(raw_points))
    for raw in raw_points
        length(raw) == 3 || error("Every Deploy observation point must contain three values.")
        point = SVector{3,T}(T(raw[1]), T(raw[2]), T(raw[3]))
        all(isfinite, point) || error("Deploy observation points must be finite.")
        push!(points, point)
    end
    isempty(points) && error("Deploy solve requires at least one observation point.")
    return points
end

function deploy_cuda_observation_plane(request, ::Type{T}) where {T<:AbstractFloat}
    raw = get_value(request, "observation_plane", nothing)
    raw isa AbstractDict || error("CUDA Deploy field request requires an observation_plane object.")
    width = T(get_value(raw, "width_m", 0.0))
    depth = T(get_value(raw, "depth_m", 0.0))
    center_x = T(get_value(raw, "center_x_m", 0.0))
    near = T(get_value(raw, "near_m", 0.0))
    height = T(get_value(raw, "height_m", 0.0))
    pitch = T(pi) * T(get_value(raw, "pitch_deg", 0.0)) / T(180)
    yaw = T(pi) * T(get_value(raw, "yaw_deg", 0.0)) / T(180)
    roll = T(pi) * T(get_value(raw, "roll_deg", 0.0)) / T(180)
    columns = Int(get_value(raw, "columns", 0))
    rows = Int(get_value(raw, "rows", 0))
    ground_tolerance = T(get_value(raw, "ground_tolerance_m", 1e-6))
    width > 0 && depth > 0 || error("CUDA observation plane dimensions must be positive.")
    columns >= 2 && rows >= 2 || error("CUDA observation plane requires at least two rows and columns.")
    all(isfinite, (width, depth, center_x, near, height, pitch, yaw, roll, ground_tolerance)) || error(
        "CUDA observation plane values must be finite.",
    )
    roll_cosine = cos(roll)
    roll_sine = sin(roll)
    pitch_cosine = cos(pitch)
    pitch_sine = sin(pitch)
    yaw_cosine = cos(yaw)
    yaw_sine = sin(yaw)
    sample_indices = Int[]
    sizehint!(sample_indices, columns * rows)
    for row in 0:(rows - 1), column in 0:(columns - 1)
        local_x = -width / T(2) + width * T(column) / T(columns - 1)
        local_z = -depth / T(2) + depth * T(row) / T(rows - 1)
        rolled_y = roll_sine * local_x
        world_y = height + pitch_cosine * rolled_y - pitch_sine * local_z
        world_y >= -ground_tolerance && push!(sample_indices, row * columns + column)
    end
    isempty(sample_indices) && error("Deploy observation plane has no sampling points on or above the ground plane.")
    return (
        width=width,
        depth=depth,
        center_x=center_x,
        center_z=near + depth / T(2),
        height=height,
        roll_cosine=roll_cosine,
        roll_sine=roll_sine,
        pitch_cosine=pitch_cosine,
        pitch_sine=pitch_sine,
        yaw_cosine=yaw_cosine,
        yaw_sine=yaw_sine,
        columns=columns,
        rows=rows,
        sample_indices=sample_indices,
    )
end

function build_deploy_cuda_observation_points(plane)
    return build_cuda_observation_points(
        plane.sample_indices,
        plane.width,
        plane.depth,
        plane.center_x,
        plane.center_z,
        plane.height,
        plane.roll_cosine,
        plane.roll_sine,
        plane.pitch_cosine,
        plane.pitch_sine,
        plane.yaw_cosine,
        plane.yaw_sine,
        plane.columns,
        plane.rows,
    )
end

function deploy_rom_binary_array(descriptor, ::Type{T}) where {T<:AbstractFloat}
    String(get_value(descriptor, "dtype", "")) == "complex64" || error(
        "Deploy speaker ROM currently requires complex64 binary arrays.",
    )
    String(get_value(descriptor, "order", "")) == "C" || error(
        "Deploy speaker ROM currently requires C-order binary arrays.",
    )
    path = String(descriptor["file"])
    isfile(path) || error("Deploy speaker ROM binary array file does not exist: $path")
    shape = Int.(get_value(descriptor, "shape", Any[]))
    isempty(shape) && error("Deploy speaker ROM binary array has no shape.")
    all(>(0), shape) || error("Deploy speaker ROM binary array has an invalid shape.")
    element_count = prod(shape)
    expected_bytes = element_count * sizeof(ComplexF32)
    Int(get_value(descriptor, "nbytes", -1)) == expected_bytes || error(
        "Deploy speaker ROM binary array byte count does not match its shape.",
    )
    offset = Int(get_value(descriptor, "offset", -1))
    offset >= 0 || error("Deploy speaker ROM binary array offset is invalid.")
    values = open(path, "r") do stream
        seek(stream, offset)
        read!(stream, Vector{ComplexF32}(undef, element_count))
    end
    native = if length(shape) == 1
        reshape(values, shape...)
    else
        reversed_dimensions = reverse(collect(1:length(shape)))
        permutedims(reshape(values, reverse(shape)...), reversed_dimensions)
    end
    return Complex{T}.(native)
end

function load_deploy_speaker_rom(request, ::Type{T}, node_count::Int, face_count::Int) where {T<:AbstractFloat}
    raw = get_value(request, "rom", nothing)
    raw isa AbstractDict || error("Deploy Level 3 ROM request is missing its rom object.")
    String(get_value(raw, "representation", "")) == "parity_petrov_galerkin_rom" || error(
        "Deploy Level 3 request uses an unsupported reduced representation.",
    )
    descriptors = get_value(raw, "binary_arrays", nothing)
    descriptors isa AbstractDict || error("Deploy speaker ROM is missing binary array descriptors.")
    arrays = Dict(
        name => deploy_rom_binary_array(descriptors[name], T)
        for name in (
            "k",
            "c",
            "d",
            "b",
            "e",
            "velocity",
            "current",
            "velocity_drive",
            "current_drive",
        )
    )
    rank = Int(get_value(raw, "rank_per_sector", 0))
    rank > 0 || error("Deploy speaker ROM rank must be positive.")
    size(arrays["k"]) == (4, rank, rank) || error("Deploy speaker ROM K shape is invalid.")
    input_count = size(arrays["b"], 3)
    size(arrays["b"]) == (4, rank, input_count) || error("Deploy speaker ROM B shape is invalid.")

    raw_node_orbits = get_value(raw, "node_orbits", Any[])
    raw_face_orbits = get_value(raw, "face_orbits", Any[])
    node_orbits = [ntuple(index -> Int(orbit[index]) + 1, 4) for orbit in raw_node_orbits]
    face_orbits = [ntuple(index -> Int(orbit[index]) + 1, 4) for orbit in raw_face_orbits]
    isempty(node_orbits) && error("Deploy speaker ROM has no node orbits.")
    isempty(face_orbits) && error("Deploy speaker ROM has no face orbits.")
    size(arrays["c"]) == (4, rank, length(node_orbits)) || error("Deploy speaker ROM C shape is invalid.")
    size(arrays["d"]) == (4, length(face_orbits), rank) || error("Deploy speaker ROM D shape is invalid.")
    size(arrays["e"]) == (4, length(face_orbits), input_count) || error("Deploy speaker ROM E shape is invalid.")
    transducer_count = size(arrays["velocity"], 2)
    size(arrays["velocity"]) == (4, transducer_count, rank) || error(
        "Deploy speaker ROM velocity output shape is invalid.",
    )
    size(arrays["current"]) == (4, transducer_count, rank) || error(
        "Deploy speaker ROM current output shape is invalid.",
    )
    size(arrays["velocity_drive"]) == (4, transducer_count, input_count) || error(
        "Deploy speaker ROM velocity drive shape is invalid.",
    )
    size(arrays["current_drive"]) == (4, transducer_count, input_count) || error(
        "Deploy speaker ROM current drive shape is invalid.",
    )
    signs = [ntuple(index -> Int(sector[index]), 2) for sector in get_value(raw, "sector_signs", Any[])]
    length(signs) == 4 || error("Deploy speaker ROM must contain four parity-sector signs.")

    package_node_count = maximum(maximum(orbit) for orbit in node_orbits)
    package_face_count = maximum(maximum(orbit) for orbit in face_orbits)
    instances = NamedTuple[]
    for instance in get_value(raw, "instances", Any[])
        input_real = T.(get_value(instance, "input_real", Any[]))
        input_imag = T.(get_value(instance, "input_imag", Any[]))
        length(input_real) == input_count == length(input_imag) || error(
            "Deploy speaker ROM instance input count is invalid.",
        )
        node_offset = Int(get_value(instance, "node_offset", -1))
        face_offset = Int(get_value(instance, "face_offset", -1))
        node_offset >= 0 && node_offset + package_node_count <= node_count || error(
            "Deploy speaker ROM instance node range is outside the scene mesh.",
        )
        face_offset >= 0 && face_offset + package_face_count <= face_count || error(
            "Deploy speaker ROM instance face range is outside the scene mesh.",
        )
        push!(instances, (
            id=String(get_value(instance, "id", "")),
            node_offset=node_offset,
            face_offset=face_offset,
            input=Complex{T}.(input_real, input_imag),
        ))
    end
    isempty(instances) && error("Deploy speaker ROM requires at least one speaker instance.")
    factors = [lu!(Matrix(view(arrays["k"], sector, :, :))) for sector in 1:4]
    return (
        rank=rank,
        arrays=arrays,
        node_orbits=node_orbits,
        face_orbits=face_orbits,
        signs=signs,
        instances=instances,
        factors=factors,
        package_node_count=package_node_count,
        package_face_count=package_face_count,
        node_count=node_count,
        face_count=face_count,
        tolerance=T(get_value(raw, "gmres_tolerance", 1e-4)),
        max_iterations=Int(get_value(raw, "gmres_max_iterations", 30)),
    )
end

function deploy_speaker_rom_response(model, pressure::AbstractVector; include_drive::Bool)
    T = typeof(real(zero(eltype(pressure))))
    q = zeros(eltype(pressure), model.face_count)
    velocities = [zeros(eltype(pressure), size(model.arrays["velocity"], 2)) for _ in model.instances]
    currents = [zeros(eltype(pressure), size(model.arrays["current"], 2)) for _ in model.instances]
    for (instance_index, instance) in enumerate(model.instances)
        local_pressure = view(
            pressure,
            (instance.node_offset + 1):(instance.node_offset + model.package_node_count),
        )
        for sector in 1:4
            sign_x, sign_y = model.signs[sector]
            image_signs = (1, sign_x, sign_y, sign_x * sign_y)
            compact_pressure = zeros(eltype(pressure), length(model.node_orbits))
            for (orbit_index, orbit) in enumerate(model.node_orbits)
                compact_pressure[orbit_index] = sum(
                    image_signs[image] * local_pressure[orbit[image]] for image in 1:4
                ) / T(4)
            end
            drive = include_drive ? instance.input : zeros(eltype(pressure), length(instance.input))
            reduced_rhs = view(model.arrays["b"], sector, :, :) * drive -
                          view(model.arrays["c"], sector, :, :) * compact_pressure
            state = model.factors[sector] \ reduced_rhs
            compact_q = view(model.arrays["d"], sector, :, :) * state +
                        view(model.arrays["e"], sector, :, :) * drive
            sector_q = zeros(eltype(pressure), model.package_face_count)
            for (orbit_index, orbit) in enumerate(model.face_orbits), image in 1:4
                sector_q[orbit[image]] = image_signs[image] * compact_q[orbit_index]
            end
            q[(instance.face_offset + 1):(instance.face_offset + model.package_face_count)] .+= sector_q
            velocities[instance_index] .+= view(model.arrays["velocity"], sector, :, :) * state +
                                            view(model.arrays["velocity_drive"], sector, :, :) * drive
            currents[instance_index] .+= view(model.arrays["current"], sector, :, :) * state +
                                         view(model.arrays["current_drive"], sector, :, :) * drive
        end
    end
    return (q=q, velocities=velocities, currents=currents)
end

function deploy_cuda_gmres(apply_operator, right_hand_side; tolerance, max_iterations)
    cuda = BeatEngineCore.CUDA_MODULE
    T = typeof(real(zero(eltype(right_hand_side))))
    max_iterations > 0 || error("Deploy speaker ROM GMRES iteration limit must be positive.")
    tolerance > zero(T) || error("Deploy speaker ROM GMRES tolerance must be positive.")
    beta = norm(right_hand_side)
    beta > eps(T) || return (cuda.zeros(eltype(right_hand_side), length(right_hand_side)), 0, zero(T), T[])
    basis = cuda.zeros(eltype(right_hand_side), length(right_hand_side), max_iterations + 1)
    hessenberg = zeros(Complex{T}, max_iterations + 1, max_iterations)
    residual_history = T[]
    solution = nothing
    used_iterations = 0
    final_coefficients = Complex{T}[]
    try
        view(basis, :, 1) .= right_hand_side ./ beta
        for iteration in 1:max_iterations
            work = apply_operator(view(basis, :, iteration))
            try
                for previous in 1:iteration
                    coefficient = dot(view(basis, :, previous), work)
                    hessenberg[previous, iteration] = coefficient
                    work .-= coefficient .* view(basis, :, previous)
                end
                next_norm = norm(work)
                hessenberg[iteration + 1, iteration] = next_norm
                if next_norm > eps(T)
                    view(basis, :, iteration + 1) .= work ./ next_norm
                end
            finally
                cuda.unsafe_free!(work)
            end
            small_rhs = zeros(Complex{T}, iteration + 1)
            small_rhs[1] = beta
            coefficients = view(hessenberg, 1:(iteration + 1), 1:iteration) \ small_rhs
            residual = norm(small_rhs - view(hessenberg, 1:(iteration + 1), 1:iteration) * coefficients) / beta
            push!(residual_history, T(residual))
            used_iterations = iteration
            final_coefficients = coefficients
            (residual <= tolerance || abs(hessenberg[iteration + 1, iteration]) <= eps(T)) && break
        end
        coefficient_device = cuda.CuArray(final_coefficients)
        try
            solution = view(basis, :, 1:used_iterations) * coefficient_device
            cuda.synchronize()
        finally
            cuda.unsafe_free!(coefficient_device)
        end
        return solution, used_iterations, last(residual_history), residual_history
    finally
        cuda.unsafe_free!(basis)
    end
end

function solve_deploy_request_impl(request; emit_completed::Bool=true)
    request_started = time()
    request_schema = String(get_value(request, "schema", "boundary_lab_deploy_solve"))
    rom_request = request_schema == "boundary_lab_deploy_rom"
    request_schema in ("boundary_lab_deploy_solve", "boundary_lab_deploy_rom") || error(
        "Unsupported Deploy boundary solve schema $request_schema.",
    )
    schema_version = Int(get_value(request, "schema_version", 1))
    schema_version in (1, 2) || error("Unsupported Deploy solve schema_version $(schema_version).")
    beat_backend = beat_backend_from_request(request)
    beat_backend in (:cuda, :cpu) || error("Deploy Level 2 currently supports BEAT CUDA or CPU.")
    requested_assembly_mode = lowercase(String(get_value(request, "burton_miller_assembly", "direct_system")))
    requested_assembly_mode in ("direct_system", "operator_matrices") || error(
        "Deploy burton_miller_assembly must be 'direct_system' or 'operator_matrices'.",
    )
    direct_cuda_assembly = beat_backend == :cuda && requested_assembly_mode == "direct_system"
    rom_request && !direct_cuda_assembly && error(
        "Deploy Level 3 parity ROM currently requires direct-system BEAT CUDA.",
    )
    assembly_mode = direct_cuda_assembly ? "direct_system" : "operator_matrices"
    retain_geometry_cache = Bool(get_value(request, "retain_geometry_cache", false))
    geometry_key = String(get_value(request, "geometry_key", ""))
    cached_geometry = retain_geometry_cache ? DEPLOY_GEOMETRY_STATE[] : nothing
    if cached_geometry !== nothing && (
        cached_geometry.key != geometry_key ||
        cached_geometry.backend != beat_backend ||
        cached_geometry.assembly_mode != assembly_mode
    )
        release_deploy_geometry_state!()
        cached_geometry = nothing
    end

    FloatType = Float32
    frequency = FloatType(request["frequency_hz"])
    frequency > 0 || error("Deploy frequency_hz must be positive.")
    density = FloatType(get_value(request, "density_kg_per_m3", 1.21))
    sound_speed = FloatType(get_value(request, "sound_speed_m_per_s", 343.0))
    density > 0 || error("Deploy density must be positive.")
    sound_speed > 0 || error("Deploy sound speed must be positive.")
    k = FloatType(2pi) * frequency / sound_speed
    boundary = get_value(request, "boundary", Dict{String,Any}())
    ground_plane = get_value(boundary, "ground_plane", Dict{String,Any}())
    get_value(ground_plane, "type", "rigid_half_space") == "rigid_half_space" || error(
        "Deploy Level 2 requires the global rigid half-space ground boundary.",
    )
    lowercase(String(get_value(ground_plane, "axis", "y"))) == "y" || error(
        "Deploy rigid ground must use the world Y axis.",
    )
    Float64(get_value(ground_plane, "offset_m", 0.0)) == 0.0 || error(
        "Deploy rigid ground must remain at Y=0.",
    )
    Float64(get_value(ground_plane, "reflection_coefficient", 1.0)) == 1.0 || error(
        "Deploy rigid ground requires reflection coefficient +1.",
    )

    cuda_observation = nothing
    input_geometry_started = time()
    mesh = nothing
    source_meshes = nothing
    if cached_geometry === nothing
        emit_event("status"; message="Loading fixed-source speaker boundary")
        package_mesh = load_gmsh22_with_tags(
            String(request["mesh_file"]),
            FloatType(get_value(request, "mesh_scale_factor", 1.0)),
        )
        mesh_is_world_space = Bool(get_value(request, "mesh_is_world_space", false))
        source_transforms = deploy_source_transforms(request, FloatType)
        source_meshes = mesh_is_world_space ? [package_mesh] :
            [transform_deploy_mesh(package_mesh, transform) for transform in source_transforms]
        mesh = mesh_is_world_space ? package_mesh : combine_boundary_meshes(source_meshes).mesh
    else
        mesh = cached_geometry.mesh
        source_meshes = [mesh]
        emit_event("status"; message="Reusing BEAT geometry caches")
    end
    speaker_rom = rom_request ? load_deploy_speaker_rom(
        request,
        FloatType,
        length(mesh.vertices),
        length(mesh.faces),
    ) : nothing
    initial_rom_response = rom_request ? deploy_speaker_rom_response(
        speaker_rom,
        zeros(Complex{FloatType}, length(mesh.vertices));
        include_drive=true,
    ) : nothing
    q_neumann = rom_request ? initial_rom_response.q : deploy_complex_vector(
        request["boundary_neumann"],
        FloatType,
    )
    length(q_neumann) == length(mesh.faces) || error(
        "Deploy boundary trace contains $(length(q_neumann)) faces, but the mesh contains $(length(mesh.faces)).",
    )
    all(isfinite, q_neumann) || error("Deploy boundary trace must be finite.")
    reference_pressure = deploy_complex_vector(request["reference_boundary_pressure"], FloatType)
    length(reference_pressure) == length(mesh.vertices) || error(
        "Deploy reference pressure contains $(length(reference_pressure)) nodes, but the mesh contains $(length(mesh.vertices)).",
    )
    reference_pressure_mask = Bool.(get_value(
        request,
        "reference_boundary_pressure_mask",
        ones(Int, length(reference_pressure)),
    ))
    length(reference_pressure_mask) == length(reference_pressure) || error(
        "Deploy reference-pressure mask does not match the boundary vertex count.",
    )
    observation_points = nothing
    observation_shape = Int[]
    observation_sample_indices = Int[]
    if beat_backend == :cuda && get_value(request, "observation_plane", nothing) !== nothing
        plane = deploy_cuda_observation_plane(request, FloatType)
        observation_shape = [plane.rows, plane.columns]
        observation_sample_indices = plane.sample_indices
        cuda_observation = build_deploy_cuda_observation_points(plane)
        observation_points = cuda_observation
    else
        observation_points = deploy_observation_points(request, FloatType)
        observation_shape = Int.(get_value(request, "observation_shape", [1, length(observation_points)]))
        length(observation_shape) == 2 || error("Deploy observation_shape must contain rows and columns.")
        observation_sample_indices = Int.(
            get_value(request, "observation_sample_indices", collect(0:(length(observation_points) - 1))),
        )
        length(observation_sample_indices) == length(observation_points) || error(
            "Deploy observation sample indices do not match the point count.",
        )
    end
    all(index -> 0 <= index < prod(observation_shape), observation_sample_indices) || error("Deploy observation sample index is outside the grid.")
    length(unique(observation_sample_indices)) == length(observation_sample_indices) || error("Deploy observation sample indices must be unique.")
    input_geometry_seconds = time() - input_geometry_started

    quadrature_order = Int(get_value(request, "quadrature_order", 2))
    singular_order = Int(get_value(request, "singular_order", 3))
    close_pair_quadrature_order = Int(get_value(request, "close_pair_quadrature_order", 8))
    close_pair_quadrature_order >= 4 || error("Deploy close-pair quadrature order must be at least 4.")
    proximity = get_value(request, "proximity", Dict{String,Any}())
    host_cache_started = time()
    p1_space = nothing
    dp0_space = nothing
    rule = nothing
    identity_p1_p1 = nothing
    identity_p1_dp0 = nothing
    singular_cache = nothing
    near_correction_cache = nothing
    ground_near_correction_cache = nothing
    cpu_field_cache = nothing
    if cached_geometry === nothing
        p1_space = build_p1_space(mesh)
        dp0_space = build_dp0_space(mesh)
        rule = triangle_rule(FloatType, quadrature_order)
        identity_p1_p1 = direct_cuda_assembly ? nothing :
            assemble_l2_identity_matrix(mesh, p1_space, dp0_space, rule, :p1, :p1)
        identity_p1_dp0 = direct_cuda_assembly ? nothing :
            assemble_l2_identity_matrix(mesh, p1_space, dp0_space, rule, :p1, :dp0)
        singular_cache = build_singular_correction_cache(mesh, singular_order)
        raw_close_face_pairs = get_value(proximity, "close_face_pairs", Any[])
        close_face_pairs = [
            length(pair) >= 3 ? (Int(pair[1]) + 1, Int(pair[2]) + 1, Int(pair[3])) :
            (Int(pair[1]) + 1, Int(pair[2]) + 1, close_pair_quadrature_order)
            for pair in raw_close_face_pairs
        ]
        near_correction_cache = build_near_correction_cache(mesh, close_face_pairs, close_pair_quadrature_order)
        raw_ground_close_face_pairs = get_value(proximity, "ground_image_close_face_pairs", Any[])
        ground_close_face_pairs = [
            length(pair) >= 3 ? (Int(pair[1]) + 1, Int(pair[2]) + 1, Int(pair[3])) :
            (Int(pair[1]) + 1, Int(pair[2]) + 1, close_pair_quadrature_order)
            for pair in raw_ground_close_face_pairs
        ]
        ground_near_correction_cache = build_near_correction_cache(
            mesh,
            ground_close_face_pairs,
            close_pair_quadrature_order;
            trial_transform=rigid_ground_transform(),
        )
        cpu_field_cache = build_field_evaluation_cache(mesh, rule; symmetry_mode=:ground)
    else
        p1_space = cached_geometry.p1_space
        dp0_space = cached_geometry.dp0_space
        rule = cached_geometry.rule
        identity_p1_p1 = cached_geometry.identity_p1_p1
        identity_p1_dp0 = cached_geometry.identity_p1_dp0
        singular_cache = cached_geometry.singular_cache
        near_correction_cache = cached_geometry.near_correction_cache
        ground_near_correction_cache = cached_geometry.ground_near_correction_cache
        cpu_field_cache = cached_geometry.cpu_field_cache
    end
    host_cache_seconds = time() - host_cache_started

    emit_event(
        "initialized";
        backend=String(beat_backend),
        frequency_hz=frequency,
        node_count=length(mesh.vertices),
        face_count=length(mesh.faces),
        source_count=Int(get_value(get_value(request, "provenance", Dict{String,Any}()), "source_count", length(source_meshes))),
        observation_count=length(observation_sample_indices),
    )

    device_cache = cached_geometry === nothing ? nothing : cached_geometry.device_cache
    device_singular_cache = cached_geometry === nothing ? nothing : cached_geometry.device_singular_cache
    device_image_singular_cache = cached_geometry === nothing ? nothing : cached_geometry.device_image_singular_cache
    device_near_correction_cache = cached_geometry === nothing ? nothing : cached_geometry.device_near_correction_cache
    device_ground_near_correction_cache = cached_geometry === nothing ? nothing : cached_geometry.device_ground_near_correction_cache
    cuda_identity_cache = cached_geometry === nothing ? nothing : cached_geometry.cuda_identity_cache
    field_cache = cached_geometry === nothing ? cpu_field_cache : cached_geometry.field_cache
    operators = nothing
    direct_system = nothing
    direct_system_consumed = false
    direct_assembly_timings = Dict{String,Float64}()
    assembly_seconds = 0.0
    solve_seconds = 0.0
    field_seconds = 0.0
    device_prepare_seconds = 0.0
    pressure = nothing
    field_pressure = nothing
    spl_db = nothing
    cached_q_neumann = nothing
    weighted_sources = nothing
    rom_factorization = nothing
    rom_iterations = 0
    rom_residual = FloatType(NaN)
    rom_residual_history = FloatType[]
    final_rom_response = initial_rom_response
    try
        device_prepare_seconds = @elapsed begin
            if beat_backend == :cuda
                if cached_geometry === nothing
                    emit_event("status"; message="Preparing BEAT CUDA geometry caches")
                    device_cache = build_cuda_regular_assembly_cache(mesh, rule)
                    device_singular_cache = BeatEngineCore.build_cuda_singular_correction_cache(
                        singular_cache,
                        p1_space,
                        dp0_space,
                    )
                    device_image_singular_cache = build_cuda_image_singular_correction_cache(
                        mesh,
                        p1_space,
                        dp0_space,
                        singular_order,
                        eachindex(mesh.faces),
                        :ground,
                    )
                    if near_correction_cache.pair_count > 0
                        device_near_correction_cache = build_cuda_near_correction_cache(
                            near_correction_cache,
                            p1_space,
                            dp0_space,
                        )
                    end
                    if ground_near_correction_cache.pair_count > 0
                        device_ground_near_correction_cache = build_cuda_near_correction_cache(
                            ground_near_correction_cache,
                            p1_space,
                            dp0_space,
                        )
                    end
                    if !direct_cuda_assembly
                        cuda_identity_cache = build_cuda_burton_miller_identity_cache(
                            identity_p1_p1,
                            identity_p1_dp0,
                            FloatType,
                        )
                    end
                    field_cache = build_cuda_field_evaluation_cache(cpu_field_cache)
                end
                cached_q_neumann = BeatEngineCore.CUDA_MODULE.CuArray(q_neumann)
            else
                cached_geometry === nothing && emit_event("status"; message="Preparing BEAT CPU geometry caches")
            end
        end

        if retain_geometry_cache && cached_geometry === nothing
            DEPLOY_GEOMETRY_STATE[] = (
                key=geometry_key,
                backend=beat_backend,
                assembly_mode=assembly_mode,
                mesh=mesh,
                p1_space=p1_space,
                dp0_space=dp0_space,
                rule=rule,
                identity_p1_p1=identity_p1_p1,
                identity_p1_dp0=identity_p1_dp0,
                singular_cache=singular_cache,
                near_correction_cache=near_correction_cache,
                ground_near_correction_cache=ground_near_correction_cache,
                cpu_field_cache=cpu_field_cache,
                device_cache=device_cache,
                device_singular_cache=device_singular_cache,
                device_image_singular_cache=device_image_singular_cache,
                device_near_correction_cache=device_near_correction_cache,
                device_ground_near_correction_cache=device_ground_near_correction_cache,
                cuda_identity_cache=cuda_identity_cache,
                field_cache=field_cache,
            )
        end

        assembly_message = rom_request ?
            "Assembling Level 3 Schur exterior preconditioner" : direct_cuda_assembly ?
            "Assembling Level 2 rigid half-space Burton-Miller system" :
            "Assembling Level 2 rigid half-space boundary operators"
        emit_event("status"; message=assembly_message)
        assembly_seconds = @elapsed begin
            if direct_cuda_assembly
                direct_system = assemble_burton_miller_neumann_system_cuda(
                    mesh,
                    p1_space,
                    dp0_space,
                    cached_q_neumann,
                    k,
                    rule;
                    device_cache=device_cache,
                    singular_cache=singular_cache,
                    device_singular_cache=device_singular_cache,
                    device_image_singular_cache=device_image_singular_cache,
                    near_correction_cache=near_correction_cache,
                    device_near_correction_cache=device_near_correction_cache,
                    image_near_correction_cache=ground_near_correction_cache,
                    device_image_near_correction_cache=device_ground_near_correction_cache,
                    symmetry_mode=:ground,
                    timing=direct_assembly_timings,
                )
            else
                operators = assemble_regular_galerkin_operators(
                    mesh,
                    p1_space,
                    dp0_space,
                    k,
                    rule;
                    skip_singular=false,
                    singular_order=singular_order,
                    backend=beat_backend,
                    device_cache=device_cache,
                    return_device=beat_backend == :cuda,
                    accelerator_quadrature=beat_backend == :cuda,
                    singular_cache=singular_cache,
                    device_singular_cache=device_singular_cache,
                    device_image_singular_cache=device_image_singular_cache,
                    near_correction_cache=near_correction_cache,
                    device_near_correction_cache=device_near_correction_cache,
                    image_near_correction_cache=ground_near_correction_cache,
                    device_image_near_correction_cache=device_ground_near_correction_cache,
                    symmetry_mode=:ground,
                )
            end
        end

        emit_event(
            "status";
            message=rom_request ?
                "Solving Schur-eliminated rank-$(speaker_rom.rank)-per-sector system" :
                "Solving fixed-Neumann exterior system",
        )
        solve_seconds = @elapsed begin
            pressure = if rom_request
                cuda = BeatEngineCore.CUDA_MODULE
                rom_factorization = lu!(direct_system.matrix)
                direct_system_consumed = true
                preconditioned_rhs = rom_factorization \ direct_system.rhs
                try
                    apply_schur = function(candidate_pressure)
                        feedback = deploy_speaker_rom_response(
                            speaker_rom,
                            Complex{FloatType}.(Array(candidate_pressure));
                            include_drive=false,
                        )
                        feedback_device = cuda.CuArray(feedback.q)
                        feedback_rhs = feedback_solution = result = nothing
                        try
                            feedback_rhs = assemble_burton_miller_rhs_cuda(
                                mesh,
                                p1_space,
                                dp0_space,
                                feedback_device,
                                k,
                                rule;
                                device_cache=device_cache,
                                singular_cache=singular_cache,
                                device_singular_cache=device_singular_cache,
                                device_image_singular_cache=device_image_singular_cache,
                                near_correction_cache=near_correction_cache,
                                device_near_correction_cache=device_near_correction_cache,
                                image_near_correction_cache=ground_near_correction_cache,
                                device_image_near_correction_cache=device_ground_near_correction_cache,
                                symmetry_mode=:ground,
                            )
                            feedback_solution = rom_factorization \ feedback_rhs
                            result = copy(candidate_pressure)
                            result .-= feedback_solution
                            return result
                        finally
                            cuda.unsafe_free!(feedback_device)
                            feedback_rhs === nothing || cuda.unsafe_free!(feedback_rhs)
                            feedback_solution === nothing || cuda.unsafe_free!(feedback_solution)
                        end
                    end
                    rom_pressure, rom_iterations, rom_residual, rom_residual_history = deploy_cuda_gmres(
                        apply_schur,
                        preconditioned_rhs;
                        tolerance=speaker_rom.tolerance,
                        max_iterations=speaker_rom.max_iterations,
                    )
                    final_rom_response = deploy_speaker_rom_response(
                        speaker_rom,
                        Complex{FloatType}.(Array(rom_pressure));
                        include_drive=true,
                    )
                    cuda.unsafe_free!(cached_q_neumann)
                    cached_q_neumann = cuda.CuArray(final_rom_response.q)
                    rom_pressure
                finally
                    cuda.unsafe_free!(preconditioned_rhs)
                end
            elseif direct_cuda_assembly
                direct_system_consumed = true
                solve_burton_miller_system_cuda!(direct_system; return_gpu=true)
            elseif beat_backend == :cuda
                solve_burton_miller_neumann(
                    operators,
                    cuda_identity_cache,
                    cached_q_neumann,
                    k;
                    return_gpu=true,
                )
            else
                cached_q_neumann = copy(q_neumann)
                solve_burton_miller_neumann(operators, identity_p1_p1, identity_p1_dp0, q_neumann, k)
            end
        end

        emit_event("status"; message="Evaluating audience plane")
        include_complex_pressure = Bool(get_value(request, "include_complex_pressure", false))
        field_seconds = @elapsed begin
            if beat_backend == :cuda
                weighted_sources = build_cuda_weighted_field_sources(field_cache, pressure, cached_q_neumann)
                if include_complex_pressure
                    field_pressure = evaluate_galerkin_field_cuda(
                        observation_points,
                        mesh,
                        pressure,
                        cached_q_neumann,
                        k,
                        field_cache;
                        weighted_sources=weighted_sources,
                    )
                    spl_db = pressure_to_spl(field_pressure, FloatType)
                else
                    spl_db = evaluate_galerkin_spl_cuda(
                        observation_points,
                        mesh,
                        pressure,
                        cached_q_neumann,
                        k,
                        field_cache;
                        weighted_sources=weighted_sources,
                    )
                end
            else
                field_pressure = field_for_points(
                    observation_points,
                    mesh,
                    pressure,
                    cached_q_neumann,
                    k,
                    field_cache,
                    beat_backend,
                )
                spl_db = pressure_to_spl(field_pressure, FloatType)
            end
        end
        solution_key = String(get_value(request, "solution_key", ""))
        isempty(solution_key) && error("Deploy Level 2 solve requires a boundary solution key.")
        release_deploy_boundary_state!()
        DEPLOY_BOUNDARY_STATE[] = (
            solution_key=solution_key,
            backend=beat_backend,
            frequency=frequency,
            wavenumber=k,
            mesh=mesh,
            pressure=pressure,
            q_neumann=cached_q_neumann,
            field_cache=field_cache,
            weighted_sources=weighted_sources,
            shared_geometry=retain_geometry_cache,
        )
        result = nothing
        postprocess_seconds = @elapsed begin
            diagnostic_pressure = beat_backend == :cuda ? Complex{FloatType}.(Array(pressure)) : pressure
            isolated_trace_relative_difference = if any(reference_pressure_mask)
                diagnostic_reference = reference_pressure[reference_pressure_mask]
                Float32(
                    norm(diagnostic_pressure[reference_pressure_mask] - diagnostic_reference) /
                    max(norm(diagnostic_reference), eps(FloatType)),
                )
            else
                nothing
            end
            proximity_pairs = get_value(proximity, "pairs", Any[])
            close_pair_count = count(pair -> Bool(get_value(pair, "close", false)), proximity_pairs)
            assembly_diagnostics = direct_cuda_assembly ? direct_system : operators
            result = Dict(
                "frequency_hz" => frequency,
                "rows" => observation_shape[1],
                "columns" => observation_shape[2],
                "spl_db" => spl_db,
                "sample_indices" => observation_sample_indices,
                "timings" => Dict(
                    "assembly_s" => Float32(assembly_seconds),
                    "solve_s" => Float32(solve_seconds),
                    "field_s" => Float32(field_seconds),
                ),
                "diagnostics" => Dict(
                    "backend" => String(beat_backend),
                    "source_count" => Int(get_value(
                        get_value(request, "provenance", Dict{String,Any}()),
                        "source_count",
                        length(source_meshes),
                    )),
                    "rigid_object_count" => Int(get_value(
                        get_value(request, "provenance", Dict{String,Any}()),
                        "rigid_object_count",
                        0,
                    )),
                    "node_count" => length(mesh.vertices),
                    "face_count" => length(mesh.faces),
                    "burton_miller_assembly" => assembly_mode,
                    "singular_pair_count" => singular_cache.pair_count,
                    "near_face_pair_count" => near_correction_cache.pair_count,
                    "ground_image_near_face_pair_count" => ground_near_correction_cache.pair_count,
                    "ground_image_singular_pair_count" => assembly_diagnostics.image_singular_pairs,
                    "exterior_domain" => "rigid_y0_half_space",
                    "ground_reflection_coefficient" => 1.0f0,
                    "close_pair_quadrature_order" => close_pair_quadrature_order,
                    "close_pair_distance_m" => Float32(get_value(proximity, "close_pair_distance_m", 0.05)),
                    "quadrature_order" => quadrature_order,
                    "singular_order" => singular_order,
                    "isolated_trace_relative_difference" => isolated_trace_relative_difference,
                    "minimum_surface_distance_m" => get_value(proximity, "minimum_surface_distance_m", nothing),
                    "close_pair_count" => close_pair_count,
                    "surface_padding_m" => Float32(get_value(proximity, "surface_padding_m", 0.01)),
                    "field_only" => false,
                    "gpu_resident_field" => beat_backend == :cuda,
                    "fidelity" => rom_request ? "level3_parity_petrov_galerkin" : "level2",
                ),
            )
            if rom_request
                result["diagnostics"]["rom_rank_per_sector"] = speaker_rom.rank
                result["diagnostics"]["rom_sector_count"] = 4
                result["diagnostics"]["schur_gmres_iterations"] = rom_iterations
                result["diagnostics"]["schur_gmres_relative_residual"] = rom_residual
                result["diagnostics"]["schur_gmres_residual_history"] = rom_residual_history
                result["diagnostics"]["transducer_velocity"] = [
                    Dict(
                        "real" => Float32.(real.(values)),
                        "imag" => Float32.(imag.(values)),
                    ) for values in final_rom_response.velocities
                ]
                result["diagnostics"]["transducer_current"] = [
                    Dict(
                        "real" => Float32.(real.(values)),
                        "imag" => Float32.(imag.(values)),
                    ) for values in final_rom_response.currents
                ]
            end
            if include_complex_pressure
                result["field_pressure"] = Dict(
                    "real" => Float32.(real.(field_pressure)),
                    "imag" => Float32.(imag.(field_pressure)),
                )
            end
        end
        result["timings"]["input_geometry_s"] = Float32(input_geometry_seconds)
        result["timings"]["host_cache_s"] = Float32(host_cache_seconds)
        result["timings"]["device_prepare_s"] = Float32(device_prepare_seconds)
        result["timings"]["postprocess_s"] = Float32(postprocess_seconds)
        result["timings"]["total_before_emit_s"] = Float32(time() - request_started)
        for (name, seconds) in direct_assembly_timings
            result["timings"][name] = Float32(seconds)
        end
        emit_event("result"; result=result)
    finally
        cuda_observation === nothing || release_cuda_observation_points!(cuda_observation)
        operators === nothing || release_operator_storage!(operators)
        (direct_system === nothing || direct_system_consumed) ||
            release_burton_miller_system_cuda!(direct_system)
        if rom_factorization !== nothing
            cuda = BeatEngineCore.CUDA_MODULE
            cuda.unsafe_free!(rom_factorization.factors)
            cuda.unsafe_free!(rom_factorization.ipiv)
            cuda.unsafe_free!(direct_system.rhs)
        end
        retained_state = DEPLOY_GEOMETRY_STATE[]
        geometry_resources_retained = retain_geometry_cache && retained_state !== nothing &&
            retained_state.field_cache === field_cache
        if !geometry_resources_retained
            cuda_identity_cache === nothing || release_cuda_burton_miller_identity_cache!(cuda_identity_cache)
            device_image_singular_cache === nothing ||
                release_cuda_image_singular_correction_cache!(device_image_singular_cache)
            device_near_correction_cache === nothing ||
                release_cuda_image_singular_correction_cache!(device_near_correction_cache)
            device_ground_near_correction_cache === nothing ||
                release_cuda_image_singular_correction_cache!(device_ground_near_correction_cache)
        end
    end
    emit_completed && emit_event("completed"; solved_count=1)
end

function solve_deploy_microphone_sweep_request_impl(request)
    frequencies = Float64.(get_value(request, "frequencies_hz", Any[]))
    isempty(frequencies) && error("Deploy microphone sweep requires at least one frequency.")
    neumann_sweep = get_value(request, "boundary_neumann_sweep", nothing)
    pressure_sweep = get_value(request, "reference_boundary_pressure_sweep", nothing)
    neumann_sweep isa AbstractDict || error("Deploy microphone sweep requires boundary_neumann_sweep.")
    pressure_sweep isa AbstractDict || error("Deploy microphone sweep requires reference_boundary_pressure_sweep.")
    neumann_real = get_value(neumann_sweep, "real", Any[])
    neumann_imag = get_value(neumann_sweep, "imag", Any[])
    pressure_real = get_value(pressure_sweep, "real", Any[])
    pressure_imag = get_value(pressure_sweep, "imag", Any[])
    all(length(rows) == length(frequencies) for rows in (neumann_real, neumann_imag, pressure_real, pressure_imag)) ||
        error("Deploy microphone sweep traces do not match the frequency count.")
    geometry_key = String(get_value(request, "geometry_key", ""))
    isempty(geometry_key) && error("Deploy microphone sweep requires a geometry_key.")
    release_deploy_geometry_state!()
    try
        for index in eachindex(frequencies)
            emit_event(
                "status";
                message="Solving microphone frequency $(index)/$(length(frequencies)) ($(round(frequencies[index]; digits=2)) Hz)",
            )
            frequency_request = copy(request)
            frequency_request["schema"] = "boundary_lab_deploy_solve"
            frequency_request["schema_version"] = 2
            frequency_request["frequency_hz"] = frequencies[index]
            frequency_request["boundary_neumann"] = Dict(
                "real" => neumann_real[index],
                "imag" => neumann_imag[index],
            )
            frequency_request["reference_boundary_pressure"] = Dict(
                "real" => pressure_real[index],
                "imag" => pressure_imag[index],
            )
            frequency_request["solution_key"] = "$(geometry_key):$(frequencies[index])"
            frequency_request["retain_geometry_cache"] = true
            solve_deploy_request_impl(frequency_request; emit_completed=false)
        end
    finally
        release_deploy_geometry_state!()
    end
    emit_event("completed"; solved_count=length(frequencies))
end

function evaluate_deploy_field_request_impl(request)
    request_started = time()
    state = DEPLOY_BOUNDARY_STATE[]
    state === nothing && error("Deploy field reuse requested before a boundary solution is available.")
    solution_key = String(get_value(request, "solution_key", ""))
    solution_key == state.solution_key || error(
        "Deploy field request does not match the cached boundary solution.",
    )
    beat_backend = beat_backend_from_request(request)
    beat_backend == state.backend || error("Deploy field request backend does not match the cached solution.")
    FloatType = Float32
    cuda_observation = nothing
    observation_prepare_seconds = 0.0
    observation_points = nothing
    observation_shape = Int[]
    observation_sample_indices = Int[]
    if beat_backend == :cuda && get_value(request, "observation_plane", nothing) !== nothing
        observation_prepare_seconds = @elapsed begin
            plane = deploy_cuda_observation_plane(request, FloatType)
            observation_shape = [plane.rows, plane.columns]
            observation_sample_indices = plane.sample_indices
            cuda_observation = build_deploy_cuda_observation_points(plane)
            observation_points = cuda_observation
        end
    else
        observation_prepare_seconds = @elapsed begin
            observation_points = deploy_observation_points(request, FloatType)
            observation_shape = Int.(get_value(request, "observation_shape", [1, length(observation_points)]))
            length(observation_shape) == 2 || error("Deploy observation_shape must contain rows and columns.")
            observation_sample_indices = Int.(
                get_value(request, "observation_sample_indices", collect(0:(length(observation_points) - 1))),
            )
            length(observation_sample_indices) == length(observation_points) || error(
                "Deploy observation sample indices do not match the point count.",
            )
        end
    end
    observation_count = length(observation_sample_indices)

    emit_event(
        "initialized";
        backend=String(beat_backend),
        frequency_hz=state.frequency,
        node_count=length(state.mesh.vertices),
        face_count=length(state.mesh.faces),
        source_count=0,
        observation_count=observation_count,
        field_only=true,
    )
    emit_event("status"; message="Reusing boundary solution for audience plane")
    field_pressure = nothing
    spl_db = nothing
    include_complex_pressure = Bool(get_value(request, "include_complex_pressure", false))
    field_seconds = try
        @elapsed begin
            if beat_backend == :cuda
                if include_complex_pressure
                    field_pressure = evaluate_galerkin_field_cuda(
                        observation_points,
                        state.mesh,
                        state.pressure,
                        state.q_neumann,
                        state.wavenumber,
                        state.field_cache;
                        weighted_sources=state.weighted_sources,
                    )
                    spl_db = pressure_to_spl(field_pressure, FloatType)
                else
                    spl_db = evaluate_galerkin_spl_cuda(
                        observation_points,
                        state.mesh,
                        state.pressure,
                        state.q_neumann,
                        state.wavenumber,
                        state.field_cache;
                        weighted_sources=state.weighted_sources,
                    )
                end
            else
                field_pressure = field_for_points(
                    observation_points,
                    state.mesh,
                    state.pressure,
                    state.q_neumann,
                    state.wavenumber,
                    state.field_cache,
                    beat_backend,
                )
                spl_db = pressure_to_spl(field_pressure, FloatType)
            end
        end
    finally
        cuda_observation === nothing || release_cuda_observation_points!(cuda_observation)
    end
    result = Dict(
        "frequency_hz" => state.frequency,
        "rows" => observation_shape[1],
        "columns" => observation_shape[2],
        "spl_db" => spl_db,
        "sample_indices" => observation_sample_indices,
        "timings" => Dict(
            "assembly_s" => 0.0f0,
            "solve_s" => 0.0f0,
            "field_s" => Float32(field_seconds),
            "input_geometry_s" => 0.0f0,
            "host_cache_s" => 0.0f0,
            "device_prepare_s" => 0.0f0,
            "observation_prepare_s" => Float32(observation_prepare_seconds),
            "postprocess_s" => 0.0f0,
            "total_before_emit_s" => Float32(time() - request_started),
        ),
        "diagnostics" => Dict(
            "backend" => String(beat_backend),
            "source_count" => 0,
            "node_count" => length(state.mesh.vertices),
            "face_count" => length(state.mesh.faces),
            "field_only" => true,
            "gpu_resident_field" => beat_backend == :cuda,
            "gpu_generated_observation" => cuda_observation !== nothing,
            "exterior_domain" => "rigid_y0_half_space",
        ),
    )
    if include_complex_pressure
        result["field_pressure"] = Dict(
            "real" => Float32.(real.(field_pressure)),
            "imag" => Float32.(imag.(field_pressure)),
        )
    end
    emit_event("result"; result=result)
    emit_event("completed"; solved_count=1)
end
