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

function solve_deploy_request_impl(request)
    schema_version = Int(get_value(request, "schema_version", 1))
    schema_version in (1, 2) || error("Unsupported Deploy solve schema_version $(schema_version).")
    beat_backend = beat_backend_from_request(request)
    beat_backend in (:cuda, :cpu) || error("Deploy Level 2 currently supports BEAT CUDA or CPU.")

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

    emit_event("status"; message="Loading fixed-source speaker boundary")
    package_mesh = load_gmsh22_with_tags(
        String(request["mesh_file"]),
        FloatType(get_value(request, "mesh_scale_factor", 1.0)),
    )
    source_transforms = deploy_source_transforms(request, FloatType)
    source_meshes = [transform_deploy_mesh(package_mesh, transform) for transform in source_transforms]
    mesh = combine_boundary_meshes(source_meshes).mesh
    q_neumann = deploy_complex_vector(request["boundary_neumann"], FloatType)
    length(q_neumann) == length(mesh.faces) || error(
        "Deploy boundary trace contains $(length(q_neumann)) faces, but the mesh contains $(length(mesh.faces)).",
    )
    all(isfinite, q_neumann) || error("Deploy boundary trace must be finite.")
    reference_pressure = deploy_complex_vector(request["reference_boundary_pressure"], FloatType)
    length(reference_pressure) == length(mesh.vertices) || error(
        "Deploy reference pressure contains $(length(reference_pressure)) nodes, but the mesh contains $(length(mesh.vertices)).",
    )
    observation_points = deploy_observation_points(request, FloatType)
    observation_shape = Int.(get_value(request, "observation_shape", [1, length(observation_points)]))
    length(observation_shape) == 2 || error("Deploy observation_shape must contain rows and columns.")
    observation_sample_indices = Int.(get_value(request, "observation_sample_indices", collect(0:(length(observation_points) - 1))))
    length(observation_sample_indices) == length(observation_points) || error("Deploy observation sample indices do not match the point count.")
    all(index -> 0 <= index < prod(observation_shape), observation_sample_indices) || error("Deploy observation sample index is outside the grid.")
    length(unique(observation_sample_indices)) == length(observation_sample_indices) || error("Deploy observation sample indices must be unique.")

    p1_space = build_p1_space(mesh)
    dp0_space = build_dp0_space(mesh)
    quadrature_order = Int(get_value(request, "quadrature_order", 2))
    singular_order = Int(get_value(request, "singular_order", 3))
    close_pair_quadrature_order = Int(get_value(request, "close_pair_quadrature_order", 8))
    close_pair_quadrature_order >= 4 || error("Deploy close-pair quadrature order must be at least 4.")
    rule = triangle_rule(FloatType, quadrature_order)
    identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1_space, dp0_space, rule, :p1, :p1)
    identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1_space, dp0_space, rule, :p1, :dp0)
    singular_cache = build_singular_correction_cache(mesh, singular_order)
    proximity = get_value(request, "proximity", Dict{String,Any}())
    raw_close_face_pairs = get_value(proximity, "close_face_pairs", Any[])
    close_face_pairs = [
        length(pair) >= 3 ? (Int(pair[1]) + 1, Int(pair[2]) + 1, Int(pair[3])) :
        (Int(pair[1]) + 1, Int(pair[2]) + 1, close_pair_quadrature_order)
        for pair in raw_close_face_pairs
    ]
    near_correction_cache = build_near_correction_cache(
        mesh,
        close_face_pairs,
        close_pair_quadrature_order,
    )
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

    emit_event(
        "initialized";
        backend=String(beat_backend),
        frequency_hz=frequency,
        node_count=length(mesh.vertices),
        face_count=length(mesh.faces),
        source_count=length(source_meshes),
        observation_count=length(observation_points),
    )

    device_cache = nothing
    device_singular_cache = nothing
    device_image_singular_cache = nothing
    device_near_correction_cache = nothing
    device_ground_near_correction_cache = nothing
    cuda_identity_cache = nothing
    field_cache = cpu_field_cache
    operators = nothing
    assembly_seconds = 0.0
    solve_seconds = 0.0
    field_seconds = 0.0
    pressure = nothing
    field_pressure = nothing
    try
        if beat_backend == :cuda
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
            cuda_identity_cache = build_cuda_burton_miller_identity_cache(
                identity_p1_p1,
                identity_p1_dp0,
                FloatType,
            )
            field_cache = build_cuda_field_evaluation_cache(cpu_field_cache)
        else
            emit_event("status"; message="Preparing BEAT CPU geometry caches")
        end

        emit_event("status"; message="Assembling Level 2 rigid half-space boundary operators")
        assembly_seconds = @elapsed begin
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

        emit_event("status"; message="Solving fixed-Neumann exterior system")
        solve_seconds = @elapsed begin
            pressure = if beat_backend == :cuda
                solve_burton_miller_neumann(operators, cuda_identity_cache, q_neumann, k)
            else
                solve_burton_miller_neumann(operators, identity_p1_p1, identity_p1_dp0, q_neumann, k)
            end
        end

        emit_event("status"; message="Evaluating audience plane")
        field_seconds = @elapsed begin
            field_pressure = field_for_points(
                observation_points,
                mesh,
                pressure,
                q_neumann,
                k,
                field_cache,
                beat_backend,
            )
        end
        spl_db = pressure_to_spl(field_pressure, FloatType)
        isolated_trace_relative_difference = Float32(
            norm(pressure - reference_pressure) / max(norm(reference_pressure), eps(FloatType)),
        )
        proximity_pairs = get_value(proximity, "pairs", Any[])
        close_pair_count = count(pair -> Bool(get_value(pair, "close", false)), proximity_pairs)
        emit_event(
            "result";
            result=Dict(
                "frequency_hz" => frequency,
                "rows" => observation_shape[1],
                "columns" => observation_shape[2],
                "spl_db" => spl_db,
                "sample_indices" => observation_sample_indices,
                "field_pressure" => Dict(
                    "real" => Float32.(real.(field_pressure)),
                    "imag" => Float32.(imag.(field_pressure)),
                ),
                "timings" => Dict(
                    "assembly_s" => Float32(assembly_seconds),
                    "solve_s" => Float32(solve_seconds),
                    "field_s" => Float32(field_seconds),
                ),
                "diagnostics" => Dict(
                    "backend" => String(beat_backend),
                    "source_count" => length(source_meshes),
                    "node_count" => length(mesh.vertices),
                    "face_count" => length(mesh.faces),
                    "singular_pair_count" => singular_cache.pair_count,
                    "near_face_pair_count" => near_correction_cache.pair_count,
                    "ground_image_near_face_pair_count" => ground_near_correction_cache.pair_count,
                    "ground_image_singular_pair_count" => operators.image_singular_pairs,
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
                ),
            ),
        )
    finally
        operators === nothing || release_operator_storage!(operators)
        cuda_identity_cache === nothing || release_cuda_burton_miller_identity_cache!(cuda_identity_cache)
        device_image_singular_cache === nothing ||
            release_cuda_image_singular_correction_cache!(device_image_singular_cache)
        device_near_correction_cache === nothing ||
            release_cuda_image_singular_correction_cache!(device_near_correction_cache)
        device_ground_near_correction_cache === nothing ||
            release_cuda_image_singular_correction_cache!(device_ground_near_correction_cache)
    end
    emit_event("completed"; solved_count=1)
end
