function _cuda_field_arrays(cache::FieldEvaluationCache{T}) where {T}
    source_count = length(cache.source_points)
    source_points = Matrix{T}(undef, source_count, 3)
    source_normals = Matrix{T}(undef, source_count, 3)
    basis_values = Matrix{T}(undef, source_count, 3)
    source_weights = Vector{T}(undef, source_count)
    source_faces = Matrix{Int32}(undef, source_count, 3)
    source_elements = Vector{Int32}(undef, source_count)

    for source_index in 1:source_count
        point = cache.source_points[source_index]
        normal = cache.source_normals[source_index]
        basis = cache.basis_values[source_index]
        face = cache.source_faces[source_index]
        source_points[source_index, 1] = point[1]
        source_points[source_index, 2] = point[2]
        source_points[source_index, 3] = point[3]
        source_normals[source_index, 1] = normal[1]
        source_normals[source_index, 2] = normal[2]
        source_normals[source_index, 3] = normal[3]
        basis_values[source_index, 1] = basis[1]
        basis_values[source_index, 2] = basis[2]
        basis_values[source_index, 3] = basis[3]
        source_weights[source_index] = cache.source_weights[source_index]
        source_faces[source_index, 1] = Int32(face[1])
        source_faces[source_index, 2] = Int32(face[2])
        source_faces[source_index, 3] = Int32(face[3])
        source_elements[source_index] = Int32(cache.source_elements[source_index])
    end

    return source_points, source_normals, source_weights, source_faces, source_elements, basis_values
end

function build_cuda_field_evaluation_cache(cache::FieldEvaluationCache{T}) where {T<:AbstractFloat}
    CUDA.functional() || error("CUDA field-evaluation cache requested, but CUDA.functional() is false.")
    source_points, source_normals, source_weights, source_faces, source_elements, basis_values = _cuda_field_arrays(cache)
    return CudaFieldEvaluationCache{T}(
        CuArray(source_points),
        CuArray(source_normals),
        CuArray(source_weights),
        CuArray(source_faces),
        CuArray(source_elements),
        CuArray(basis_values),
        length(source_weights),
    )
end

function build_cuda_field_evaluation_cache(mesh::BoundaryMesh{T}, rule::TriangleRule{T}; symmetry_mode::Symbol=:off) where {T<:AbstractFloat}
    return build_cuda_field_evaluation_cache(build_field_evaluation_cache(mesh, rule; symmetry_mode=symmetry_mode))
end

function _cuda_eval_point_arrays(eval_points, ::Type{T}) where {T}
    point_count = length(eval_points)
    points = Matrix{T}(undef, point_count, 3)
    for point_index in 1:point_count
        point = eval_points[point_index]
        points[point_index, 1] = T(point[1])
        points[point_index, 2] = T(point[2])
        points[point_index, 3] = T(point[3])
    end
    return points
end

function _cuda_weighted_field_sources_kernel!(
    pressure_re,
    pressure_im,
    neumann_re,
    neumann_im,
    pressure,
    q_neumann,
    source_weights,
    source_faces,
    source_elements,
    basis_values,
    source_count,
)
    source_index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    while source_index <= source_count
        face1 = source_faces[source_index]
        face2 = source_faces[source_index + source_count]
        face3 = source_faces[source_index + 2 * source_count]
        basis1 = basis_values[source_index]
        basis2 = basis_values[source_index + source_count]
        basis3 = basis_values[source_index + 2 * source_count]
        weight = source_weights[source_index]

        p = (basis1 * pressure[face1] + basis2 * pressure[face2] + basis3 * pressure[face3]) * weight
        q = q_neumann[source_elements[source_index]] * weight
        pressure_re[source_index] = real(p)
        pressure_im[source_index] = imag(p)
        neumann_re[source_index] = real(q)
        neumann_im[source_index] = imag(q)

        source_index += stride
    end

    return nothing
end

function _cuda_field_eval_kernel!(
    pot_re,
    pot_im,
    eval_points,
    source_points,
    source_normals,
    pressure_re,
    pressure_im,
    neumann_re,
    neumann_im,
    k,
    source_count,
    point_count,
)
    point_index = blockIdx().x
    point_index > point_count && return nothing

    tid = threadIdx().x
    threads = blockDim().x
    T = typeof(k)
    four_pi = T(12.566370614359172)
    scratch = CUDA.@cuDynamicSharedMem(T, 2 * threads)
    local_re = zero(T)
    local_im = zero(T)

    x1 = eval_points[point_index]
    x2 = eval_points[point_index + point_count]
    x3 = eval_points[point_index + 2 * point_count]

    source_index = tid
    while source_index <= source_count
        y1 = source_points[source_index]
        y2 = source_points[source_index + source_count]
        y3 = source_points[source_index + 2 * source_count]
        r1 = y1 - x1
        r2 = y2 - x2
        r3 = y3 - x3
        radius2 = r1 * r1 + r2 * r2 + r3 * r3

        if radius2 > zero(T)
            radius = sqrt(radius2)
            phase = k * radius
            green_scale = inv(four_pi * radius)
            green_re = cos(phase) * green_scale
            green_im = sin(phase) * green_scale
            grad_scale_re = -inv(radius)
            grad_scale_im = k
            normal = (
                r1 * source_normals[source_index] +
                r2 * source_normals[source_index + source_count] +
                r3 * source_normals[source_index + 2 * source_count]
            ) / radius
            double_re = (green_re * grad_scale_re - green_im * grad_scale_im) * normal
            double_im = (green_re * grad_scale_im + green_im * grad_scale_re) * normal

            p_re = pressure_re[source_index]
            p_im = pressure_im[source_index]
            q_re = neumann_re[source_index]
            q_im = neumann_im[source_index]

            local_re += double_re * p_re - double_im * p_im - (green_re * q_re - green_im * q_im)
            local_im += double_re * p_im + double_im * p_re - (green_re * q_im + green_im * q_re)
        end

        source_index += threads
    end

    scratch[tid] = local_re
    scratch[tid + threads] = local_im
    sync_threads()

    offset = threads >>> 1
    while offset > 0
        if tid <= offset
            scratch[tid] += scratch[tid + offset]
            scratch[tid + threads] += scratch[tid + threads + offset]
        end
        sync_threads()
        offset >>>= 1
    end

    if tid == 1
        pot_re[point_index] = scratch[1]
        pot_im[point_index] = scratch[threads + 1]
    end

    return nothing
end

function evaluate_galerkin_field_cuda(
    eval_points,
    mesh::BoundaryMesh{T},
    pressure,
    q_neumann,
    k::T,
    cache::CudaFieldEvaluationCache{T};
    return_gpu::Bool=false,
) where {T<:AbstractFloat}
    point_count = length(eval_points)
    point_count == 0 && return return_gpu ? CuArray(Complex{T}[]) : Complex{T}[]
    CUDA.functional() || error("CUDA field evaluation requested, but CUDA.functional() is false.")

    d_eval_points = CuArray(_cuda_eval_point_arrays(eval_points, T))
    pressure_on_gpu = pressure isa CuArray
    neumann_on_gpu = q_neumann isa CuArray
    d_pressure = pressure_on_gpu ? pressure : CuArray(pressure)
    d_q_neumann = neumann_on_gpu ? q_neumann : CuArray(q_neumann)
    d_pressure_re = CUDA.zeros(T, cache.source_count)
    d_pressure_im = CUDA.zeros(T, cache.source_count)
    d_neumann_re = CUDA.zeros(T, cache.source_count)
    d_neumann_im = CUDA.zeros(T, cache.source_count)
    d_pot_re = CUDA.zeros(T, point_count)
    d_pot_im = CUDA.zeros(T, point_count)

    threads = 256
    source_blocks = min(cld(cache.source_count, threads), 65_535)
    CUDA.@cuda threads=threads blocks=source_blocks _cuda_weighted_field_sources_kernel!(
        d_pressure_re,
        d_pressure_im,
        d_neumann_re,
        d_neumann_im,
        d_pressure,
        d_q_neumann,
        cache.source_weights,
        cache.source_faces,
        cache.source_elements,
        cache.basis_values,
        cache.source_count,
    )

    eval_threads = 256
    CUDA.@cuda threads=eval_threads blocks=point_count shmem=2 * eval_threads * sizeof(T) _cuda_field_eval_kernel!(
        d_pot_re,
        d_pot_im,
        d_eval_points,
        cache.source_points,
        cache.source_normals,
        d_pressure_re,
        d_pressure_im,
        d_neumann_re,
        d_neumann_im,
        k,
        cache.source_count,
        point_count,
    )
    CUDA.synchronize()

    d_pot = complex.(d_pot_re, d_pot_im)
    result = return_gpu ? d_pot : Complex{T}.(Array(d_pot))

    CUDA.unsafe_free!(d_eval_points)
    pressure_on_gpu || CUDA.unsafe_free!(d_pressure)
    neumann_on_gpu || CUDA.unsafe_free!(d_q_neumann)
    CUDA.unsafe_free!(d_pressure_re)
    CUDA.unsafe_free!(d_pressure_im)
    CUDA.unsafe_free!(d_neumann_re)
    CUDA.unsafe_free!(d_neumann_im)
    CUDA.unsafe_free!(d_pot_re)
    CUDA.unsafe_free!(d_pot_im)
    return_gpu || CUDA.unsafe_free!(d_pot)

    return result
end
