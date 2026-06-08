function evaluate_galerkin_field_cpu(
    eval_points,
    mesh::BoundaryMesh{T},
    pressure,
    q_neumann,
    k::T,
    cache::FieldEvaluationCache{T},
) where {T<:AbstractFloat}
    point_count = length(eval_points)
    point_count == 0 && return Complex{T}[]

    potentials = Vector{Complex{T}}(undef, point_count)
    source_count = length(cache.source_points)

    Threads.@threads for point_index in 1:point_count
        x = SVector{3,T}(T(eval_points[point_index][1]), T(eval_points[point_index][2]), T(eval_points[point_index][3]))
        potential = zero(Complex{T})
        for source_index in 1:source_count
            y = cache.source_points[source_index]
            r_vec = y - x
            radius = norm(r_vec)
            radius == zero(T) && continue

            phase = k * radius
            green_scale = inv(T(4.0) * T(pi) * radius)
            green = Complex{T}(cos(phase) * green_scale, sin(phase) * green_scale)
            normal_projection = dot(r_vec, cache.source_normals[source_index]) / radius
            double_layer = green * (Complex{T}(0, k) - inv(radius)) * normal_projection

            face = cache.source_faces[source_index]
            basis = cache.basis_values[source_index]
            weight = cache.source_weights[source_index]
            p_source = (
                basis[1] * pressure[face[1]] +
                basis[2] * pressure[face[2]] +
                basis[3] * pressure[face[3]]
            ) * weight
            q_source = q_neumann[cache.source_elements[source_index]] * weight
            potential += double_layer * p_source - green * q_source
        end
        potentials[point_index] = potential
    end

    return potentials
end
