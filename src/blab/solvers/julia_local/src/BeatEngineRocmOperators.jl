function release_operator_storage!(operators::NamedTuple)
    get(operators, :on_gpu, false) || return nothing
    get(operators, :gpu_backend, nothing) == :rocm || return nothing
    AMDGPU.unsafe_free!(operators.single_layer)
    AMDGPU.unsafe_free!(operators.double_layer)
    AMDGPU.unsafe_free!(operators.adjoint_double_layer)
    AMDGPU.unsafe_free!(operators.hypersingular)
    return nothing
end

function _apply_rocm_operator_p1_row_weights!(operators, mesh::BoundaryMesh{T}, symmetry_mode) where {T<:AbstractFloat}
    normalized_symmetry_mode(symmetry_mode) == :off && return nothing
    d_weights = AMDGPU.ROCArray(Complex{T}.(p1_symmetry_orbit_weights(mesh, symmetry_mode)))
    operators.single_layer .*= reshape(d_weights, :, 1)
    operators.double_layer .*= reshape(d_weights, :, 1)
    operators.adjoint_double_layer .*= reshape(d_weights, :, 1)
    operators.hypersingular .*= reshape(d_weights, :, 1)
    AMDGPU.synchronize()
    AMDGPU.unsafe_free!(d_weights)
    return nothing
end

function build_rocm_burton_miller_identity_cache(identity_p1_p1, identity_p1_dp0, ::Type{T}) where {T<:AbstractFloat}
    _require_rocm!(rocsolver=true)
    return RocmBurtonMillerIdentityCache(
        AMDGPU.ROCArray(Complex{T}.(identity_p1_p1)),
        AMDGPU.ROCArray(Complex{T}.(identity_p1_dp0)),
    )
end

function release_rocm_burton_miller_identity_cache!(cache::RocmBurtonMillerIdentityCache)
    AMDGPU.unsafe_free!(cache.identity_p1_p1)
    AMDGPU.unsafe_free!(cache.identity_p1_dp0)
    return nothing
end

function _rocm_burton_miller_rhs(
    operators,
    identity_cache::RocmBurtonMillerIdentityCache,
    d_q_neumann,
    coupling::Complex{T},
) where {T<:AbstractFloat}
    d_rhs = similar(d_q_neumann, size(operators.single_layer, 1))
    mul!(d_rhs, operators.single_layer, d_q_neumann, -one(Complex{T}), zero(Complex{T}))
    mul!(d_rhs, operators.adjoint_double_layer, d_q_neumann, -coupling, one(Complex{T}))
    mul!(d_rhs, identity_cache.identity_p1_dp0, d_q_neumann, -T(0.5) * coupling, one(Complex{T}))
    return d_rhs
end

function _is_corrupt_rocsolver_info(error)
    error isa ArgumentError || return false
    matched = match(r"invalid argument #(\d+) to LAPACK call", sprint(showerror, error))
    matched === nothing && return false
    # getrf has only a handful of parameters. A large negative `info` value is
    # therefore uninitialized/corrupt device status, not a caller error.
    return parse(Int, matched.captures[1]) > 16
end

function _rocm_dense_solve(d_lhs, d_rhs; max_attempts::Int=3)
    max_attempts >= 1 || throw(ArgumentError("max_attempts must be positive."))
    for attempt in 1:max_attempts
        try
            # rocSOLVER's `info` is device-resident. Keep preceding assembly and
            # allocator work out of the status read that AMDGPU.jl performs.
            AMDGPU.synchronize()
            result = d_lhs \ d_rhs
            AMDGPU.synchronize()
            return result
        catch exception
            if attempt == max_attempts || !_is_corrupt_rocsolver_info(exception)
                rethrow()
            end
            @warn "rocSOLVER returned a corrupt LAPACK info value; retrying the dense solve." attempt exception=(exception, catch_backtrace())
            try
                AMDGPU.synchronize()
            catch
            end
            GC.gc(true)
            try
                AMDGPU.reclaim()
            catch
            end
        end
    end
    error("Unreachable ROCm dense-solve retry state.")
end

function solve_burton_miller_neumann(
    operators,
    identity_cache::RocmBurtonMillerIdentityCache,
    q_neumann,
    k::T,
) where {T<:AbstractFloat}
    get(operators, :on_gpu, false) || error("Cached ROCm solve requires GPU-resident operators.")
    get(operators, :gpu_backend, nothing) == :rocm || error("Cached ROCm solve requires ROCm operators.")
    _require_rocm!(rocsolver=true)

    coupling = Complex{T}(0, 1) / k
    d_q_neumann = d_lhs = d_rhs = d_pressure = nothing
    pressure = nothing
    try
        d_q_neumann = AMDGPU.ROCArray(Complex{T}.(q_neumann))
        d_lhs = Complex{T}(0.5) .* identity_cache.identity_p1_p1 .-
            operators.double_layer .+ coupling .* operators.hypersingular
        d_rhs = _rocm_burton_miller_rhs(operators, identity_cache, d_q_neumann, coupling)
        d_pressure = _rocm_dense_solve(d_lhs, d_rhs)
        pressure = Complex{T}.(Array(d_pressure))
    finally
        for item in (d_q_neumann, d_lhs, d_rhs, d_pressure)
            item === nothing && continue
            AMDGPU.unsafe_free!(item)
        end
    end
    return pressure
end
