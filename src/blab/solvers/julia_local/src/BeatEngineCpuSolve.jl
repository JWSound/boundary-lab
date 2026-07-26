const BEAT_CPU_BLAS_THREADS_ENV = "BLAB_BEAT_CPU_BLAS_THREADS"

function beat_cpu_blas_thread_count(
    p1_dofs::Integer;
    available_threads::Integer=Threads.nthreads(),
    override::AbstractString=get(ENV, BEAT_CPU_BLAS_THREADS_ENV, "auto"),
)
    dof_count = max(1, Int(p1_dofs))
    thread_limit = max(1, Int(available_threads))
    override_text = lowercase(strip(override))
    if !isempty(override_text) && override_text != "auto"
        requested = try
            parse(Int, override_text)
        catch
            error("$BEAT_CPU_BLAS_THREADS_ENV must be 'auto' or a positive integer; got $(repr(override)).")
        end
        requested > 0 || error("$BEAT_CPU_BLAS_THREADS_ENV must be a positive integer; got $requested.")
        return min(requested, thread_limit)
    end

    if dof_count <= 768
        return 1
    elseif dof_count <= 2048
        return min(4, thread_limit)
    elseif dof_count <= 4096
        return min(8, thread_limit)
    end
    return thread_limit
end

function configure_beat_cpu_blas_threads!(
    p1_dofs::Integer;
    available_threads::Integer=Threads.nthreads(),
    override::AbstractString=get(ENV, BEAT_CPU_BLAS_THREADS_ENV, "auto"),
)
    thread_count = beat_cpu_blas_thread_count(
        p1_dofs;
        available_threads=available_threads,
        override=override,
    )
    BLAS.set_num_threads(thread_count)
    return BLAS.get_num_threads()
end

function burton_miller_neumann_matrices(operators, identity_p1_p1, identity_p1_dp0, k::T) where {T<:AbstractFloat}
    coupling = Complex{T}(0, 1) / k
    identity_p1_p1_complex = Complex{T}.(identity_p1_p1)
    identity_p1_dp0_complex = Complex{T}.(identity_p1_dp0)

    lhs = Complex{T}(0.5) .* identity_p1_p1_complex .- operators.double_layer .+ coupling .* operators.hypersingular
    rhs_operator = -operators.single_layer .- coupling .* (operators.adjoint_double_layer .+ Complex{T}(0.5) .* identity_p1_dp0_complex)
    return lhs, rhs_operator
end

function build_burton_miller_neumann_cpu_system(operators, identity_p1_p1, identity_p1_dp0, k::T) where {T<:AbstractFloat}
    lhs, rhs_operator = burton_miller_neumann_matrices(
        operators,
        identity_p1_p1,
        identity_p1_dp0,
        k,
    )
    return (
        factorization=lu!(lhs),
        rhs_operator=rhs_operator,
    )
end

function solve_burton_miller_neumann_cpu_system(system, q_neumann, ::Type{T}) where {T<:AbstractFloat}
    rhs = system.rhs_operator * Complex{T}.(q_neumann)
    return Complex{T}.(system.factorization \ rhs)
end

function solve_burton_miller_neumann_cpu(operators, identity_p1_p1, identity_p1_dp0, q_neumann, k::T) where {T<:AbstractFloat}
    system = build_burton_miller_neumann_cpu_system(operators, identity_p1_p1, identity_p1_dp0, k)
    return solve_burton_miller_neumann_cpu_system(system, q_neumann, T)
end
