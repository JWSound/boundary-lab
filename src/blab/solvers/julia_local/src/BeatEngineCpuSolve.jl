function build_burton_miller_neumann_cpu_system(operators, identity_p1_p1, identity_p1_dp0, k::T) where {T<:AbstractFloat}
    coupling = Complex{T}(0, 1) / k
    identity_p1_p1_complex = Complex{T}.(identity_p1_p1)
    identity_p1_dp0_complex = Complex{T}.(identity_p1_dp0)

    lhs = Complex{T}(0.5) .* identity_p1_p1_complex .- operators.double_layer .+ coupling .* operators.hypersingular
    rhs_operator = -operators.single_layer .- coupling .* (operators.adjoint_double_layer .+ Complex{T}(0.5) .* identity_p1_dp0_complex)

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
