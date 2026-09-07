using Test, LinearAlgebra, StaticArrays

function reference_sphere(refinements)
    vertices = SVector{3,Float64}[
        SVector(1.,0.,0.), SVector(-1.,0.,0.), SVector(0.,1.,0.),
        SVector(0.,-1.,0.), SVector(0.,0.,1.), SVector(0.,0.,-1.),
    ]
    faces = [(1,3,5), (3,2,5), (2,4,5), (4,1,5), (3,1,6), (2,3,6), (4,2,6), (1,4,6)]
    for _ in 1:refinements
        midpoints = Dict{Tuple{Int,Int},Int}()
        function midpoint(a, b)
            get!(midpoints, minmax(a,b)) do
                push!(vertices, normalize(vertices[a] + vertices[b]))
                length(vertices)
            end
        end
        refined = NTuple{3,Int}[]
        for (a,b,c) in faces
            ab,bc,ca = midpoint(a,b), midpoint(b,c), midpoint(c,a)
            append!(refined, [(a,ab,ca), (ab,b,bc), (ca,bc,c), (ab,bc,ca)])
        end
        faces = refined
    end
    return BoundaryMesh(vertices, faces, ones(Int, length(faces)))
end

@testset "exterior outgoing monopole complex pressure" begin
    mesh = reference_sphere(2)
    k = 0.7
    exact(x) = exp(im * k * norm(x)) / norm(x)
    # The manufactured field solves Helmholtz everywhere outside the enclosed
    # point source. Its normal derivative is evaluated on each planar facet.
    q = ComplexF64[]
    for (a,b,c) in mesh.faces
        x = (mesh.vertices[a] + mesh.vertices[b] + mesh.vertices[c]) / 3
        normal = normalize(cross(mesh.vertices[b] - mesh.vertices[a], mesh.vertices[c] - mesh.vertices[a]))
        @test dot(normal, x) > 0
        push!(q, (im*k - 1/norm(x)) * exact(x) * dot(normal, normalize(x)))
    end
    p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
    rule = triangle_rule(Float64, 3)
    operators = assemble_regular_galerkin_operators(mesh, p1, dp0, k, rule;
        skip_singular=false, singular_order=3, backend=:cpu)
    identity_pp = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1)
    identity_pq = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0)
    pressure = solve_burton_miller_neumann(operators, identity_pp, identity_pq, q, k)
    points = [SVector(2.,0.,0.), SVector(0.,0.,3.), SVector(-2.,1.,2.)]
    cache = build_field_evaluation_cache(mesh, rule)
    field = evaluate_galerkin_field_cpu(points, mesh, pressure, q, k, cache)
    analytic = exact.(points)
    error = norm(field - analytic) / norm(analytic)
    @info "Analytical outgoing monopole" relative_complex_error=error
    @test error < 0.08
    # Preserve the complex excitation basis and retained-trace field replay.
    amplitude = 0.3 - 0.7im
    replay = evaluate_galerkin_field_cpu(points, mesh, amplitude .* pressure, amplitude .* q, k, cache)
    @test replay ≈ amplitude .* field rtol=1e-12
    @test norm(imag.(field)) > 0

    # A second enclosed point source provides an independent excitation column,
    # rather than merely a rescaled copy of the first one.
    source = SVector(0.2, -0.1, 0.0)
    second_exact(x) = exp(im*k*norm(x-source)) / norm(x-source)
    second_q = ComplexF64[]
    for (a,b,c) in mesh.faces
        x = (mesh.vertices[a] + mesh.vertices[b] + mesh.vertices[c]) / 3
        normal = normalize(cross(mesh.vertices[b] - mesh.vertices[a], mesh.vertices[c] - mesh.vertices[a]))
        delta = x - source
        push!(second_q, (im*k - 1/norm(delta)) * second_exact(x) * dot(normal, normalize(delta)))
    end
    second_pressure = solve_burton_miller_neumann(operators, identity_pp, identity_pq, second_q, k)
    second_field = evaluate_galerkin_field_cpu(points, mesh, second_pressure, second_q, k, cache)
    @test norm(second_field - second_exact.(points)) / norm(second_exact.(points)) < 0.08
    mixed_q = q + amplitude .* second_q
    mixed_pressure = solve_burton_miller_neumann(operators, identity_pp, identity_pq, mixed_q, k)
    mixed_field = evaluate_galerkin_field_cpu(points, mesh, mixed_pressure, mixed_q, k, cache)
    @test mixed_field ≈ field + amplitude .* second_field rtol=1e-12
    @test norm(second_field - field) / norm(field) > 0.01
end

@testset "retained complex fields match explicit symmetry images" begin
    T = Float64
    vertices = [SVector{3,T}(0.2,0.3,0.0), SVector{3,T}(0.4,0.3,0.0), SVector{3,T}(0.2,0.5,0.1)]
    mesh = BoundaryMesh(vertices, [(1,2,3)], [1])
    pressure = ComplexF64[1+0.2im, 0.5-0.3im, 0.7+0.1im]
    q = ComplexF64[0.3-0.2im]
    points = [SVector{3,T}(1,2,3), SVector{3,T}(-1,-2,3)]
    rule = triangle_rule(T, 3)
    for symmetry in (:x, :xy)
        signs = symmetry == :x ? [(1,1,1), (-1,1,1)] : [(1,1,1), (-1,1,1), (1,-1,1), (-1,-1,1)]
        full_vertices = SVector{3,T}[]
        full_faces = NTuple{3,Int}[]
        for sign in signs
            offset = length(full_vertices)
            append!(full_vertices, [SVector{3,T}(sign) .* vertex for vertex in vertices])
            face = prod(sign) == 1 ? (1,2,3) : (1,3,2)
            push!(full_faces, face .+ offset)
        end
        full = BoundaryMesh(full_vertices, full_faces, ones(Int, length(full_faces)))
        reduced_cache = build_field_evaluation_cache(mesh, rule; symmetry_mode=symmetry)
        full_cache = build_field_evaluation_cache(full, rule)
        reduced = evaluate_galerkin_field_cpu(points, mesh, pressure, q, 0.7, reduced_cache)
        explicit = evaluate_galerkin_field_cpu(points, full, repeat(pressure, length(signs)), repeat(q, length(signs)), 0.7, full_cache)
        @test reduced ≈ explicit rtol=1e-12 atol=1e-14
        @test norm(reduced) > 0
    end
end
