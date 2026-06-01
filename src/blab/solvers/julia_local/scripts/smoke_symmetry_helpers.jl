include(joinpath(@__DIR__, "..", "src", "JBEMCore.jl"))

using .JBEMCore
using StaticArrays

x_image = only(symmetry_image_transforms("x"))
xy_images = symmetry_image_transforms(:xy)

@assert x_image.label == :x
@assert x_image.signs == SVector{3,Int}(-1, 1, 1)
@assert x_image.determinant == -1
@assert length(xy_images) == 3
@assert symmetry_reduction_factor("off") == 1
@assert symmetry_reduction_factor("x") == 2
@assert symmetry_reduction_factor("xy") == 4

vector = SVector{3,Float64}(1.0, 2.0, 3.0)
@assert reflect_point(x_image, vector) == SVector{3,Float64}(-1.0, 2.0, 3.0)
@assert reflect_normal(x_image, vector) == SVector{3,Float64}(-1.0, 2.0, 3.0)
@assert reflect_curl(x_image, vector) == SVector{3,Float64}(1.0, -2.0, -3.0)

xy_image = xy_images[3]
@assert xy_image.label == :xy
@assert reflect_point(xy_image, vector) == SVector{3,Float64}(-1.0, -2.0, 3.0)
@assert reflect_curl(xy_image, vector) == SVector{3,Float64}(-1.0, -2.0, 3.0)

valid_mesh = BoundaryMesh(
    [SVector{3,Float64}(0.0, 0.0, 0.0), SVector{3,Float64}(1.0, 0.0, 0.0), SVector{3,Float64}(0.0, 1.0, 0.0)],
    [(1, 2, 3)],
    [2],
)
validate_symmetry_fundamental_domain!(valid_mesh, "xy")
@assert p1_symmetry_orbit_weights(valid_mesh, "xy") == Float64[4.0, 2.0, 2.0]

invalid_mesh = BoundaryMesh(
    [SVector{3,Float64}(-0.1, 0.0, 0.0), SVector{3,Float64}(1.0, 0.0, 0.0), SVector{3,Float64}(0.0, 1.0, 0.0)],
    [(1, 2, 3)],
    [2],
)
@assert try
    validate_symmetry_fundamental_domain!(invalid_mesh, "x")
    false
catch err
    occursin("positive X fundamental domain", sprint(showerror, err))
end

println("symmetry helpers ok")
