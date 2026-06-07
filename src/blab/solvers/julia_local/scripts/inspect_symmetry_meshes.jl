include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))

using .BeatEngineCore

for name in ("sample.msh", "sample_half.msh", "sample_quarter.msh")
    mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", name), Float32(0.001))
    xs = [vertex[1] for vertex in mesh.vertices]
    ys = [vertex[2] for vertex in mesh.vertices]
    tags = sort(collect(Set(mesh.physical_tags)))
    println((
        name=name,
        vertices=length(mesh.vertices),
        faces=length(mesh.faces),
        tags=tags,
        xmin=minimum(xs),
        xmax=maximum(xs),
        ymin=minimum(ys),
        ymax=maximum(ys),
    ))
end
