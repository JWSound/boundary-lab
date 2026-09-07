using Test, JSON, SHA

@testset "frozen numerical fixture integrity" begin
    directory = joinpath(@__DIR__, "fixtures")
    manifest = JSON.parsefile(joinpath(directory, "manifest.json"))
    @test manifest["schema_version"] == 1
    for fixture in manifest["fixtures"]
        @test basename(fixture["file"]) == fixture["file"]
        @test bytes2hex(open(sha256, joinpath(directory, fixture["file"]))) == fixture["sha256"]
    end
end
