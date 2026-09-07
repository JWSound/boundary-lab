using Test, JSON

include(joinpath(@__DIR__, "..", "src", "BeatEngineContract.jl"))
using .BeatEngineContract

const CONTRACT_CORPUS = JSON.parsefile(joinpath(@__DIR__, "..", "..", "beat_contract", "conformance.json"))

@testset "BEAT compiled-system wire conformance" begin
    for case in CONTRACT_CORPUS["cases"]
        @testset "$(case["name"])" begin
            request = deepcopy(CONTRACT_CORPUS["base_request"])
            for change in case["changes"]
                parent = request
                for key in change["path"][1:end-1]
                    parent = parent[key isa Integer ? key + 1 : key]
                end
                key = last(change["path"])
                key = key isa Integer ? key + 1 : key
                if get(change, "remove", false)
                    delete!(parent, key)
                else
                    parent[key] = change["value"]
                end
            end
            if case["valid"]
                @test validate_system_request(request) === nothing
            else
                @test_throws ErrorException validate_system_request(request)
            end
        end
    end
    for value in (NaN, Inf)
        request = deepcopy(CONTRACT_CORPUS["base_request"])
        request["solver_options"]["custom"] = value
        @test_throws ErrorException validate_system_request(request)
    end
end

@testset "BEAT identity without Git" begin
    mktempdir() do directory
        source = joinpath(@__DIR__, "..", "src", "BeatEngineProvenance.jl")
        # Keep the copied engine and contract together in this isolated fixture.
        fixture_root = joinpath(directory, "engine")
        mkpath(joinpath(fixture_root, "src"))
        cp(source, joinpath(fixture_root, "src", "BeatEngineProvenance.jl"))
        mkpath(joinpath(directory, "beat_contract"))
        first_module = Module(:FirstProvenanceFixture)
        Base.include(first_module, joinpath(fixture_root, "src", "BeatEngineProvenance.jl"))
        first = withenv("PATH" => "") do
            Base.invokelatest(() -> first_module.BeatEngineProvenance.engine_identity())
        end
        @test first["repository_revision"] === nothing
        @test first["repository_dirty"] === nothing
        @test length(first["source_sha256"]) == 64
        write(joinpath(fixture_root, "solver.jl"), "# changed engine source\n")
        @test Base.invokelatest(() -> first_module.BeatEngineProvenance.engine_identity())["source_sha256"] == first["source_sha256"]
        second_module = Module(:SecondProvenanceFixture)
        Base.include(second_module, joinpath(fixture_root, "src", "BeatEngineProvenance.jl"))
        second = Base.invokelatest(() -> second_module.BeatEngineProvenance.engine_identity())
        @test second["source_sha256"] != first["source_sha256"]
    end
end

@testset "BEAT worker negotiation" begin
    info = worker_ready(Dict("cpu" => Dict("available" => true, "reason" => "")))
    @test info["protocol"]["version"] == 1
    @test info["contracts"]["system_request"] == [1]
    @test info["contracts"]["compiled_system"] == [1]
    @test info["contracts"]["system_result"] == [2]
    @test info["runtime"]["julia_version"] == string(VERSION)
    @test length(info["engine"]["source_sha256"]) == 64
    @test haskey(info["engine"]["source_files_sha256"], "julia_local/coupled_solver.jl")
    @test haskey(info["engine"], "repository_revision")
    @test haskey(info["engine"], "repository_dirty")
    @test length(info["runtime"]["project_sha256"]) == 64
    @test info["runtime"]["julia_threads"] >= 1
    @test info["runtime"]["blas_threads"] >= 1
    empty!(info["engine"]["source_files_sha256"])
    @test !isempty(worker_ready(Dict())["engine"]["source_files_sha256"])
    command = Dict{String,Any}("protocol_version" => 1, "operation" => "solve",
        "request" => "does-not-exist.json", "result_schema_version" => 2)
    @test validate_worker_submission(command) === nothing
    for value in (nothing, true, 1.0, 2, "1")
        bad = merge(command, Dict("protocol_version" => value))
        @test_throws ErrorException validate_worker_submission(bad)
    end
    for value in (nothing, true, 1, 3)
        @test_throws ErrorException validate_worker_submission(merge(command, Dict("result_schema_version" => value)))
    end
    @test_throws ErrorException validate_worker_submission(merge(command, Dict("operation" => "unknown")))
    @test_throws ErrorException validate_worker_submission(merge(command, Dict("request" => "")))
    field = Dict("protocol_version" => 1, "operation" => "bem_field", "request" => "field.json", "field_array_schema_version" => 1)
    @test validate_worker_submission(field) === nothing
    @test_throws ErrorException validate_worker_submission(merge(field, Dict("field_array_schema_version" => 2)))
end
