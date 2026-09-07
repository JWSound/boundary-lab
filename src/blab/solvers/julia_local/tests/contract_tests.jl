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
