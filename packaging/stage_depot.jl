# Pinned to Julia 1.12.6: collect_artifacts is a Pkg internal API.
using Pkg, TOML, SHA
target = abspath(ARGS[1])
mkpath(target)
cuda_runtime = string(VersionNumber(ARGS[2]).major, ".", VersionNumber(ARGS[2]).minor)
records = Dict{String, Any}("packages" => Dict(), "artifacts" => Dict())
for project in ARGS[3:end]
    if basename(project) == "julia_cuda"
        # Preferences on transitive dependencies require UUID registration. Without
        # these extras a GPU-less build host silently selects no CUDA artifacts.
        project_file = joinpath(project, "Project.toml")
        project_data = TOML.parsefile(project_file)
        extras = get!(project_data, "extras", Dict{String, Any}())
        extras["CUDA_Runtime_jll"] = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
        extras["CUDA_Compiler_jll"] = "d1e2174e-dfdc-576e-b43e-73b79eb1aca8"
        open(project_file, "w") do io
            TOML.print(io, project_data; sorted=true)
        end
        # The pinned CUDSS selector expects the string form of the local flag.
        preferences = Dict(name => Dict("version" => cuda_runtime, "local" => "false")
                           for name in ("CUDA_Runtime_jll", "CUDA_Compiler_jll"))
        open(joinpath(project, "LocalPreferences.toml"), "w") do io
            TOML.print(io, preferences; sorted=true)
        end
    end
    Pkg.activate(project)
    Pkg.instantiate(; allow_autoprecomp=false)
    for (uuid, info) in Pkg.dependencies()
        source = info.source
        source === nothing && continue
        source = abspath(source)
        # Standard libraries ship with Julia. Local engine bundles ship in its wheel.
        in_depot = any(d -> startswith(source, joinpath(abspath(d), "packages") * string(Base.Filesystem.path_separator)), DEPOT_PATH)
        in_depot || continue
        destination = joinpath(target, "packages", info.name, basename(source))
        if !isdir(destination)
            mkpath(dirname(destination))
            cp(source, destination)
        end
        records["packages"][string(uuid)] = Dict("name" => info.name, "version" => string(info.version), "tree" => string(info.tree_hash))
        selected = Pkg.Operations.collect_artifacts(source; include_lazy=true)
        if info.name in ("CUDA_Runtime_jll", "CUDA_Compiler_jll") &&
           !any(!isempty(artifacts) for (_, artifacts) in selected)
            error("No artifacts selected for $(info.name); refusing an incomplete CUDA bundle")
        end
        for (toml, artifacts) in selected
            for (name, meta) in artifacts
                haskey(meta, "git-tree-sha1") || continue
                Pkg.Artifacts.ensure_artifact_installed(name, meta, toml)
                hash = meta["git-tree-sha1"]
                location = Pkg.Artifacts.artifact_path(Base.SHA1(hash))
                destination = joinpath(target, "artifacts", hash)
                if !isdir(destination)
                    println("Staging artifact ", name, " ", hash)
                    mkpath(dirname(destination))
                    cp(location, destination)
                end
                records["artifacts"][hash] = name
            end
        end
    end
end
open(joinpath(target, "inventory.toml"), "w") do io
    TOML.print(io, records; sorted=true)
end
