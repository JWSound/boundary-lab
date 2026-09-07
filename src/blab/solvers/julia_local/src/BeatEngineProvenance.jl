module BeatEngineProvenance

using SHA, LinearAlgebra

export engine_identity, runtime_identity

const ENGINE_ROOT = normpath(joinpath(@__DIR__, ".."))
const IDENTITY = Ref{Any}(nothing)
const RUNTIME = Ref{Any}(nothing)

file_hash(path) = path !== nothing && isfile(path) ? open(sha256, path) |> bytes2hex : nothing

function git_value(args)
    try
        return strip(read(pipeline(`git -C $ENGINE_ROOT $args`, stderr=devnull), String))
    catch
        return nothing
    end
end

function engine_identity()
    if IDENTITY[] === nothing
        files = Dict{String,String}()
        for (label, directory) in (("julia_local", ENGINE_ROOT), ("beat_contract", joinpath(ENGINE_ROOT, "..", "beat_contract")))
            for (root, _, names) in walkdir(directory)
                relative = replace(relpath(root, directory), '\\' => '/')
                (relative == "." || label == "julia_local" && (relative == "src" || startswith(relative, "src/"))) || continue
                for name in names
                    (endswith(name, ".jl") || label == "beat_contract" && (endswith(name, ".json") || endswith(name, ".py"))) || continue
                    path = joinpath(root, name)
                    files["$label/" * replace(relpath(path, directory), '\\' => '/')] = file_hash(path)
                end
            end
        end
        digest_input = join(["$name\0$(files[name])\n" for name in sort!(collect(keys(files)))])
        revision = git_value(["rev-parse", "HEAD"])
        status = revision === nothing ? nothing : git_value(["status", "--porcelain", "--untracked-files=all", "--", ".", "../beat_contract"])
        IDENTITY[] = Dict("repository_revision" => revision,
            "repository_dirty" => status === nothing ? nothing : !isempty(status),
            "source_sha256" => bytes2hex(sha256(digest_input)), "source_files_sha256" => files)
    end
    return deepcopy(IDENTITY[])
end

function runtime_identity()
    if RUNTIME[] !== nothing
        return merge(deepcopy(RUNTIME[]), Dict("blas_threads" => BLAS.get_num_threads()))
    end
    project = Base.active_project()
    directory = project === nothing ? nothing : dirname(project)
    manifest = directory === nothing ? nothing : joinpath(directory, "Manifest-v$(VERSION.major).$(VERSION.minor).toml")
    if manifest !== nothing && !isfile(manifest)
        manifest = joinpath(directory, "Manifest.toml")
    end
    image_pointer = Base.JLOptions().image_file
    image = image_pointer == C_NULL ? nothing : unsafe_string(image_pointer)
    RUNTIME[] = Dict{String,Any}(
        "julia_version" => string(VERSION), "executable" => joinpath(Sys.BINDIR, Base.julia_exename()),
        "project_file" => project, "project_sha256" => file_hash(project),
        "manifest_file" => manifest !== nothing && isfile(manifest) ? manifest : nothing,
        "manifest_sha256" => file_hash(manifest), "julia_threads" => Threads.nthreads(),
        "sysimage_file" => image, "sysimage_sha256" => file_hash(image),
        "blas_threads" => BLAS.get_num_threads(), "blas_config" => string(BLAS.get_config()),
        "machine" => Sys.MACHINE, "kernel" => string(Sys.KERNEL), "cpu_name" => Sys.CPU_NAME,
    )
    return deepcopy(RUNTIME[])
end

end
