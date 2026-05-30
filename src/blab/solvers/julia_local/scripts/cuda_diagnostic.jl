println("Julia version: ", VERSION)

try
    import CUDA
    println("CUDA.jl loaded: yes")
    println("CUDA.functional(): ", CUDA.functional())
    println("CUDA.has_cuda(): ", CUDA.has_cuda())

    try
        println("CUDA runtime version: ", CUDA.runtime_version())
    catch err
        println("CUDA runtime version: unavailable (", typeof(err), ": ", err, ")")
    end

    try
        println("CUDA driver version: ", CUDA.driver_version())
    catch err
        println("CUDA driver version: unavailable (", typeof(err), ": ", err, ")")
    end

    try
        devices = collect(CUDA.devices())
        println("CUDA device count: ", length(devices))
        for (i, device) in enumerate(devices)
            println("CUDA device ", i, ": ", device)
        end
    catch err
        println("CUDA devices unavailable (", typeof(err), ": ", err, ")")
    end
catch err
    println("CUDA.jl loaded: no")
    println("Import error: ", typeof(err), ": ", err)
    showerror(stdout, err, catch_backtrace())
    println()
end

try
    cuda = Base.require(Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))
    println("Base.require CUDA.functional(): ", cuda.functional())
catch err
    println("Base.require CUDA failed: ", typeof(err), ": ", err)
end
