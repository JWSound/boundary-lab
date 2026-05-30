using Pkg

Pkg.status(["CUDA", "CUDA_Runtime_jll"])

try
    using CUDA
    CUDA.versioninfo()
catch err
    showerror(stdout, err)
    println()
    for frame in stacktrace(catch_backtrace())[1:min(end, 12)]
        println(frame)
    end
end
