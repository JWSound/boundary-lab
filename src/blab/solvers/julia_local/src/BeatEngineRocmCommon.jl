function _rocm_not_implemented(feature::AbstractString)
    error("BEAT Engine ROCm $(feature) is not implemented yet.")
end
