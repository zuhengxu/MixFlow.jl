using Random, Distributions
using LinearAlgebra
using Base.Threads: @threads
using LogExpFunctions
using LogDensityProblems, LogDensityProblemsAD
using MixFlow
using DataFrames, CSV
# using JLD2

const MF = MixFlow

function _get_kernel_name(K::MF.InvolutiveKernel)
    str = string(typeof(K))
    name = split(str, "{", limit = 2)[1]
    return name
end

