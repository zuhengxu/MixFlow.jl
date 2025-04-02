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

function running_mean(xs::Matrix{T}) where T
    cumsum(xs; dims = 2) ./ [1:size(xs, 2) ;]'
end

function running_square(xs::Matrix{T}) where T
    cumsum(xs.^2, dims = 2) ./ [1: size(xs, 2) ;]'
end

