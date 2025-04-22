using ProgressMeter
using Random, Distributions, Plots
using LinearAlgebra
using Base.Threads: @threads
using LogExpFunctions
using LogDensityProblems, LogDensityProblemsAD
using ADTypes, Mooncake
using DataFrames, CSV
using JLD2

using MixFlow 
const MF = MixFlow

include("Model.jl")
include("mfvi.jl")
include("utils.jl")

include(joinpath(@__DIR__, "plotting.jl"))

function run_elbo(
    seed, name::String, flowtype, T::Int, kernel_type, step_size; 
    nsample = 1024, leapfrog_steps=50,
    )

    flow = flowtype(T)
    if (kernel_type == MF.uncorrectHMC) && !(flow isa MF.DeterministicMixFlow)
        println("no this combo")
        return DataFrame( nchains = NaN, logZ = NaN, elbo = NaN, ess = NaN, nparticles = NaN) 
    end

    Random.seed!(seed)

    vi_res = JLD2.load(
        joinpath(@__DIR__, "syn_mfvi_fit/result/$(name)_mfvi.jld2"),
    )
    prob = vi_res["prob"]

    dims = LogDensityProblems.dimension(prob)

    if flow isa MF.DeterministicMixFlow
        mixer = ErgodicShift(dims, T)
    else
        mixer = RandomShift(dims, T)
    end

    if kernel_type == MF.HMC
        kernel = MF.HMC(leapfrog_steps, step_size)
    elseif kernel_type == MF.uncorrectHMC
        kernel = MF.uncorrectHMC(leapfrog_steps, step_size)
    else 
        kernel =  kernel_type(step_size, ones(dims))
    end

    output = MF.mixflow(flow, prob, kernel, mixer, nsample)
    
    df = DataFrame(
        nchains = 1, 
        logZ = output.logZ,
        elbo = output.elbo,
        ess = output.ess,
        nparticles = nsample,
    )
    return df
end

# df = run_elbo(1, "Banana", MF.DeterministicMixFlow, 10, MF.uncorrectHMC, 0.05; nsample = 512)