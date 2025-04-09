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

function run_ensemble(
    seed, name::String, flow_length::Int, nchains::Int, kernel_type, step_size; 
    nsample = 1024, leapfrog_steps=50,
    )
    Random.seed!(seed)

    vi_res = JLD2.load(
        joinpath(@__DIR__, "result/$(name)_mfvi.jld2"),
    )
    prob = vi_res["prob"]
    dims = LogDensityProblems.dimension(prob)
 

    if kernel_type == MF.HMC
        kernel = MF.HMC(leapfrog_steps, step_size)
    else
        kernel =  kernel_type(step_size, ones(dims))
    end
    
    # flow_length = div(total_cost, nchains)
    EM = EnsembleRandomShift(dims, flow_length, nchains)

    flow = EnsembleIRFFlow(flow_length, nchains)
    output = MF.mixflow(flow, prob, kernel, EM, nsample)
    
    df = DataFrame(
        logZ = output.logZ,
        elbo = output.elbo,
        ess = output.ess,
        nparticles = nsample,
    )
    return df
end

# df = run_ensemble(1, "Banana", 10, 5, MF.HMC, 0.1; nsample = 32)

