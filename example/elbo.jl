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

function run_elbo(
    seed, name::String, flowtype, T::Int, kernel_type, step_size; 
    nsample = 1024, leapfrog_steps=50,
    )

    flow = flowtype(T)
    if (kernel_type == MF.uncorrectHMC) && !(flow isa MF.DeterministicMixFlow)
        println("no this combo")
        return DataFrame( nchains = NaN, logZ = NaN, elbo = NaN, ess = NaN, nparticles = NaN) 
    end

    # name = "Banana"
    Random.seed!(seed)

    vi_res = JLD2.load(
        joinpath(@__DIR__, "result/$(name)_mfvi.jld2"),
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


function run_tv(
    seed, name::String, flowtype, T::Int, kernel_type, step_size; 
    nsample = 512, leapfrog_steps=50,
) 
    flow = flowtype(T)

    Random.seed!(seed)

    vi_res = JLD2.load(
        joinpath(@__DIR__, "result/$(name)_mfvi.jld2"),
    )
    prob = vi_res["prob"]

    dims = LogDensityProblems.dimension(prob)
    mixer = ErgodicShift(dims, T)
        
    if kernel_type == MF.HMC
        kernel = MF.HMC(leapfrog_steps, step_size)
    elseif kernel_type == MF.uncorrectHMC
        kernel = MF.uncorrectHMC(leapfrog_steps, step_size)
    else 
        kernel =  kernel_type(step_size, ones(dims))
    end

    xsπ = rand(prob.target.ℓ, nsample) 
    vsπ = reduce(hcat, [MF._rand_v_given_x(kernel, prob, x) for x in eachcol(xsπ)])
    uvπ = rand(dims, nsample)
    uaπ = rand(nsample)
    
    lrs = zeros(T+1, nsample)
    @showprogress @threads for i in 1:nsample
        x = xsπ[:, i]
        v = vsπ[:, i]
        uv = kernel_type == MF.uncorrectHMC ? nothing : uvπ[:, i]
        ua = kernel_type == MF.uncorrectHMC ? nothing : uaπ[i]
        lrs[:, i] .= MF.log_density_ratio_flow_sweep(flow, prob, kernel, mixer, x, v, uv, ua)
    end
    
    tvs = mean(abs.(expm1.(lrs)), dims = 2) ./ 2

    df = DataFrame(
        tv = vec(tvs),
        Ts = [1:T+1 ;],
        nparticles = nsample,
    ) 
    return df
end

# tv_uhmc = run_tv(1, "Funnel", MF.DeterministicMixFlow, 200, MF.uncorrectHMC, 0.1; nsample = 512)
# tv_hmc = run_tv(1, "Funnel", MF.DeterministicMixFlow, 30, MF.HMC, 0.1; nsample = 512)
