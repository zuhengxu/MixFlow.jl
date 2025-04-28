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

include(joinpath(@__DIR__, "../mfvi.jl"))
include(joinpath(@__DIR__, "../Model.jl"))
include(joinpath(@__DIR__, "../plotting.jl"))

function ensemble_sample_eval(
    seed, name::String, flow_length::Int, nchains::Int, kernel_type, step_size; 
    nsample = 1024, leapfrog_steps=20,
    )
    Random.seed!(seed)

    vi_res = JLD2.load(
        joinpath(@__DIR__, "../syn_mfvi_fit/result/$(name)_mfvi.jld2"),
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


function ensemble_tv_sweep(
    seed, name::String, flow_length::Int, nchains::Int, kernel_type, step_size; 
    nsample = 1024, leapfrog_steps=10,
    )
    Random.seed!(seed)

    vi_res = JLD2.load(
        joinpath(@__DIR__, "../syn_mfvi_fit/result/$(name)_mfvi.jld2"),
    )
    prob = vi_res["prob"]

    dims = LogDensityProblems.dimension(prob)
 

    if kernel_type == MF.HMC
        kernel = MF.HMC(leapfrog_steps, step_size)
    else
        kernel =  kernel_type(step_size, ones(dims))
    end
    
    EM = EnsembleRandomShift(dims, flow_length, nchains)

    flow = EnsembleIRFFlow(flow_length, nchains)

    @info "TV eval using $nsample samples on Ensemble Flow length: $flow_length, nchains: $nchains"
    # output = MF.mixflow(flow, prob, kernel, EM, nsample)
    
    # generate samples from the target
    xsπ = rand(prob.target.ℓ, nsample) 
    vsπ = reduce(hcat, [MF._rand_v_given_x(kernel, prob, xsπ[:, i]) for i in 1:size(xsπ, 2)])
    uvπ = rand(dims, nsample)
    uaπ = rand(nsample)
    
    lrs = zeros(nchains, nsample)
    @showprogress @threads for i in 1:nsample
        x = xsπ[:, i]
        v = vsπ[:, i]
        uv = uvπ[:, i]
        ua = uaπ[i]
        lrs[:, i] .= MF.log_density_ratio_flow_sweep_ensemble(flow, prob, kernel, EM, x, v, uv, ua)
    end
    
    tvs = mean(abs.(expm1.(lrs)), dims = 2) ./ 2

    df = DataFrame(
        nchains = [1:nchains ;],
        tv = vec(tvs),
        nparticles = nsample,
    ) 
    return df
end



# fl = 100
# nchains = 10
# name = "Banana"
# kernel = MF.MALA
# stepsize = 0.1

# # df = ensemble_sample_eval(1, name, fl, nchains, kernel, stepsize; nsample = 1024)

# df_tv = ensemble_tv_sweep(1, name, fl, nchains, kernel, stepsize; nsample = 1024, leapfrog_steps=1)
# # # # plot(df_tv.tv)

