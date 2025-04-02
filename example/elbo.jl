using Random, Distributions, Plots
using LinearAlgebra
using Base.Threads: @threads
using LogExpFunctions
using LogDensityProblems, LogDensityProblemsAD
using ADTypes, Mooncake
using DataFrames, CSV

using MixFlow 
const MF = MixFlow

include("Model.jl")
include("mfvi.jl")
include("utils.jl")

function run_elbo(seed, name, flowtype, kernel_type, step_size, nsample, T; leapfrog_steps=50)
    # name = "Banana"
    Random.seed!(seed)

    target = load_model(name)

    ad = AutoMooncake(; config = Mooncake.Config())
    target_ad = ADgradient(ad, target)
    reference, _ = mfvi(target_ad; sample_per_iter = 10, max_iters = 10000, adtype = ad)
    prob = MixFlowProblem(reference, target_ad)

    dims = LogDensityProblems.dimension(target_ad)

    if flowtype isa MF.DeterministicMixFlow
        mixer = ErgodicShift(dims, T)
    else
        mixer = RandomShift(dims, T)
    end

    if kernel_type == MF.HMC
        kernel = MF.HMC(leapfrog_steps, step_size)
    else 
        kernel =  kernel_type(step_size, ones(dims))
    end

    flow = flowtype(T)
    output = MF.mixflow(flow,  prob, kernel, mixer, nsample)
    
    df = DataFrame(
        flow = string(flowtype),
        nchains = 1, 
        logZ = output.logZ,
        elbo = output.elbo,
        ess = output.ess,
        nparticles = nsample,
    )
    return df
end

# df = run_elbo(1, "Banana", MF.BackwardIRFMixFlow, MF.HMC, 0.05, 32, 10)

# nsample = 1024

# Ts = [10, 20, 50, 100, 200, 350, 500]
# ϵs = [0.01, 0.05, 0.1]

# P = plot()
# for ϵ in ϵs
#     # ϵ = 0.05
#     K = HMC(10, ϵ)
#     Ku = uncorrectHMC(10, ϵ)

#     @info "ϵ = $ϵ"
#     Els_uhmc_deter = elbo_sweep(DeterministicMixFlow, prob, Ku, mix_deter, nsample, Ts)
#     Els_hmc = elbo_sweep(BackwardIRFMixFlow, prob, K, mixer, nsample, Ts)
#     Els_hmc_deter = elbo_sweep(DeterministicMixFlow, prob, K, mix_deter, nsample, Ts)

#     plot!(P, Ts, Els_hmc, label="HMC_bwd_mixflow $(ϵ)", lw=2)
#     plot!(P, Ts, Els_hmc_deter, label="HMC_std_mixflow $(ϵ)", lw=2)
#     plot!(P, Ts, Els_uhmc_deter, label="uncorrectHMC_std_mixflow $(ϵ)", lw=2)
# end

# savefig("figure/$(name)_elbo_sweep.png")


# using StructArrays

# MMixer = StructArray{ErgodicShift}(ξs_uv = [rand(2, 10) for _ in 1:10], ξs_ua = [rand(10) for _ in 1:10])
