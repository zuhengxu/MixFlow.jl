using MixFlow: log_density_flow, _rand_joint_reference
using Random, Distributions, Plots
using LinearAlgebra
using Base.Threads: @threads
using JLD2
using LogExpFunctions
using LogDensityProblems, LogDensityProblemsAD
using MixFlow 
using ADTypes, Mooncake
using NormalizingFlows
using Bijectors
using ProgressMeter

const MF = MixFlow

include("Model.jl")
include("mfvi.jl")

name = "Banana"
target = load_model(name)

ad = AutoMooncake(; config = Mooncake.Config())
target_ad = ADgradient(ad, target)
reference, _ = mfvi(target_ad; sample_per_iter = 10, max_iters = 10000, adtype = ad)
prob = MixFlowProblem(reference, target_ad)

dims = LogDensityProblems.dimension(target_ad)


function elbo_sweep(flowtype, prob, K, mixer, nsample, Ts)
    Els = []
    @showprogress for T in Ts
        F = flowtype(T) 
        el = MF.elbo(F, prob, K, mixer, nsample)
        push!(Els, el)
    end
    return map(identity, Els)
end


# log_density_flow(F, prob, K, mixer, sample...)
# MF._elbo_single(F, prob, K, mixer, sample...)
T_max = 20_000
mixer = RandomShift(2, T_max)
mix_deter = ErgodicShift(2, T_max)

nsample = 500
# T = 10
# F = RandomBackwardMixFlow(T)
# x0, v0, uv0, ua0 = MF._rand_joint_reference(prob, K)
# x, v, uv, ua = simulate_from_past_T_step(prob, K, mixer, x0, v0, uv0, ua0, T)
# sample = iid_sample(F, prob, K, mixer)

Ts = [10, 20, 50, 100, 200, 350, 500]
ϵs = [0.01, 0.05, 0.1]

P = plot()
for ϵ in ϵs
    # ϵ = 0.05
    K = HMC(10, ϵ)
    Ku = uncorrectHMC(10, ϵ)

    @info "ϵ = $ϵ"
    Els_uhmc_deter = elbo_sweep(DeterministicMixFlow, prob, Ku, mix_deter, nsample, Ts)
    Els_hmc = elbo_sweep(BackwardIRFMixFlow, prob, K, mixer, nsample, Ts)
    Els_hmc_deter = elbo_sweep(DeterministicMixFlow, prob, K, mix_deter, nsample, Ts)

    plot!(P, Ts, Els_hmc, label="HMC_bwd_mixflow $(ϵ)", lw=2)
    plot!(P, Ts, Els_hmc_deter, label="HMC_std_mixflow $(ϵ)", lw=2)
    plot!(P, Ts, Els_uhmc_deter, label="uncorrectHMC_std_mixflow $(ϵ)", lw=2)
end

savefig("figure/$(name)_elbo_sweep.png")


using StructArrays

MMixer = StructArray{ErgodicShift}(ξs_uv = [rand(2, 10) for _ in 1:10], ξs_ua = [rand(10) for _ in 1:10])
