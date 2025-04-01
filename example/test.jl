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
using MixFlow: _rand_joint_reference 

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



# log_density_flow(F, prob, K, mixer, sample...)
# MF._elbo_single(F, prob, K, mixer, sample...)
T_max = 8_000
mixer = RandomShift(2, T_max)
mix_deter = ErgodicShift(2, T_max)

###############
# generating trajectories
###############
kernel = HMC(10, 0.02) 
x0, v0, uv0, ua0 = MF._rand_joint_reference(prob, kernel)
x_traj_fwd = MF.forward_trajectory_x(prob, kernel, mixer, x0, v0, uv0, ua0, T_max)
x_traj_fwd_homo = MF.forward_trajectory_x(prob, kernel, mix_deter, x0, v0, uv0, ua0, T_max) 
x_traj_bwd = MF.backward_process_trajectory_x(prob, kernel, mixer, x0, v0, uv0, ua0, T_max)






#################
# elbo sweep
################
function elbo_sweep(flowtype, prob, K, mixer, nsample, Ts)
    Els = []
    @showprogress for T in Ts
        F = flowtype(T) 
        el = MF.elbo(F, prob, K, mixer, nsample)
        push!(Els, el)
    end
    return map(identity, Els)
end

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
