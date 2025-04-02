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
include("utils.jl")

name = "Banana"
target = load_model(name)

ad = AutoMooncake(; config = Mooncake.Config())
target_ad = ADgradient(ad, target)
reference, _ = mfvi(target_ad; sample_per_iter = 10, max_iters = 10000, adtype = ad)
prob = MixFlowProblem(reference, target_ad)

dims = LogDensityProblems.dimension(target_ad)

samples, stats = advanced_hmc_sampler(prob, MF.HMC(20, 0.01), nothing, 5000, 10000, 0.8)

# log_density_flow(F, prob, K, mixer, sample...)
# MF._elbo_single(F, prob, K, mixer, sample...)
T_max = 1000
mixer = RandomShift(2, T_max)
mix_deter = ErgodicShift(2, T_max)

###############
# generating trajectories
###############
kernel = MF.HMC(50, 0.02) 
x0, v0, uv0, ua0 = MF._rand_joint_reference(prob, kernel)
x_traj_fwd = MF.forward_trajectory_x(prob, kernel, mixer, x0, v0, uv0, ua0, T_max)
x_traj_fwd_homo = MF.forward_trajectory_x(prob, kernel, mix_deter, x0, v0, uv0, ua0, T_max) 
x_traj_inv = MF.inverse_trajectory_x(prob, kernel, mixer, x0, v0, uv0, ua0, T_max)
x_traj_bwd = MF.backward_process_trajectory_x(prob, kernel, mixer, x0, v0, uv0, ua0, T_max)

JLD2.save(
    "result/banana_hmc_trajectory.jld2",
    "fwd", x_traj_fwd, "fwd_homo", x_traj_fwd_homo, "bwd", x_traj_bwd, "inv", x_traj_inv
)

p1 = plot(x_traj_fwd', label = "irf fwd")
p2 = plot(x_traj_inv', label = "irf inv")
p3 = plot(x_traj_fwd_homo', label = "fwd homogenous")
p4 = plot(x_traj_bwd', label = "irf bwd")
plot(p1, p2, p3, p4, layout = 4)
plot!(title = "HMC trace")
savefig("figure/banana_hmc_trajectory.png")

m_fwd = running_mean(x_traj_fwd) 
m_inv = running_mean(x_traj_inv)
m_fwd_homo = running_mean(x_traj_fwd_homo)
m_bwd = running_mean(x_traj_bwd)

p1 = plot(m_fwd', label = "irf fwd")
p2 = plot(m_inv', label = "irf inv")
p3 = plot(m_fwd_homo', label = "fwd homogenous")
p4 = plot(m_bwd', label = "irf bwd")
plot(p1, p2, p3, p4, layout = 4)
plot!(title = "running mean HMC")
savefig("figure/banana_hmc_mean.png")

v_fwd = running_square(x_traj_fwd)
v_inv = running_square(x_traj_inv)
v_fwd_homo = running_square(x_traj_fwd_homo)
v_bwd = running_square(x_traj_bwd)

p1 = plot(v_fwd', label = "irf fwd")
p2 = plot(v_inv', label = "irf inv")
p3 = plot(v_fwd_homo', label = "fwd homogenous")
p4 = plot(v_bwd', label = "irf bwd")
plot(p1, p2, p3, p4, layout = 4)
plot!(title = "running E(x^2) HMC")
savefig("figure/banana_hmc_var.png")

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
