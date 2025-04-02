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



# log_density_flow(F, prob, K, mixer, sample...)
# MF._elbo_single(F, prob, K, mixer, sample...)
T_max = 1_000
mixer = RandomShift(2, T_max)
mix_deter = ErgodicShift(2, T_max)

###############
# generating trajectories
###############
kernel = HMC(100, 0.01) 

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

function running_mean(xs::Matrix{T}) where T
    cumsum(xs; dims = 2) ./ [1:size(xs, 2) ;]'
end

function running_square(xs::Matrix{T}) where T
    cumsum(xs.^2, dims = 2) ./ [1: size(xs, 2) ;]'
end

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

