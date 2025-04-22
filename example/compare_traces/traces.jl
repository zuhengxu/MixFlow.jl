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
using MixFlow: _rand_joint_reference, _log_density_ratio

const MF = MixFlow

include(joinpath(@__DIR__, "../mfvi.jl"))
include(joinpath(@__DIR__, "../Model.jl"))
include(joinpath(@__DIR__, "../plotting.jl"))

function run_traces(name::String, kernel::MultivariateInvolutiveKernel, T_max::Int)
    Random.seed!(1)
    target = load_model(name)

    ad = AutoMooncake(; config = Mooncake.Config())
    target_ad = ADgradient(ad, target)
    reference, _ = mfvi(target_ad; sample_per_iter = 10, max_iters = 10000, adtype = ad)
    prob = MixFlowProblem(reference, target_ad)

    dims = LogDensityProblems.dimension(target_ad)
    # T_max = 1000
    mixer = RandomShift(dims, T_max)
    mix_deter = ErgodicShift(dims, T_max)

    fn_prefix = "$(name)_$(_get_kernel_name(kernel))"
    @info "Running $(fn_prefix) with T_max = $T_max"

    fig_dir = joinpath(@__DIR__, "figure/")
    res_dir = joinpath(@__DIR__, "result/")

    ###############
    # generating trajectories
    ###############
    x0, v0, uv0, ua0 = MF._rand_joint_reference(prob, kernel)
    x_traj_mcmc = MF.mcmc_sampler(prob, kernel, x0, T_max)
    x_traj_fwd = MF.forward_trajectory_x(prob, kernel, mixer, x0, v0, uv0, ua0, T_max)
    x_traj_fwd_homo = MF.forward_trajectory_x(prob, kernel, mix_deter, x0, v0, uv0, ua0, T_max) 
    x_traj_inv = MF.inverse_trajectory_x(prob, kernel, mixer, x0, v0, uv0, ua0, T_max)
    x_traj_bwd = MF.backward_process_trajectory_x(prob, kernel, mixer, x0, v0, uv0, ua0, T_max)


    JLD2.save(
        joinpath(res_dir, "$(fn_prefix)_trajectory.jld2"),
        "mcmc", x_traj_mcmc,
        "fwd", x_traj_fwd, 
        "fwd_homo", x_traj_fwd_homo, 
        "bwd", x_traj_bwd, 
        "inv", x_traj_inv
    )

    p1 = plot(x_traj_fwd', label = "irf fwd")
    p2 = plot(x_traj_inv', label = "irf inv")
    p3 = plot(x_traj_fwd_homo', label = "fwd homogenous")
    p4 = plot(x_traj_bwd', label = "irf bwd")
    p5 = plot(x_traj_mcmc', label = "mcmc")
    plot(p1, p2, p3, p4, p5, layout = 5)
    plot!(title = "$(fn_prefix) trace")
    savefig(joinpath(fig_dir, "$(fn_prefix)_trajectory.png"))

    m_fwd = running_mean(x_traj_fwd) 
    m_inv = running_mean(x_traj_inv)
    m_fwd_homo = running_mean(x_traj_fwd_homo)
    m_bwd = running_mean(x_traj_bwd)
    m_mcmc = running_mean(x_traj_mcmc)

    p1 = plot(m_fwd', label = "irf fwd")
    p2 = plot(m_inv', label = "irf inv")
    p3 = plot(m_fwd_homo', label = "fwd homogenous")
    p4 = plot(m_bwd', label = "irf bwd")
    p5 = plot(m_mcmc', label = "mcmc")
    plot(p1, p2, p3, p4, p5, layout = 5)
    plot!(title = "$(fn_prefix) running mean")
    savefig(joinpath(fig_dir, "$(fn_prefix)_mean.png"))

    v_fwd = running_square(x_traj_fwd)
    v_inv = running_square(x_traj_inv)
    v_fwd_homo = running_square(x_traj_fwd_homo)
    v_bwd = running_square(x_traj_bwd)
    v_mcmc = running_square(x_traj_mcmc)

    p1 = plot(v_fwd', label = "irf fwd")
    p2 = plot(v_inv', label = "irf inv")
    p3 = plot(v_fwd_homo', label = "fwd homogenous")
    p4 = plot(v_bwd', label = "irf bwd")
    p5 = plot(v_mcmc', label = "mcmc")
    plot(p1, p2, p3, p4, p5, layout = 5)
    plot!(title = "$(fn_prefix) running E(x^2)")
    savefig(joinpath(fig_dir, "$(fn_prefix)_var.png"))
end
