using Random, Distributions
using LinearAlgebra
using Base.Threads: @threads
using JLD2
using LogExpFunctions
using LogDensityProblems, LogDensityProblemsAD
using Bijectors
using DataFrames, CSV
using MCMCChains

using MixFlow 
using MixFlow: _rand_joint_reference, _log_density_ratio

const MF = MixFlow

include(joinpath(@__DIR__, "../julia_env/mfvi.jl"))
include(joinpath(@__DIR__, "../julia_env/Model.jl"))
include(joinpath(@__DIR__, "../julia_env/utils.jl"))
include(joinpath(@__DIR__, "../julia_env/plotting.jl"))

# return Dict with named tuple each is a kernel
function simulation_setting(name)
    if name == "Banana"
        dic = Dict(
            "HMC" => (500, 50, 0.1),
            "MALA" => (3000, 0.05, ones(2)),
            "RWMH" => (3000, 0.5, ones(2)),
        )
    elseif name == "Cross"
        dic = Dict(
            "HMC" => (500, 50, 0.05),
            "MALA" => (3000, 0.05, ones(2)),
            "RWMH" => (3000, 1.0, ones(2)),
        )
    elseif name == "Funnel"
        dic = Dict(
            "HMC" => (500, 50, 0.1),
            "MALA" => (3000, 0.1, ones(2)),
            "RWMH" => (3000, 1.0, ones(2)),
        )
    elseif name == "WarpedGaussian"
        dic = Dict(
            "HMC" => (500, 50, 0.08),
            "MALA" => (3000, 0.05, ones(2)),
            "RWMH" => (3000, 1.0, ones(2)),
        )
    else
        error("Unknown target: $name")
    end
    return dic
end

function construct_kernel(name::String, kernel_type)
    dic = simulation_setting(name)
    T_max, arg1, arg2 = dic[string(kernel_type)]
    kernel = kernel_type(arg1, arg2)
    return kernel, T_max
end
# kernel, T_max = construct_kernel("Banana", MF.RWMH)


function run_traces(seed, name::String, kernel_type, trace_type)
    Random.seed!(seed)

    vi_res = JLD2.load(
        joinpath(@__DIR__, "../syn_mfvi_fit/result/$(name)_mfvi.jld2"),
    )
    prob = vi_res["prob"]

    dims = LogDensityProblems.dimension(prob)

    kernel, T_max = construct_kernel(name, kernel_type)

    # T_max = 1000
    mixer = RandomShift(dims, T_max)
    mix_deter = ErgodicShift(dims, T_max)
    

    fn_prefix = "$(name)_$(_get_kernel_name(kernel))"
    @info "Running $(trace_type) on $(name) $(kernel) with T_max = $T_max"

    ###############
    # generating trajectories
    ###############
    x0, v0, uv0, ua0 = MF._rand_joint_reference(prob, kernel)

    if trace_type == "mcmc"
        x_traj = MF.mcmc_sampler(prob, kernel, x0, T_max)
    elseif trace_type == "fwd_homo"
        x_traj = MF.forward_trajectory_x(prob, kernel, mix_deter, x0, v0, uv0, ua0, T_max)
    elseif trace_type == "fwd_irf"
        x_traj = MF.forward_trajectory_x(prob, kernel, mixer, x0, v0, uv0, ua0, T_max)
    elseif trace_type == "bwd_irf"
        x_traj = MF.backward_process_trajectory_x(prob, kernel, mixer, x0, v0, uv0, ua0, T_max)
    elseif trace_type == "inv_irf"
        x_traj = MF.inverse_trajectory_x(prob, kernel, mixer, x0, v0, uv0, ua0, T_max)
    elseif trace_type == "bwd_inv_irf"
        x_traj = MF.backward_inverse_process_trajectory_x(prob, kernel, mixer, x0, v0, uv0, ua0, T_max)
    else
        error("Unknown trace type: $trace_type")
    end

    
    f_log_ratio = Base.Fix1(MF._log_density_ratio, prob)
    lrs = [f_log_ratio(x) for x in eachcol(x_traj)]
    q0_π_m1 = expm1.(lrs) 

    df = DataFrame(
        iter = [1:T_max+1 ;],
        d1 = x_traj[1, :],
        d2 = x_traj[2, :],
        ldr = lrs,
        dr = q0_π_m1,
    )
    return df
end

# dts = run_traces(2, "Cross", MF.HMC, "inv_irf")
