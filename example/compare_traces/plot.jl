using MCMCDiagnosticTools

fig_dir = joinpath("figure/")
res_dir = joinpath("result/")

name = "Banana"

vi_res = JLD2.load(
        joinpath(@__DIR__, "result/$(name)_mfvi.jld2"),
    )
prob = vi_res["prob"]

kernel = HMC(50, 0.02)
fn_prefix = "$(name)_$(_get_kernel_name(kernel))"
res = JLD2.load( joinpath(res_dir, "$(fn_prefix)_trajectory.jld2"))


f_log_ratio = Base.Fix1(MF._log_density_ratio, prob)


x_mcmc = res["mcmc"]


x0 = res["bwd"][:, 1]
x_mcmc = hcat(x0, x_mcmc)
es = [test_func(x) for x in eachcol(x_mcmc)]
drs = expm1.(es) .+ 1



using MCMCChains

dims, N = size(x_mcmc)
nchains = 4 

samples_all = zeros(N, dims, nchains)
samples_all[:, :, 1] .= x_mcmc'
for (i, k) in enumerate(["fwd", "bwd", "fwd_homo", "inv"])
    s = res[k]
    samples_all[:, :, i+1] .= s'
end
chn = Chains(samples_all, [:d1, :d2])

samples = [test_func(x) for x in eachcol(x_mcmc)]

ess_chn = ess(chn[:, :, 1]; autocov_method = FFTAutocovMethod(), maxlag = typemax(Int))



using StatsPlots

# Define the experiment
n_iter = 100
n_name = 3
n_chain = 2

# experiment results
val = randn(n_iter, n_name, n_chain) .+ [1, 2, 3]'
val = hcat(val, rand(1:2, n_iter, 1, n_chain))




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

include(joinpath(@__DIR__, "../mfvi.jl"))
include(joinpath(@__DIR__, "../Model.jl"))
include(joinpath(@__DIR__, "../utils.jl"))
include(joinpath(@__DIR__, "../plotting.jl"))

# return Dict with named tuple each is a kernel
function simulation_setting(name)
    if name == "Banana"
        dic = Dict(
            "HMC" => (500, 50, 0.2),
            "MALA" => (3000, 0.05, ones(2)),
            "RWMH" => (3000, 0.5, ones(2)),
        )
    elseif name == "Cross"
        dic = Dict(
            "HMC" => (500, 50, 0.1),
            "MALA" => (3000, 0.05, ones(2)),
            "RWMH" => (3000, 1.0, ones(2)),
        )
    elseif name == "Funnel"
        dic = Dict(
            "HMC" => (500, 50, 0.2),
            "MALA" => (3000, 0.1, ones(2)),
            "RWMH" => (3000, 1.0, ones(2)),
        )
    elseif name == "WarpedGaussian"
        dic = Dict(
            "HMC" => (500, 50, 0.2),
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
    @info "Running $(trace_type) on $(fn_prefix) with T_max = $T_max"

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

dts = run_traces(1, "Banana", MF.RWMH, "mcmc")

function chain_from_combine_csvs( 
    combined_csvs_folder::String,
    target,
    kernel_str, 
    trace_type, 
)
    df = CSV.read(joinpath(combined_csvs_folder, "summary.csv"), DataFrame)

    selector = Dict(
        :target => target, 
        :kernel => kernel_str,
        :trace_type => trace_type,
    )
    # groupby then iter over groupby and put in 3way array
    ds = _subset_expt(df, selector)
    dgs = @pipe ds |> 
            select(_, [:iter, :d1, :d2, :dr, :seed]) |>
            groupby(_, [:seed])

    Cs = zeros(size(dds[1], 1), 3, length(dgs))
    for (i, d) in enumerate(dds)
        Cs[:, :, i] .= Array(d[:, 2:end-1])
    end
    # xs = [1:size(dds, 1) ;]
    return Cs
end
    
df = CSV.read(
    joinpath(
        "/home/zuhdav/Research/MixFlow.jl/example/compare_traces/deliverables/scriptName=traces.nf___dryRun=false___nrunThreads=5/output",
        "summary.csv"
    ), DataFrame)

target = "Banana"
kernel_str = "MF.RWMH"
trace_type = "mcmc"

selector = Dict(
    :target => target, 
    :kernel => kernel_str,
    :tracetype => trace_type,
)
ds = _subset_expt(df, selector)
dds = @pipe ds |> 
        select(_, [:iter, :d1, :d2, :dr, :seed]) |>
        groupby(_, [:seed])

size(dds[1])
Cs = zeros(size(dds[1], 1), size(dds[1], 2)-2, length(dds))
for (i, d) in enumerate(dds)
    Cs[:, :, i] .= Array(d[:, 2:end-1])
end

chn = Chains(Cs, [:d1, :d2, :dr])


Es = [ess(chn[:, :, i]; autocov_method = FFTAutocovMethod(), maxlag = typemax(Int)) for i in 1:size(Cs, 3)]
E = ess(chn[:, :, 1]; autocov_method = FFTAutocovMethod(), maxlag = typemax(Int))
# groupby then iter over groupby and put in 3way array
using Pipe

sort!(ds)
