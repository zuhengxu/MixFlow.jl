using Random, Distributions, Plots
using LinearAlgebra
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

include("Model.jl")
include("mfvi.jl")
include("utils.jl")


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


test_func = Base.Fix1(MF._log_density_ratio, prob)

x_mcmc = res["mcmc"]
x0 = res["bwd"][:, 1]
x_mcmc = hcat(x0, x_mcmc)
es = [test_func(x) for x in eachcol(x_mcmc)]
drs = expm1.(es) .+ 1





dims, N = size(x_mcmc)
nchains = 5 

samples_all = zeros(N, dims, nchains)
samples_all[:, :, 1] .= x_mcmc'
for (i, k) in enumerate(["mcmc", "fwd", "bwd", "fwd_homo", "inv"])
    s = res[k]
    samples_all[:, :, i+1] .= s'
end
chn = Chains(samples_all, [:d1, :d2])

samples = [test_func(x) for x in eachcol(x_mcmc)]

ess(chn; autocov_method = FFTAutocovMethod(), maxlag = typemax(Int))



using MCMCChains
using StatsPlots

# Define the experiment
n_iter = 100
n_name = 3
n_chain = 2

# experiment results
val = randn(n_iter, n_name, n_chain) .+ [1, 2, 3]'
val = hcat(val, rand(1:2, n_iter, 1, n_chain))
