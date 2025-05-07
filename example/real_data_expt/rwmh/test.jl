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

include(joinpath(@__DIR__, "../../mfvi.jl"))
include(joinpath(@__DIR__, "../../Model.jl"))
include(joinpath(@__DIR__, "../../plotting.jl"))

function load_prob(name)
    ref = JLD2.load(joinpath(@__DIR__, "../reference/result/$(name)_mfvi.jld2"))
    prob = ref["prob"]
end

function run_elbo(
    seed, name::String, flowtype, kernel_type, step_size, T::Int, nchains; 
    nsample = 1024, leapfrog_steps=50,
    )

    flow = (flowtype <: MF.EnsembleIRFFlow) ? flowtype(T, nchains) : flowtype(T)

    Random.seed!(seed)
    prob = load_prob(name)

    dims = LogDensityProblems.dimension(prob)

    if flow isa MF.EnsembleIRFFlow
        mixer = EnsembleRandomShift(dims, T, nchains) 
    elseif flow isa MF.DeterministicMixFlow
        mixer = ErgodicShift(dims, T)
        nchains = 1
    else
        mixer = RandomShift(dims, T)
        nchains = 1
    end

    if kernel_type == MF.HMC
        kernel = MF.HMC(leapfrog_steps, step_size)
    else 
        kernel = kernel_type(step_size, ones(dims))
    end

    output = MF.mixflow(flow, prob, kernel, mixer, nsample)
    
    df = DataFrame(
        nensembles = nchains, 
        logZ = output.logZ,
        elbo = output.elbo,
        ess = output.ess/nsample,  # ess per sample
        nparticles = nsample,
    )
    return df
end

function check_error(prob, K, T::Int)
    mixer = RandomShift(LogDensityProblems.dimension(prob), T)
    
    x0, v0, uv0, ua0 = MF._rand_joint_reference(prob, K)
    x, v, uv, ua = x0, v0, uv0, ua0

    rejs_fwd = []
    for t in 1:T
        x, v, uv, ua, acc = MF.forward(prob, K, mixer, x, v, uv, ua, t)
        if !acc 
            push!(rejs_fwd, t)
        end
    end

    rejs_inv = []
    for t in T:-1:1
        x, v, uv, ua, acc = MF.inverse(prob, K, mixer, x, v, uv, ua, t)
        if !acc
            push!(rejs_inv, t)
        end
    end

    errsq = sum(abs2, x - x0) + sum(abs2, v - v0) + sum(abs2, uv - uv0) + sum(abs2, ua - ua0)
    err = sqrt(errsq)
    return err, rejs_fwd, sort(rejs_inv)
end

name = "LGCP"
prob = load_prob(name)
dims = LogDensityProblems.dimension(prob)
ϵ = 0.005
T = 5000

err, rejsfwd, rejsinv = check_error(prob, MF.RWMH(ϵ, ones(dims)), T)
plot(rejsfwd, label = "forward rejection")
plot!(rejsinv, label = "inverse rejection")


kerneltype = MF.RWMH

# flowtype = MF.BackwardIRFMixFlow
# flowtype = MF.IRFMixFlow
flowtype = MF.EnsembleIRFFlow

T = 1000
nchains = 20

df = run_elbo(1, name, flowtype, kerneltype,  ϵ, T, nchains; nsample = 64)
