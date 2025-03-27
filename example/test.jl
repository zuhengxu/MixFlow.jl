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


function check_error(prob, K, mixer, T::Int)
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

    if uv0 === nothing 
        errsq = sum(abs2, x - x0) + sum(abs2, v - v0)
    else
        errsq = sum(abs2, x - x0) + sum(abs2, v - v0) + sum(abs2, uv - uv0) + sum(abs2, ua - ua0)
    end
    err = sqrt(errsq)
    return err, rejs_fwd, sort(rejs_inv)
end


function check_error(prob, K, mixer, Ts::Vector{Int})
    stats = []
    for T in Ts
        err, _, _ = check_error(prob, K, mixer, T)
        # stat = (T=T, error=err)
        push!(stats, err)
    end
    return map(identity, stats)
end


target = load_model("Banana")

target_ad = ADgradient(AutoMooncake(; config = Mooncake.Config()), target)
reference, _ = mfvi(target_ad; sample_per_iter = 10, max_iters = 10000)
prob = MixFlowProblem(reference, target_ad)

dims = LogDensityProblems.dimension(target_ad)

T_max = 20_000
mixer = RandomShift(2, T_max)
# mixer = ErgodicShift(2, T)


Ts = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
K = HMC(10, 0.02) 
# K = uncorrectHMC(10, 0.02)
# K = MALA(0.25) 
# K = RWMH(0.3*ones(dims))

stats = []
for K in [
    uncorrectHMC(10, 0.02), 
    HMC(10, 0.02), 
    MALA(0.25), 
    RWMH(0.3*ones(dims)), 
]
    Es = [] 
    @threads for id in 1:32
        Random.seed!(id)
        err = check_error(prob, K, mixer, Ts)
        push!(Es, err)
    end
    stat = (kernel = typeof(K), Ts=Ts, errors=reduce(hcat, Es))
    push!(stats, stat)
    println("$(typeof(K)) done")
end

# plot(rejs_fwd, label="rejs_fwd", lw = 2)
# plot!(rejs_inv, label="rejs_inv", lw = 2)
#
#




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
    Els_hmc = elbo_sweep(RandomBackwardMixFlow, prob, K, mixer, nsample, Ts)
    Els_hmc_deter = elbo_sweep(DeterministicMixFlow, prob, K, mix_deter, nsample, Ts)

    plot!(P, Ts, Els_hmc, label="HMC_bwd_mixflow $(ϵ)", lw=2)
    plot!(P, Ts, Els_hmc_deter, label="HMC_std_mixflow $(ϵ)", lw=2)
    plot!(P, Ts, Els_uhmc_deter, label="uncorrectHMC_std_mixflow $(ϵ)", lw=2)
end

savefig("figure/$(name)_elbo_sweep.png")
