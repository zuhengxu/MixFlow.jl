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

const MF = MixFlow

include("Model.jl")
include("mfvi.jl")
# plot 
include("plotting.jl")



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


name = "Cross"
target = load_model(name)

ad = AutoMooncake(; config = Mooncake.Config())
target_ad = ADgradient(ad, target)
reference, _ = mfvi(target_ad; sample_per_iter = 10, max_iters = 10000, adtype = ad)
prob = MixFlowProblem(reference, target_ad)

dims = LogDensityProblems.dimension(target_ad)

T_max = 20_000
mixer = RandomShift(2, T_max)
# mixer = ErgodicShift(2, T)


# Ts = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
Ts = vcat([10, 20, 50], 100:100:1200)

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

JLD2.save("result/stability_"*"$(name).jld2", "stats", stats)


P =  plot()
for i in 1:length(stats)
    plot!(stats[i].Ts, get_median(stats[i].errors), ribbon = get_percentiles(stats[i].errors), lw = 3, label = string(stats[i].kernel))
end
plot!(title = "$(name) inv error", xlabel = "T", ylabel = "error", yscale = :log10, legend = :bottomright)
savefig("figure/stability_" * "$(name).png")

