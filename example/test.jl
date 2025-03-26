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


# target = MvNormal(zeros(2), ones(2))
target = load_model("Banana")

target_ad = ADgradient(AutoMooncake(; config = Mooncake.Config()), target)
reference, _ = mfvi(target_ad; sample_per_iter = 10, max_iters = 10000)
prob = MixFlowProblem(reference, target_ad)

dim = LogDensityProblems.dimension(target_ad)

T_max = 20_000
mixer = RandomShift(2, T_max)
# mixer = ErgodicShift(2, T)


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
    return stats
end


# K = uncorrectHMC(10, 0.02)
# K = HMC(10, 0.02)
# K = MALA(0.25)
# K = RWMH(0.3*ones(dim))
T = 100
err, _, _ = check_error(prob, K, mixer, T)



for K in [
    uncorrectHMC(10, 0.02), 
    HMC(10, 0.02), 
    MALA(0.25), 
    RWMH(0.3*ones(dim)), 
]
     
    @threads for _ in 1:32
        err, _, _ = check_error(prob, K, mixer, T)
        println("$(K): $(err)")
    end
end


# plot(rejs_fwd, label="rejs_fwd", lw = 2)
# plot!(rejs_inv, label="rejs_inv", lw = 2)


# function backward_process(prob, K, mixer, x0, v0, uv0, ua0, T)
#     x, v, uv, ua = x0, v0, uv0, ua0
#     for t in T:-1:1
#         x, v, uv, ua = forward(prob, K, mixer, x, v, uv, ua, t)
#     end
#     return x, v, uv, ua
# end

# x0 = randn()
# v0 = x0 + randn()
# uv0 = rand()
# ua0 = rand()

# xs = []
# vs = []
# uvs = []
# uas = []
# for i in 1:T
#     x, v, uv, ua = backward_process(prob, K, mixer, x0, v0, uv0, ua0, i)
#     push!(xs, x)
#     push!(vs, v)
#     push!(uvs, uv)
#     push!(uas, ua)
# end


# # diaconis example
# T = 20
# us = rand(T)
# as = rand(Bernoulli(0.5), T)

# ϕ(x, a, u) = a == 0 ? u*(1-x) : u*x

# function ϕ_bwd(x0, as, us, T)
#     xs = []
#     for t in 1:T
#         x = x0
#         for i in t:-1:1
#             x = ϕ(x, as[i], us[i])
#         end
#         push!(xs, x)
#     end
#     return xs
# end

# xs1 = ϕ_bwd(rand(), as, us, T)
# xs2 = ϕ_bwd(rand(), as, us, T)
# xs3 = ϕ_bwd(rand(), as, us, T)

# plot(xs1, label="1")
# plot!(xs2, label="2")
# plot!(xs3, label="3")
