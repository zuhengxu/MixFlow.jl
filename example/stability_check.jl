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

include("Model.jl")


function check_error_single_run(id, name, )
    Random.seed!(id)
    

end



# target = MvNormal(zeros(2), ones(2))
target = load_model("Banana")

target_ad = ADgradient(AutoMooncake(; config = Mooncake.Config()), target)
reference, _ = mfvi(target_ad; sample_per_iter = 10, max_iters = 10000)
prob = MixFlowProblem(reference, target_ad)

dim = LogDensityProblems.dimension(target_ad)

T_max = 20_000
mixer = RandomShift(2, T_max)
# mixer = ErgodicShift(2, T)

# K = uncorrectHMC(10, 0.02)
K = HMC(10, 0.02)
# K = MALA(0.25)
# K = RWMH(0.3*ones(dim))
x0, v0, uv0, ua0 = MixFlow._rand_joint_reference(prob, K)
x, v, uv, ua = x0, v0, uv0, ua0


Es = []
Ts = [10, 100, 200, 500, 1000, 2000, 5000, 10000]
for T in Ts
    for t in 1:T
        x, v, uv, ua, _ = forward(prob, K, mixer, x, v, uv, ua, t)
    end

    for t in T:-1:1
        x, v, uv, ua, _ = inverse(prob, K, mixer, x, v, uv, ua, t)
    end

    errsq = sum(abs2, x - x0) + sum(abs2, v - v0)
    err = sqrt(errsq)
    push!(Es, err)
end
