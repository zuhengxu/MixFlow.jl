using Random, Distributions
using LinearAlgebra
using LogExpFunctions
using LogDensityProblems, LogDensityProblemsAD
using MixFlow
using ADTypes, Mooncake
using NormalizingFlows
using Bijectors

include("Model.jl")
include("mfvi.jl")
using Test

target = load_model("Banana")

target_ad = ADgradient(AutoMooncake(; config = Mooncake.Config()), target)
reference, _ = mfvi(target_ad; sample_per_iter = 10, max_iters = 10000)
prob = MixFlowProblem(reference, target_ad)

dim = LogDensityProblems.dimension(target_ad)

T_max = 200
mixer = RandomShift(dim, T_max)

@testset "check invertibility" begin
    @testset "$K" for K in [
        K = uncorrectHMC(10, 0.02)
        K = HMC(10, 0.02)
        K = MALA(0.25)
        K = RWMH(0.3 * ones(dim))
    ]

    T = 20
    
    x0, v0, uv0, ua0 = MixFlow._rand_joint_reference(prob, K)
    x, v, uv, ua = x0, v0, uv0, ua0

    rejs_fwd = []
    for t in 1:T
        x, v, uv, ua, acc = forward(prob, K, mixer, x, v, uv, ua, t)
        if !acc 
            push!(rejs_fwd, t)
        end
    end

    rejs_inv = []
    for t in T:-1:1
        x, v, uv, ua, acc = inverse(prob, K, mixer, x, v, uv, ua, t)
        if !acc
            push!(rejs_inv, t)
        end
    end
    rejs_inv = sort(rejs_inv)

    # ensure that the forward and inverse steps are consistent
    # @test rejs_fwd .- rejs_inv ≈ zeros(length(rejs_fwd))

    @test x .- x0 ≈ zeros(dim)
    @test v .- v0 ≈ zeros(dim)
    end
end
