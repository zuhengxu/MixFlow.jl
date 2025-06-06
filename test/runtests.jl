using Random, Distributions
using LinearAlgebra
using LogExpFunctions
using LogDensityProblems, LogDensityProblemsAD
using ADTypes, Mooncake
using NormalizingFlows
using Bijectors
using MixFlow

const MF = MixFlow

include("Model.jl")
include("mfvi.jl")
using Test


@testset "check invertibility" begin
    @testset "$(name)" for name in [
        "Banana",
        "Cross",
        "Funnel",
        "WarpedGaussian",
    ]
        target = load_model(name)

        target_ad = ADgradient(AutoMooncake(; config=Mooncake.Config()), target)
        reference, _ = mfvi(target_ad; sample_per_iter=10, max_iters=10000)
        prob = MixFlowProblem(reference, target_ad)

        dim = LogDensityProblems.dimension(target_ad)

        T_max = 200
        mixer = RandomShift(dim, T_max)
        @testset "$K" for K in [
            uncorrectHMC(10, 0.02),
            HMC(10, 0.02),
            MALA(0.25, ones(dim)),
            RWMH(0.3, ones(dim)),
        ]

            T = 10

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
            rejs_inv = sort(rejs_inv)

            # ensure that the forward and inverse steps are consistent
            # @test rejs_fwd .- rejs_inv ≈ zeros(length(rejs_fwd))

            @test norm(x .- x0) ≤ 1e-3
            @test norm(v .- v0) ≤ 1e-3
        end
    end
end
