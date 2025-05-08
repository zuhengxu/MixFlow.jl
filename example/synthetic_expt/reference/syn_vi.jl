using Random, Distributions
using LinearAlgebra
using LogExpFunctions
using LogDensityProblems, LogDensityProblemsAD
using ADTypes, Mooncake
using JLD2

using MixFlow 
const MF = MixFlow


include(joinpath(@__DIR__, "../mfvi.jl"))
include(joinpath(@__DIR__, "../Model.jl"))
include(joinpath(@__DIR__, "../utils.jl"))


function get_vi_reference(
    seed, name::String; batchsize::Int = 10, niters::Int=100_000,
)
    Random.seed!(seed)

    target = load_model(name)

    ad = AutoMooncake(; config = Mooncake.Config())
    target_ad = ADgradient(ad, target)
    reference, _ = mfvi(target_ad; sample_per_iter = batchsize, max_iters = niters, adtype = ad)
    prob = MixFlowProblem(reference, target_ad)

    res_dir = joinpath(@__DIR__, "result/")

    JLD2.save(
        joinpath(res_dir, "$(name)_mfvi.jld2"),
        "prob", prob,
        "niters", niters,
        "batchsize", batchsize,
        "seed", seed,
    )
end

# get_vi_reference(1, "Banana"; batchsize = 10, niters = 100_000)

# res = JLD2.load(
#     joinpath(@__DIR__, "result/Banana_mfvi.jld2"),
# )

