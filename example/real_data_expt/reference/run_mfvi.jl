using Random, Distributions
using LinearAlgebra
using LogDensityProblems, LogDensityProblemsAD
using JLD2

using MixFlow 
const MF = MixFlow


include(joinpath(@__DIR__, "../../mfvi.jl"))
include(joinpath(@__DIR__, "../../Model.jl"))


function get_vi_reference(
    seed, name::String; batchsize::Int = 10, niters::Int=50_000
)
    Random.seed!(seed)

    @info "load model $(name)"
    target, dims, ad = load_model(name)

    @info "learning mfvi for $(name), dims = $(dims)"
    reference, _ = mfvi(target; sample_per_iter = batchsize, max_iters = niters, adtype = ad)
    prob = MixFlowProblem(reference, target)

    res_dir = joinpath(@__DIR__, "result/")

    @info "save learned reference to jld"
    JLD2.save(
        joinpath(res_dir, "$(name)_mfvi.jld2"),
        "reference", reference, 
        "prob", prob,
        "niters", niters,
        "batchsize", batchsize,
        "seed", seed,
    )
    return reference
end


for name in ["TReg", "SparseRegression", "Brownian", "Sonar", "LGCP"]
    get_vi_reference(1, name; batchsize = 10, niters = 100_000)
end

# res = JLD2.load(joinpath(@__DIR__, "result/Brownian_mfvi.jld2"))
# pr = res["prob"]


