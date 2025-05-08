using Distributions, Random, LinearAlgebra
using LogDensityProblems, LogDensityProblemsAD
using DifferentiationInterface
using IrrationalConstants
using LogExpFunctions
using StatsBase, Statistics, StatsFuns
using DataFrames, CSV, DelimitedFiles
using ADTypes
using Zygote, Mooncake
using Tullio
using FillArrays


using MixFlow 
# using Plots

# 1d targets
Mixture1D() = MixtureModel(Normal, [(0.0, 0.8), (-3.0, 1.5), (3.0, 0.5)], [0.3, 0.5, 0.2])

# Synthetic targets
include("targets/banana.jl")
include("targets/cross.jl")
include("targets/neal_funnel.jl")
include("targets/warped_gaussian.jl")

# real-data targets
include("targets/brownian.jl")
include("targets/sparse_reg.jl")
include("targets/logreg_sonar.jl")
include("targets/t_reg.jl")
include("targets/lgcp.jl")


# wrapper function to load models
function load_model(name::String)
    if name == "Mixture1D"
        return Mixture1D()
    elseif name == "Cauchy1D"
        return Cauchy()
    elseif name == "Banana"
        return Banana(2, 1.0, 10.0)
    elseif name == "Cross"
        return Cross()
    elseif name == "Funnel"
        return Funnel(2)
    elseif name == "WarpedGaussian"
        return WarpedGauss()
    elseif name == "Brownian"
        return _load_brownian()
    elseif name == "Sonar"
        return _load_sonar()
    elseif name == "SparseRegression"
        return _load_sparse_reg()
    elseif name == "TReg"
        return _load_t_reg()
    elseif name == "LGCP"
        return _load_lgcp()
    else
        error("Model not defined")
    end
end

synthetic_list = Set(["Mixture1D", "Cauchy1D", "Banana", "Cross", "Funnel", "WarpedGaussian"])
real_data_list = Set(["Brownian", "Sonar", "SparseRegression", "TReg", "LGCP"])


# target, dims, ad = load_model("Sonar")
# x = randn(dims)
# l, g = LogDensityProblems.logdensity_and_gradient(target, x)

# LogDensityProblems.capabilities(::ContinuousDistribution) =  LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(dist::ContinuousDistribution) = length(dist)
LogDensityProblems.logdensity(dist::ContinuousDistribution, x) = logpdf(dist, x)


MixFlow.iid_sample(dist::ContinuousDistribution, n::Int) = rand(dist, n)
MixFlow.iid_sample(dist::ContinuousDistribution) = rand(dist)



# tracking logdensity prob
mutable struct TrackedLogDensityProblem{Prob}
    n_density_evals  :: Int
    n_gradient_evals :: Int
    prob             :: Prob
end

function TrackedLogDensityProblem(prob)
    TrackedLogDensityProblem{typeof(prob)}(0, 0, prob)
end
is_tracked(prob) = prob isa TrackedLogDensityProblem

function LogDensityProblems.capabilities(::Type{TrackedLogDensityProblem{Prob}}) where {Prob}
    return LogDensityProblems.capabilities(Prob)
end

LogDensityProblems.dimension(prob::TrackedLogDensityProblem) = LogDensityProblems.dimension(prob.prob)

function LogDensityProblems.logdensity(prob::TrackedLogDensityProblem, x)
    prob.n_density_evals += 1
    return LogDensityProblems.logdensity(prob.prob, x)
end

function LogDensityProblems.logdensity_and_gradient(prob::TrackedLogDensityProblem, x)
    prob.n_gradient_evals += 1
    return LogDensityProblems.logdensity_and_gradient(prob.prob, x)
end

compute_cost(prob::TrackedLogDensityProblem) = prob.n_density_evals + prob.n_gradient_evals * LogDensityProblems.dimension(prob)

