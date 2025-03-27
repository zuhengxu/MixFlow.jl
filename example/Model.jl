using Distributions, Random, LinearAlgebra
using LogDensityProblems
using IrrationalConstants
using MixFlow 
using Plots

# 1d targets
Mixture1D() = MixtureModel(Normal, [(0.0, 0.8), (-3.0, 1.5), (3.0, 0.5)], [0.3, 0.5, 0.2])

# other synthetics
include("targets/banana.jl")
include("targets/cross.jl")
include("targets/neal_funnel.jl")
include("targets/warped_gaussian.jl")

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
    else
        error("Model not defined")
    end
end

# LogDensityProblems.capabilities(::ContinuousDistribution) =  LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(dist::ContinuousDistribution) = length(dist)
LogDensityProblems.logdensity(dist::ContinuousDistribution, x) = logpdf(dist, x)


MixFlow.iid_sample(dist::ContinuousDistribution, n::Int) = rand(dist, n)
MixFlow.iid_sample(dist::ContinuousDistribution) = rand(dist)
