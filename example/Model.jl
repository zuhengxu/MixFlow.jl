using Distributions, Random, LinearAlgebra
using IrrationalConstants
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
        return Banana(2, 1.0, 100)
    elseif name == "Cross"
        return Cross()
    elseif name == "Funnel"
        return Funnel(2)
    elseif name == "warped_gaussian"
        return WarpedGauss()
    else
        error("Model not defined")
    end
end
