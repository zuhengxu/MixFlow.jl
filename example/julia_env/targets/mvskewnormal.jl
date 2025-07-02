# using Distributions, Random, LinearAlgebra
# using LogDensityProblems, LogDensityProblemsAD

# dist = SkewNormal(2, 0.5, 10.0)

# logpdf(dist, 0.5)

# rand(dist, 10)

# function MvSkewNormal(ξs::AbstractVector, ws::AbstractVector, αs::AbstractVector)
#     StructArray{SkewNormal}((ξs, ws, αs)) 
# end

# dims = 10
# ξs = randn(dims)*10
# ws = rand(dims)*10 .+ 1
# αs = randn(dims) * 2
# dv = StructArray{SkewNormal}((ξs, ws, αs))

# xx = randn()
# ℓs = map(p -> logpdf(p, xx), dv)
# rs = map(p -> rand(p, 2), dv)


struct MvSkewNormal{T<:AbstractVector}
    dim::Int       #dimension
    ξs::T           # location parameters
    ws::T           # scale parameters
    αs::T           # skewness parameters
end
# function MvSkewNormal(ξs::AbstractVector, ws::AbstractVector, αs::AbstractVector)
#     dim = length(ξs)
#     @assert length(ws) == dim && length(αs) == dim "All parameters must have the same dimension"
#     return SkewNormal.(ξs, ws, αs)
# end

function MvSkewNormal()
    dims = 100
    ξs = randn(Xoshiro(123), dims)*10
    ws = rand(Xoshiro(123), dims)*10 .+ 1
    αs = 10randn(Xoshiro(456), dims) 
    return MvSkewNormal(dims, ξs, ws, αs)
end
    
LogDensityProblems.dimension(prob::MvSkewNormal) = prob.dim

function LogDensityProblems.logdensity(prob::MvSkewNormal, z::AbstractVector)
    @assert prob.dim == length(z) 
    (; dim, ξs, ws, αs) = prob
    ℓs = logpdf.(SkewNormal.(ξs, ws, αs), z)
    return sum(ℓs)
end

function target_sample(dist::MvSkewNormal, n::Int)
    (; dim, ξs, ws, αs) = dist
    dists = SkewNormal.(ξs, ws, αs)
    xs = zeros(dim, n)
    for y in eachcol(xs)
        y .= rand.(dists)
    end
    return xs
end

function _load_skewnormal()
    target = MvSkewNormal()
    ad = AutoMooncake(; config = Mooncake.Config())
    dims = LogDensityProblems.dimension(target)
    # target_ad = ADgradient(ad, target; x = randn(dims))
    return target, dims, ad 
end

