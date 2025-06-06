"""
    Funnel{T<:Real}

Multidimensional Neal's Funnel distribution

# Reference
[1] Stan User’s Guide: 
https://mc-stan.org/docs/2_18/stan-users-guide/reparameterization-section.html#ref-Neal:2003
[2] Radford Neal 2003. “Slice Sampling.” Annals of Statistics 31 (3): 705–67.
"""
struct Funnel{T<:Real} <: ContinuousMultivariateDistribution
    "Dimension of the distribution, must be >= 2"
    dim::Int
    "Mean of the first dimension"
    μ::T
    "Standard deviation of the first dimension, must be > 0"
    σ::T
    function Funnel{T}(dim::Int, μ::T, σ::T) where {T<:Real}
        dim >= 2 || error("dim must be >= 2")
        σ > 0 || error("σ must be > 0")
        return new{T}(dim, μ, σ)
    end
end
Funnel(dim::Int, μ::T, σ::T) where {T<:Real} = Funnel{T}(dim, μ, σ)
Funnel(dim::Int, σ::T) where {T<:Real} = Funnel{T}(dim, zero(T), σ)
Funnel(dim::Int) = Funnel(dim, 0.0, 9.0)

Base.length(p::Funnel) = p.dim
Base.eltype(::Funnel{T}) where {T<:Real} = T

function Distributions._rand!(rng::AbstractRNG, p::Funnel{T}, x::AbstractVecOrMat{T}) where {T<:Real}
    d, μ, σ = p.dim, p.μ, p.σ
    d == size(x, 1) || error("Dimension mismatch")
    x[1, :] .= randn(rng, T, size(x, 2)) .* σ .+ μ
    x[2:end, :] .= randn(rng, T, d - 1, size(x, 2)) .* exp.(@view(x[1, :]) ./ 2)'
    return x
end

function Distributions._logpdf(p::Funnel{T}, x::AbstractVector{T}) where {T<:Real}
    d, μ, σ = p.dim, p.μ, p.σ
    x1 = x[1]
    x2 = x[2:end]
    lpdf_x1 = logpdf(Normal(μ, σ), x1)
    lpdf_x2_given_1 = logpdf(MvNormal(zeros(T, d-1), exp(x1)I), x2)
    return lpdf_x1 + lpdf_x2_given_1
end
