
abstract type AbstractUnifMixer end

struct ErgodicShift{F} <: AbstractUnifMixer 
    ξs::F
end
ntransitions(S::ErgodicShift{Vector}) = size(S.ξs, 1) 
ntransitions(S::ErgodicShift{Matrix}) = size(S.ξs, 2)

ErgodicShift(D::Int, nsteps::Int) = ErgodicShift(π/4 .* ones(D, nsteps))
RandomShift(rng, D::Int, nsteps::Int) = ErgodicShift(rand(rng, D, nsteps))

forward(S::ErgodicShift{Vector{T}}, u::T, t) where T<:Real = (u + S.ξs[:, t]) % 1
inverse(S::ErgodicShift{Vector{T}}, u::T, t) where T<:Real = (u + 1 - S.ξs[:, t]) % 1

forward(S::ErgodicShift{Matrix{T}}, u::AbstractVector{T}, t) where T<:Real = (u .+ @view(S.ξs[:, t])) .% 1
inverse(S::ErgodicShift{Matrix{T}}, u::AbstractVector{T}, t) where T<:Real = (u .+ 1 .- @view(S.ξs[:, t])) .% 1

