struct ErgodicShift1D{F, T} <: AbstractUnifMixer 
    ξs_uv::F
    ξs_ua::T 
end
ntransitions(S::ErgodicShift1D) = size(S.ξs_ua, 1) 
ErgodicShift1D(nsteps::Int) = ErgodicShift1D(π/4 .* ones(nsteps), π/4 .* ones(nsteps))
RandomShift1D(rng, nsteps::Int) = ErgodicShift1D(rand(rng, nsteps), rand(rng, nsteps))
RandomShift1D(nsteps::Int) = RandomShift1D(Random.default_rng(), nsteps)

_ergodic_shift(u::T, ξ::T) where T<:Real = (u + ξ) % 1
_inv_ergodic_shift(u::T, ξ::T) where T<:Real = (u + 1 - ξ) % 1

function update_uniform(S::ErgodicShift1D, uv::T, ua::T, t::Int) where {T<:Real}
    uvn = _ergodic_shift(uv, S.ξs_uv[t])
    uan = _ergodic_shift(ua, S.ξs_ua[t])
    return uvn, uan
end

function inv_update_uniform(S::ErgodicShift1D, uv::T, ua::T, t::Int) where {T<:Real}
    uan = _inv_ergodic_shift(ua, S.ξs_ua[t])
    uvn = _inv_ergodic_shift(uv, S.ξs_uv[t])
    return uvn, uan
end

struct ErgodicShift{F, T} <: AbstractUnifMixer 
    ξs_uv::F
    ξs_ua::T
end
ntransitions(S::ErgodicShift) = size(S.ξs_ua, 1)

ErgodicShift(D::Int, nsteps::Int) = ErgodicShift(π/4 .* ones(D, nsteps), π/4 .* ones(nsteps))
RandomShift(rng, D::Int, nsteps::Int) = ErgodicShift(rand(rng, D, nsteps), rand(rng, nsteps))
RandomShift(D::Int, nsteps::Int) = RandomShift(Random.default_rng(), D, nsteps)

function update_uniform(S::ErgodicShift, uv::AbstractVector{T}, ua::T, t::Int) where {T<:Real}
    uvn = _ergodic_shift.(uv, S.ξs_uv[:, t])
    uan = _ergodic_shift(ua, S.ξs_ua[t])
    return uvn, uan
end

function inv_update_uniform(S::ErgodicShift, uv::AbstractVector{T}, ua::T, t::Int) where {T<:Real}
    # println("inv_update_uniform")
    uvn = _inv_ergodic_shift.(uv, S.ξs_uv[:, t])
    uan = _inv_ergodic_shift(ua, S.ξs_ua[t])
    return uvn, uan
end

# set up mixers for the ensemble flow
function EnsembleRandomShift(D::I, T::I, Nensemble::I) where {I<:Int}
    uvs = [rand(D, T) for _ in 1:Nensemble]
    uas = [rand(T) for _ in 1:Nensemble]
    return StructArrays{ErgodicShift}(ξs_uv=uvs, ξs_ua=uas)
end

function EnsembleErgodicShift(D::I, T::I, Nensemble::I) where {I<:Int}
    uvs = [π/4 .* ones(D, T) for _ in 1:Nensemble]
    uas = [π/4 .* ones(T) for _ in 1:Nensemble]
    return StructArrays{ErgodicShift}(ξs_uv=uvs, ξs_ua=uas)
end
