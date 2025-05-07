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

ErgodicShift(D::Int, nsteps::Int) = ErgodicShift(π/8 .* ones(D, nsteps), π/7 .* ones(nsteps))
RandomShift(rng, D::Int, nsteps::Int) = ErgodicShift(rand(rng, D, nsteps), rand(rng, nsteps))
RandomShift(D::Int, nsteps::Int) = RandomShift(Random.default_rng(), D, nsteps)

function update_uniform(S::ErgodicShift, uv::AbstractVector{T}, ua::T, t::Int) where {T<:Real}
    uvn = _ergodic_shift.(uv, @views(S.ξs_uv[:, t]))
    uan = _ergodic_shift(ua, S.ξs_ua[t])
    # uvn, uan = _arnold_cat_map(uvn, uan)
    return uvn, uan
end

function inv_update_uniform(S::ErgodicShift, uv::AbstractVector{T}, ua::T, t::Int) where {T<:Real}
    # uv, ua = _inv_arnold_cat_map(uv, ua)
    uvn = _inv_ergodic_shift.(uv, @views(S.ξs_uv[:, t]))
    uan = _inv_ergodic_shift(ua, S.ξs_ua[t])
    return uvn, uan
end

# arnold cat map and its inverse
function _arnold_cat_map(x::T, y::T) where T<:Real
    x_ = (x + y) % 1
    y_ = (x + 2y) % 1
    return x_, y_
end

function _inv_arnold_cat_map(x::T, y::T) where T<:Real
    y_ = (y + 1 - x) % 1
    x_ = (x + 1 - y_) % 1
    return x_, y_
end

function _arnold_cat_map(x::AbstractVector{T}, y::T) where T<:Real
    dims = size(x, 2)
    for d in 1:dims
        x[d], y = _arnold_cat_map(x[d], y)
    end
    return x, y
end

function _inv_arnold_cat_map(x::AbstractVector{T}, y::T) where T<:Real
    dims = size(x, 2)
    for d in dims:-1:1
        x[d], y = _inv_arnold_cat_map(x[d], y)
    end
    return x, y
end

# set up mixers for the ensemble flow
function EnsembleRandomShift(D::I, T::I, Nensemble::I) where {I<:Int}
    uvs = [rand(D, T) for _ in 1:Nensemble]
    uas = [rand(T) for _ in 1:Nensemble]
    return StructArray{ErgodicShift}(ξs_uv=uvs, ξs_ua=uas)
end

function EnsembleErgodicShift(D::I, T::I, Nensemble::I) where {I<:Int}
    uvs = [π/8 .* ones(D, T) for _ in 1:Nensemble]
    uas = [π/7 .* ones(T) for _ in 1:Nensemble]
    return StructArray{ErgodicShift}(ξs_uv=uvs, ξs_ua=uas)
end


# ua0, uv0 = rand(2), rand()
# ua, uv = copy(ua0), copy(uv0)

# T = 100
# for i in 1:T
#     ua, uv = MF._arnold_cat_map(ua, uv)
# end

# for i in 1:T
#     ua, uv = MF._inv_arnold_cat_map(ua, uv)
# end
# [ua, uv] .- [ua0, uv0]

