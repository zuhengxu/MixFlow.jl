using PDMats: PDiagMat
using StatsFuns: normcdf, norminvcdf

struct RWMH{T} <: MultivariateInvolutiveKernel 
    σs::PDiagMat{T} # diagonal stds of the proposal
end

RWMH(σs::AbstractVector{T}) where {T} = RWMH(PDiagMat{T}(σs))
RWMH(dim::Int) = RWMH(PDiagMat(ones(dim)))

involution(::RWMH{T}, x::AbstractVector{T}, v::AbstractVector{T}) where T = (v, x)
_dist_v_given_x(K::RWMH, x) = MvNormal(x, K.σs.^2)
_rand_v_given_x(K::RWMH, x) = rand(_dist_v_given_x(K, x))
_rand_v_given_x(K::RWMH, x, n::Int) = rand(_dist_v_given_x(K, x), n)

# normcdf(μ, σ, x)
# norminvcdf(μ, σ, x)
_cdf_v_given_x(K::RWMH, x::AbstractVector{T}, v::AbstractVector{T}) where T = normcdf.(x, K.σs.diag, v)
_invcdf_v_given_x(K::RWMH, x::AbstractVector{T}, uv::AbstractVector{T}) where T = norminvcdf.(x, K.σs.diag, uv)


function forward(
    prob::MixFlowProblem, K::RWMH{T}, unif_mixer::ErgodicShift,
    x::AbstractVector{T}, v::AbstractVector{T}, uv::AbstractVector{T}, ua::T,
    t::Int,
) where {T<:Real}
    # refresh uniform aux variables
    uv, ua = update_uniform(unif_mixer, uv, ua, t)

    # involutive mcmc step
    uv_ = _cdf_v_given_x(K, x, v)
    ṽ = _invcdf_v_given_x(K, x, uv)
    x_, v_ = involution(K, x, ṽ)

    logr = logpdf_aug_target(prob, K, x_, v_) - logpdf_aug_target(prob, K, x, ṽ)

    if check_acc(ua, logr)
        ua = exp(log(ua) - logr)
        return x_, v_, uv_, ua
    else
        return x, ṽ, uv_, ua
    end
end

function inverse(
    prob::MixFlowProblem, K::RWMH{T}, unif_mixer::ErgodicShift,
    x_::AbstractVector{T}, v_::AbstractVector{T}, uv::AbstractVector{T}, ua::T,
    t::Int,
) where {T<:Real}
    x, ṽ = involution(K, x_, v_)
    logr = logpdf_aug_target(prob, K, x_, v_) - logpdf_aug_target(prob, K, x, ṽ)

    loguã = log(ua) + logr

    if loguã > 0
        # reject
        x, ṽ = x_, v_
    else
        # accept
        ua = exp(loguã)
    end

    v = _invcdf_v_given_x(K, x, uv)
    uv = _cdf_v_given_x(K, x, ṽ)

    uv, ua = inv_update_uniform(unif_mixer, uv, ua, t)
    return x, v, uv, ua
end


function forward_with_logdetjac(
    prob::MixFlowProblem, K::RWMH{T}, unif_mixer::ErgodicShift,
    x::AbstractVector{T}, v::AbstractVector{T}, uv::AbstractVector{T}, ua::T,
    t::Int,
) where {T<:Real}    
    xn, vn, uvn, uan = forward(prob, K, unif_mixer, x, v, uv, ua, t)
    logabsjac = logpdf_aug_target(prob, K, x, v) - logpdf_aug_target(prob, K, xn, vn)
    return xn, vn, uvn, uan, logabsjac
end


 

