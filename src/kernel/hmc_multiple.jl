# using Gaussian momentum 
struct HMCmultiple{T} <: MultivariateInvolutiveKernel
    ϵs::AbstractVector{T}
    ξv::AbstractMatrix{T}
    ξa::AbstractVector{T}
end
function HMCmultiple(ϵs::AbstractVector{T}, dims::Int) where T
    n_ϵs = length(ϵs) 
    ξv = rand(T, dims, n_ϵs)
    ξa = rand(T, n_ϵs)
    return HMCmultiple(ϵs, ξv, ξa)
end

_n_ϵs(K::HMCmultiple) = length(K.ϵs)


function _dist_v_given_x(::HMCmultiple{T}, ::MixFlowProblem, x::AbstractVector{T}) where T
    dim = length(x)
    return MvNormal(zeros(T, dim), I)
end
_rand_v_given_x(::HMCmultiple{T}, ::MixFlowProblem, x::AbstractVector{T}) where T = randn(length(x))
# _rand_v_given_x(::HMCmultiple{T}, ::MixFlowProblem, x::AbstractVector{T}, n::Int) where T = randn(length(x), n)
# _cdf_v_given_x(::HMCmultiple{T}, ::MixFlowProblem, ::AbstractVector{T}, v::AbstractVector{T}) where T = normcdf.(v)
# _invcdf_v_given_x(::HMCmultiple{T}, ::MixFlowProblem, ::AbstractVector{T}, uv::AbstractVector{T}) where T = norminvcdf.(uv)

function _rand_joint_reference(prob::MixFlowProblem, K::HMCmultiple)
    dims = LogDensityProblems.dimension(prob.target)
    x = rand(prob.reference)
    v = _rand_v_given_x(K, prob, x)
    uv = rand(dims)
    ua = rand()
    return x, v, uv, ua
end

function _involution_i(K::HMCmultiple{T}, ∇ℓπ, i::Int, x::AbstractVector{T}, v::AbstractVector{T}) where T
    ϵ = K.ϵs[i]
    # single leapfrog step
    x_, v_ = _leapfrog(∇ℓπ, ϵ, 1, x, v) 
    # flip momentum to ensure involutivity
    return x_, -v_
end

# just need to dispatch single forward and backward

function _forward_i(
    prob::MixFlowProblem, K::HMCmultiple{T}, ∇ℓπ,  i::Int,
    x::AbstractVector{T}, v::AbstractVector{T}, uv::AbstractVector{T}, ua::T,
) where {T<:Real}

    # update the uniform auxiliary variables deterministically
    uv = _ergodic_shift.(uv, @view(K.ξv[:, i])) 
    ua = _ergodic_shift(ua, K.ξa[i])

    uv_ = normcdf.(v)
    ṽ = norminvcdf.(uv)
    x_, v_ = _involution_i(K, ∇ℓπ, i, x, ṽ)

    
    # TODO: notice that the following log ratio does not include the jacobian term of the involution map
    # this is because that for RWMH, MALA, HMC, the involution map has unit jacobian
    logr = logpdf_aug_target(prob, K, x_, v_) - logpdf_aug_target(prob, K, x, ṽ)

    if check_acc(ua, logr)
        ua = exp(log(ua) - logr)
        return x_, v_, uv_, ua
    else
        return x, ṽ, uv_, ua
    end
end

function forward(
    prob::MixFlowProblem, K::HMCmultiple{T}, unif_mixer::AbstractUnifMixer,
    x::AbstractVector{T}, v::AbstractVector{T}, uv::AbstractVector{T}, ua::T,
    t::Int,
) where {T<:Real}
    ∇ℓπ = Base.Fix1(∇logpdf_target, prob)
    n_ϵs = _n_ϵs(K)

    # refresh uniform aux variables
    uv, ua = update_uniform(unif_mixer, uv, ua, t)

    for i in 1:n_ϵs
        x, v, uv, ua = _forward_i(prob, K, ∇ℓπ, i, x, v, uv, ua)
    end
    return x, v, uv, ua, nothing
end


function _inverse_i(
    prob::MixFlowProblem, K::HMCmultiple{T}, ∇ℓπ, i::Int,
    x_::AbstractVector{T}, v_::AbstractVector{T}, uv::AbstractVector{T}, ua::T,
) where {T<:Real}

    x, ṽ = _involution_i(K, ∇ℓπ, i, x_, v_)
    # TODO: notice that the following log ratio does not include the jacobian term of the involution map
    # this is because that for RWMH, MALA, HMC, the involution map has unit jacobian
    logr = logpdf_aug_target(prob, K, x_, v_) - logpdf_aug_target(prob, K, x, ṽ)

    loguã = log(ua) + logr

    if loguã > 0
        # reject
        x, ṽ = x_, v_
    else
        # accept
        ua = exp(loguã)
    end

    v = norminvcdf.(uv)
    uv = normcdf.(ṽ)

    # inv update the uniform auxiliary variables deterministically
    ua = _inv_ergodic_shift(ua, K.ξa[i])
    uv = _inv_ergodic_shift.(uv, @view(K.ξv[:, i])) 
    return x, v, uv, ua
end

function inverse(
    prob::MixFlowProblem, K::HMCmultiple{T}, unif_mixer::AbstractUnifMixer,
    x_::AbstractVector{T}, v_::AbstractVector{T}, uv::AbstractVector{T}, ua::T,
    t::Int,
) where {T<:Real}
    ∇ℓπ = Base.Fix1(∇logpdf_target, prob)
    n_ϵs = _n_ϵs(K)

    for i in n_ϵs:-1:1
        x_, v_, uv, ua = _inverse_i(prob, K, ∇ℓπ, i, x_, v_, uv, ua)
    end

    uv, ua = inv_update_uniform(unif_mixer, uv, ua, t)
    return x_, v_, uv, ua, nothing
end

