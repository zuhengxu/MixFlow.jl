struct RWMH1D{T} <: UnivariateInvolutiveKernel
    σ::T
end
RWMH1D() = RWMH1D(1.0)

_involution(::RWMH1D{T}, x::T, v::T) where T = (v, x)
_dist_v_given_x(K::RWMH1D, x) = Normal(x, K.σ)
_rand_v_given_x(K::RWMH1D, x) = rand(_dist_v_given_x(K, x))
_rand_v_given_x(K::RWMH1D, x, n::Int) = rand(_dist_v_given_x(K, x), n)

function _rand_joint_reference(prob::MixFlowProblem, K::RWMH1D)
    x = rand(prob.reference)
    v = _rand_v_given_x(K, x)
    uv = rand()
    ua = rand()
    return x, v, uv, ua
end

function forward(
    prob::MixFlowProblem, K::RWMH1D{T}, unif_mixer::ErgodicShift1D,
    x::T, v::T, uv::T, ua::T,
    t::Int,
) where {T<:Real}
    # refresh uniform aux variables
    uv, ua = update_uniform(unif_mixer, uv, ua, t)

    # involutive mcmc step
    uv_ = normcdf(x, K.σ, v)
    ṽ = norminvcdf(x, K.σ, uv)
    x_, v_ = _involution(K, x, ṽ)

    logr = logpdf_aug_target(prob, K, x_, v_) - logpdf_aug_target(prob, K, x, ṽ)

    if check_acc(ua, logr)
        ua = exp(log(ua) - logr)
        acc = true
        return x_, v_, uv_, ua, acc
    else
        acc = false
        return x, ṽ, uv_, ua, acc
    end
end

function inverse( 
    prob::MixFlowProblem, K::RWMH1D{T}, unif_mixer::ErgodicShift1D,
    x_::T, v_::T, uv::T, ua::T,
    t::Int,
) where {T<:Real}
    x, ṽ = _involution(K, x_, v_)
    logr = logpdf_aug_target(prob, K, x_, v_) - logpdf_aug_target(prob, K, x, ṽ)

    loguã = log(ua) + logr


    if loguã > 0
        acc = false
        x, ṽ = x_, v_
    else
        acc = true
        ua = exp(loguã)
    end

    v = norminvcdf(x, K.σ, uv)
    uv = normcdf(x, K.σ, ṽ)

    uv, ua = inv_update_uniform(unif_mixer, uv, ua, t)
    return x, v, uv, ua, acc
end

function forward_with_logdetjac(
    prob::MixFlowProblem, K::UnivariateInvolutiveKernel, unif_mixer::AbstractUnifMixer,
    x::T, v::T, uv::T, ua::T,
    t::Int,
) where {T <: Real}
    xn, vn, uvn, uan, acc = forward(prob, K, unif_mixer, x, v, uv, ua, t)
    logabsjac = logpdf_aug_target(prob, K, x, v) - logpdf_aug_target(prob, K, xn, vn)
    return xn, vn, uvn, uan, acc, logabsjac
end

function inverse_with_logdetjac(
    prob::MixFlowProblem, K::UnivariateInvolutiveKernel, unif_mixer::AbstractUnifMixer,
    x_::T, v_::T, uv::T, ua::T, 
    t::Int,
) where {T<:Real}
    x, v, uv, ua, acc = inverse(prob, K, unif_mixer, x_, v_, uv, ua, t)
    logabsjac = logpdf_aug_target(prob, K, x_, v_) - logpdf_aug_target(prob, K, x, v)
    return x, v, uv, ua, acc, logabsjac
end
