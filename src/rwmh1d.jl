using StatsFuns: normcdf, norminvcdf

struct RWMH1D{T} <: UnivariateInvolutiveKernel
    σ::T
end
RWMH1D() = RWMH1D(1.0)

involution(::RWMH1D{T}, x::T, v::T) where T = (v, x)
_dist_v_given_x(K::RWMH1D, x) = Normal(x, K.σ)
_rand_v_given_x(K::RWMH1D, x) = rand(_dist_v_given_x(K, x))
_rand_v_given_x(K::RWMH1D, x, n::Int) = rand(_dist_v_given_x(K, x), n)

function forward(
    prob::MixFlowProblem, K::RWMH1D{T}, unif_mixer::ErgodicShift1D,
    x::T, v::T, uv::T, ua::T,
    t::Int,
) where {T<:Real}
    # refresh uniform aux variables
    uv, ua = update_uniform(unif_mixer, uv, ua, t)

    # involutive mcmc step
    # dist_v = _dist_v_given_x(K, x)
    uv_ = normcdf(x, K.σ, v)
    ṽ = norminvcdf(x, K.σ, uv)
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
    prob::MixFlowProblem, K::RWMH1D{T}, unif_mixer::ErgodicShift1D,
    x_::T, v_::T, uv::T, ua::T,
    t::Int,
) where {T<:Real}
    x, ṽ = involution(K, x_, v_)
    logr = logpdf_aug_target(prob, K, x_, v_) - logpdf_aug_target(prob, K, x, ṽ)

    loguã = log(ua) + logr

    if loguã > 0
        x, ṽ = x_, v_
    else
        ua = exp(loguã)
    end

    # dist_v = _dist_v_given_x(K, x)
    v = norminvcdf(x, K.σ, uv)
    uv = normcdf(x, K.σ, ṽ)

    uv, ua = inv_update_uniform(unif_mixer, uv, ua, t)
    return x, v, uv, ua
end

# function forward_with_logdetjac(
#     prob::MixFlowProblem, K::RWMH1D{T}, unif_mixer::ErgodicShift1D,
#     x::T, v::T, uv::T, ua::T,
#     t::Int,
# ) where {T<:Real}
#     xn, vn, uvn, uan = forward(prob, K, unif_mixer, x, v, uv, ua, t)
#     logabsjac = logpdf_aug_target(prob, K, x, v) - logpdf_aug_target(prob, K, xn, vn)
#     return xn, vn, uvn, uan, logabsjac
# end

# function inverse_with_logdetjac(
#     prob::MixFlowProblem, K::RWMH1D{T}, unif_mixer::ErgodicShift1D,
#     x_::T, v_::T, uv::T, ua::T, 
#     t::Int,
# ) where {T<:Real}
#     x, v, uv, ua = inverse(prob, K, unif_mixer, x_, v_, uv, ua, t)
#     logabsjac = logpdf_aug_target(prob, K, x_, v_) - logpdf_aug_target(prob, K, x, v)
#     return x, v, uv, ua, logabsjac
# end

# function forward_trajectory(
#     prob::MixFlowProblem, K::RWMH1D{T}, unif_mixer::ErgodicShift1D,
#     x::T, v::T, uv::T, ua::T, 
# ) where {T<:Real}
#     nsteps = ntransitions(unif_mixer)
#     xs = zeros(T, nsteps)
#     vs = zeros(T, nsteps)
#     uvs = zeros(T, nsteps)
#     uas = zeros(T, nsteps)

#     for t in 1:nsteps
#         # simulate forward
#         x, v, uv, ua = forward_with_logdetjac(prob, K, unif_mixer, x, v, uv, ua, t)
#         xs[t] = x
#         vs[t] = v
#         uvs[t] = uv
#         uas[t] = ua
#     end
#     return xs, vs, uvs, uas
# end

# function logpdf_last(
#     prob::MixFlowProblem, K::InvolutiveKernel, unif_mixer::AbstractUnifMixer,
#     x::T, v::T, uv::T, ua::T
# ) where {T<:Real}
#     nsteps = ntransitions(unif_mixer)
#     x0, v0 = copy(x), copy(v)

#     for t in nsteps:-1:1
#         # simulate backwards
#         x, v, uv, ua = inverse(prob, K, unif_mixer, x, v, uv, ua, t)
#     end
#     logJ = logpdf_aug_target(prob, K, x0, v0) - logpdf_aug_target(prob, K, x, v)
#     return logpdf_aug_target(prob, K, x, v) + logJ
# end

# function logpdf_intermediate(
#     prob::MixFlowProblem, K::InvolutiveKernel, unif_mixer::AbstractUnifMixer,
#     x::T, v::T, uv::T, ua::T
# ) where {T<:Real}

#     logJ = zero(T)
#     nsteps = ntransitions(unif_mixer)
#     ls = zeros(T, nsteps)

#     for t in nsteps:-1:1
#         # simulate backwards
#         x, v, uv, ua, logabsjac = inverse_with_logdetjac(prob, K, unif_mixer, x, v, uv, ua, t)
#         # accumulate jacobian
#         logJ += logabsjac
#         # compute logpdf for each mixture component
#         ℓ = logpdf_aug_target(prob, K, x, v) + logJ
#         ls[t] = ℓ
#     end
#     return ls
# end

# function logpdf_mixflow(
#     prob::MixFlowProblem, K::InvolutiveKernel, unif_mixer::AbstractUnifMixer,
#     x::T, v::T, uv::T, ua::T
# ) where {T<:Real}
#     nsteps = ntransitions(unif_mixer)
#     ℓs = logpdf_intermediate(prob, K, unif_mixer, x, v, uv, ua)
#     return logsumexp(ℓs) - log(nsteps)
# end

