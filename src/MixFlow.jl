module MixFlow

using LinearAlgebra, Distributions, Random, StatsBase, ProgressMeter
using LogExpFunctions, IrrationalConstants, SpecialFunctions, StatsFuns 
using LogDensityProblems, ADTypes

using Base.Threads: @threads

# setup mixflow problem with specified reference and target
# all wrapped in logdensityprobs
struct MixFlowProblem{F,T}
    reference::F
    target::T
    function MixFlowProblem(reference::F, target::T) where {F, T}
        if LogDensityProblems.dimension(reference) != LogDensityProblems.dimension(target)
            throw(ArgumentError("The reference and target densities must have the same dimension."))
        end
        new{F, T}(reference, target)
    end
end
LogDensityProblems.dimension(prob::MixFlowProblem) = LogDensityProblems.dimension(prob.target)

logdensity_reference(prob::MixFlowProblem, x) = LogDensityProblems.logdensity(prob.reference, x)
logdensity_target(prob::MixFlowProblem, x) = LogDensityProblems.logdensity(prob.target, x)
∇logpdf_target(prob::MixFlowProblem, x) = LogDensityProblems.logdensity_and_gradient(prob.target, x)[2]

function iid_sample end
iid_sample_reference(prob::MixFlowProblem, n::Int) = iid_sample(prob.reference, n)

export logdensity_reference, logdensity_target, MixFlowProblem
export iid_sample_reference, iid_sample


# invertible ergodic shift or other refreshment (e.g., alnold cat mao) that refresh uniform aux variables
abstract type AbstractUnifMixer end

include("uniform_mixer.jl")
export AbstractUnifMixer, ErgodicShift, RandomShift, ErgodicShift1D, RandomShift1D
export _ergodic_shift, _inv_ergodic_shift

# involutive mcmc kernel that defines the involutive IRF mapping
abstract type InvolutiveKernel end
abstract type UnivariateInvolutiveKernel<:InvolutiveKernel end
abstract type MultivariateInvolutiveKernel<:InvolutiveKernel end

# check mh acceptance condition
check_acc(ua, logr) = log(ua) > logr ? false : true

# to be dispatched for each kernel
function _dist_v_given_x end
function _cdf_v_given_x end
function _invcdf_v_given_x end
function _rand_v_given_x end
function _involution end

logpdf_aug_target(prob::MixFlowProblem, K::InvolutiveKernel, x, v) =
    logdensity_target(prob, x) + logpdf(_dist_v_given_x(K, prob, x), v)

logpdf_aug_reference(prob::MixFlowProblem, K::InvolutiveKernel, x, v) =
    logdensity_reference(prob, x) + logpdf(_dist_v_given_x(K, prob, x), v)

function forward(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, unif_mixer::AbstractUnifMixer,
    x::AbstractVector{T}, v::AbstractVector{T}, uv::AbstractVector{T}, ua::T,
    t::Int,
) where {T<:Real}
    # refresh uniform aux variables
    uv, ua = update_uniform(unif_mixer, uv, ua, t)

    # involutive mcmc step
    uv_ = _cdf_v_given_x(K, prob, x, v)
    ṽ = _invcdf_v_given_x(K, prob, x, uv)
    # println("uv_", uv_)
    # println("ṽ", ṽ)
    x_, v_ = _involution(K, prob, x, ṽ)
logr = logpdf_aug_target(prob, K, x_, v_) - logpdf_aug_target(prob, K, x, ṽ)

    if check_acc(ua, logr)
        ua = exp(log(ua) - logr)
        # println("ua", ua)
        acc = true
        return x_, v_, uv_, ua, acc
    else
        acc = false
        return x, ṽ, uv_, ua, acc
    end
end

function inverse(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, unif_mixer::AbstractUnifMixer,
    x_::AbstractVector{T}, v_::AbstractVector{T}, uv::AbstractVector{T}, ua::T,
    t::Int,
) where {T<:Real}
    x, ṽ = _involution(K, prob, x_, v_)
    logr = logpdf_aug_target(prob, K, x_, v_) - logpdf_aug_target(prob, K, x, ṽ)

    loguã = log(ua) + logr

    if loguã > 0
        # reject
        x, ṽ = x_, v_
        acc = false
    else
        # accept
        ua = exp(loguã)
        acc = true
    end

    v = _invcdf_v_given_x(K, prob, x, uv)
    uv = _cdf_v_given_x(K, prob, x, ṽ)

    uv, ua = inv_update_uniform(unif_mixer, uv, ua, t)
    return x, v, uv, ua, acc
end

function forward_with_logdetjac(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, unif_mixer::AbstractUnifMixer,
    x::AbstractVector{T}, v::AbstractVector{T}, uv::AbstractVector{T}, ua::T,
    t::Int,
) where {T <: Real}
    xn, vn, uvn, uan, acc = forward(prob, K, unif_mixer, x, v, uv, ua, t)
    logabsjac = logpdf_aug_target(prob, K, x, v) - logpdf_aug_target(prob, K, xn, vn)
    return xn, vn, uvn, uan, acc, logabsjac
end

function inverse_with_logdetjac(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, unif_mixer::AbstractUnifMixer,
    x_::AbstractVector{T}, v_::AbstractVector{T}, uv::AbstractVector{T}, ua::T, 
    t::Int,
) where {T<:Real}
    x, v, uv, ua, acc = inverse(prob, K, unif_mixer, x_, v_, uv, ua, t)
    logabsjac = logpdf_aug_target(prob, K, x_, v_) - logpdf_aug_target(prob, K, x, v)
    return x, v, uv, ua, acc, logabsjac
end

export forward, inverse, forward_with_logdetjac, inverse_with_logdetjac
export logpdf_aug_target

using StatsFuns: normcdf, norminvcdf

include("kernel/rwmh1d.jl")
include("kernel/rwmh.jl")
include("kernel/mala.jl")
include("kernel/hmc_uncorrect.jl")
include("kernel/hmc.jl")

export _involution 

export RWMH1D
export RWMH
export uncorrectHMC, HMC, MALA

# there are some weird flow types 
# this will influence how we compute the density and so on
abstract type AbstractFlowType end

# time-inhomogeneous mixflow with IRF (quadrtic density cost)
struct RandomMixFlow <: AbstractFlowType 
    flow_length::Int
end
# time-inhomogeneous mixflow with IRF but simulate the inverse (linear density cost, cant do trajectory sampling)
struct RandomInverseMixFlow <: AbstractFlowType 
    flow_length::Int
end
# M short runs, no mix
struct RandomFlow <: AbstractFlowType 
    flow_length::Int
    num_flows::Int # M
end

function inverse_T_step(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, mixer::AbstractUnifMixer,
    x::AbstractVector{T}, v::AbstractVector{T}, uv::Union{AbstractVector{T}, Nothing}, ua::Union{T,Nothing},
    steps::Int,
) where T 
    for t in steps:-1:1
        x, v, uv, ua, _ = inverse(prob, K, mixer, x, v, uv, ua, t)
    end
    return x, v, uv, ua
end

function forward_T_step(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, mixer::AbstractUnifMixer,
    x::AbstractVector{T}, v::AbstractVector{T}, uv::Union{AbstractVector{T}, Nothing}, ua::Union{T,Nothing},
    steps::Int,
) where T 
    for t in 1:steps
        x, v, uv, ua, _ = forward(prob, K, mixer, x, v, uv, ua, t)
    end
    return x, v, uv, ua
end

function simulate_from_past_T_step(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, mixer::AbstractUnifMixer,
    x::AbstractVector{T}, v::AbstractVector{T}, uv::Union{AbstractVector{T}, Nothing}, ua::Union{T,Nothing},
    steps::Int,
) where T  
    for t in steps:-1:1
        x, v, uv, ua, _ = forward(prob, K, mixer, x, v, uv, ua, t)
    end
    return x, v, uv, ua
end

iid_sample(
    flow::AbstractFlowType, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer, nsample::Int
) = [iid_sample(flow, prob, K, mixer) for _ in 1:nsample]

function _elbo_single(
    flow::AbstractFlowType, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer, 
    x, v, uv, ua,
)
    el = logpdf_aug_target(prob, K, x, v) - log_density_flow(flow, prob, K, mixer, x, v, uv, ua)
    return el
end

function elbo(
    flow::AbstractFlowType, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer, 
    nsample::Int,
)
    samples = iid_sample(flow, prob, K, mixer, nsample)
    els = zeros(nsample)
    @threads for i in 1:nsample
        els[i] = _elbo_single(flow, prob, K, mixer, samples[i]...)
    end
    # els = map(x -> _elbo_single(flow, prob, K, mixer, x...), samples)
    return mean(els)
end

include("random_bwd_mixflow.jl")
include("deterministic_mixflow.jl")
export elbo, log_density_flow, simulate_from_past_T_step 
export RandomBackwardMixFlow, DeterministicMixFlow

end
