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

function iid_sample end
iid_sample_reference(prob::MixFlowProblem, n::Int) = iid_sample(prob.reference, n)

export logdensity_reference, logdensity_target, MixFlowProblem
export iid_sample_reference, iid_sample


# invertible ergodic shift or other refreshment (e.g., alnold cat mao) that refresh uniform aux variables
abstract type AbstractUnifMixer end

include("uniform_mixer.jl")
export AbstractUnifMixer, ErgodicShift, RandomShift, ErgodicShift1D, RandomShift1D
export _ergodic_shift, _inv_ergodic_shift


# there are some weird flow types 
# this will influence how we compute the density and so on
abstract type AbstractFlowType end

# typical mixflow from time homogenous mapping
struct DeterministicMixFlow <: AbstractFlowType end
# time-inhomogeneous mixflow with IRF (quadrtic density cost)
struct RandomMixFlow <: AbstractFlowType end
# time-inhomogeneous mixflow with IRF but simulate from past (linear density cost, cant do trajectory sampling)
struct RandomBwdMixFlow <: AbstractFlowType end
# time-inhomogeneous mixflow with IRF but simulate the inverse (linear density cost, cant do trajectory sampling)
struct RandomInverseMixFlow <: AbstractFlowType end
# K short runs, no mix
struct RandomFlow <: AbstractFlowType end



# involutive mcmc kernel that defines the involutive IRF mapping
abstract type InvolutiveKernel end
abstract type UnivariateInvolutiveKernel<:InvolutiveKernel end
abstract type MultivariateInvolutiveKernel<:InvolutiveKernel end

# check mh acceptance condition
check_acc(ua, logr) = log(ua) > logr ? false : true

function logpdf_aug_target end
function _dist_v_given_x end

logpdf_aug_target(prob::MixFlowProblem, K::InvolutiveKernel, x, v) =
    logdensity_target(prob, x) + logpdf(_dist_v_given_x(K, x), v)


function forward end
function inverse end
function log_density_flow end
function elbo end

function forward_with_logdetjac(
    prob::MixFlowProblem, K::UnivariateInvolutiveKernel, unif_mixer::AbstractUnifMixer,
    x::T, v::T, uv::T, ua::T,
    t::Int,
) where {T <: Real}
    xn, vn, uvn, uan = forward(prob, K, unif_mixer, x, v, uv, ua, t)
    logabsjac = logpdf_aug_target(prob, K, x, v) - logpdf_aug_target(prob, K, xn, vn)
    return xn, vn, uvn, uan, logabsjac
end
function forward_with_logdetjac(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, unif_mixer::AbstractUnifMixer,
    x::AbstractVector{T}, v::AbstractVector{T}, uv::AbstractVector{T}, ua::T,
    t::Int,
) where {T <: Real}
    xn, vn, uvn, uan = forward(prob, K, unif_mixer, x, v, uv, ua, t)
    logabsjac = logpdf_aug_target(prob, K, x, v) - logpdf_aug_target(prob, K, xn, vn)
    return xn, vn, uvn, uan, logabsjac
end

function inverse_with_logdetjac(
    prob::MixFlowProblem, K::UnivariateInvolutiveKernel, unif_mixer::AbstractUnifMixer,
    x_::T, v_::T, uv::T, ua::T, 
    t::Int,
) where {T<:Real}
    x, v, uv, ua = inverse(prob, K, unif_mixer, x_, v_, uv, ua, t)
    logabsjac = logpdf_aug_target(prob, K, x_, v_) - logpdf_aug_target(prob, K, x, v)
    return x, v, uv, ua, logabsjac
end
function inverse_with_logdetjac(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, unif_mixer::AbstractUnifMixer,
    x_::AbstractVector{T}, v_::AbstractVector{T}, uv::AbstractVector{T}, ua::T, 
    t::Int,
) where {T<:Real}
    x, v, uv, ua = inverse(prob, K, unif_mixer, x_, v_, uv, ua, t)
    logabsjac = logpdf_aug_target(prob, K, x_, v_) - logpdf_aug_target(prob, K, x, v)
    return x, v, uv, ua, logabsjac
end

export forward, inverse, forware_with_logdetjac, inverse_with_logdetjac
export logpdf_aug_target


include("rwmh1d.jl")
include("rwmh.jl")
include("hmc_uncorrect.jl")
include("hmc.jl")

export RWMH1D
export RWMH


end
