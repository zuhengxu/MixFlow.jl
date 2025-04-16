module MixFlow

using LinearAlgebra, Distributions, Random, StatsBase, ProgressMeter 
using LogExpFunctions, IrrationalConstants, SpecialFunctions
using LogDensityProblems, ADTypes

using Base.Threads: @threads
using StatsFuns: normcdf, norminvcdf
using Statistics
using StructArrays

# import mcmc sampler from other libraries
using AdvancedHMC
using AdvancedMH

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

# ensure numerical stability, mapping infiniteness to zero
ensure_finite(x::Real) = isfinite(x) ? x : zero(x)
# ensure_finite(x::Real, v::AbstractVecOrMat) = isfinite(x) ? v : zeros(size(v))
logdensity_reference(prob::MixFlowProblem, x) = LogDensityProblems.logdensity(prob.reference, x)
logdensity_target(prob::MixFlowProblem, x) = LogDensityProblems.logdensity(prob.target, x)
∇logpdf_target(prob::MixFlowProblem, x) = LogDensityProblems.logdensity_and_gradient(prob.target, x)[2]

function _log_density_ratio(prob::MixFlowProblem, x::T) where T
    ℓπ0 = logdensity_reference(prob, x)
    ℓπT = logdensity_target(prob, x)
    return ensure_finite(ℓπ0 - ℓπT)
end

function iid_sample end
iid_sample_reference(prob::MixFlowProblem, n::Int) = iid_sample(prob.reference, n)

export logdensity_reference, logdensity_target, MixFlowProblem
export iid_sample_reference, iid_sample

#################################################################################
# invertible ergodic shift or other refreshment (e.g., Alnold cat map) that refresh uniform aux variables
#################################################################################
abstract type AbstractUnifMixer end

include("uniform_mixer.jl")
export _ergodic_shift, _inv_ergodic_shift
export AbstractUnifMixer, ErgodicShift, RandomShift, ErgodicShift1D, RandomShift1D
export EnsembleErgodicShift, EnsembleRandomShift

#################################################################################
# involutive mcmc kernel that defines the involutive IRF mapping
#################################################################################
abstract type InvolutiveKernel end
abstract type UnivariateInvolutiveKernel<:InvolutiveKernel end
abstract type MultivariateInvolutiveKernel<:InvolutiveKernel end
export InvolutiveKernel, UnivariateInvolutiveKernel, MultivariateInvolutiveKernel

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

export logpdf_aug_target, logpdf_aug_reference

include("transform.jl")

export forward, inverse, forward_with_logdetjac, inverse_with_logdetjac
export simulate_from_past_T_step, forward_T_step
export forward_trajectory, backward_process_trajectory
export forward_trajectory_x, backward_process_trajectory_x

include("kernel/rwmh1d.jl")
include("kernel/rwmh.jl")
include("kernel/mala.jl")
include("kernel/hmc_uncorrect.jl")
include("kernel/hmc.jl")

export RWMH1D
export RWMH
export uncorrectHMC, HMC, MALA

function mcmc_sampler(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, x0::AbstractVector{T}, nsteps::Int,
) where {T<:Real}
    # Initialize the chain
    x = x0
    samples = zeros(T, length(x0), nsteps)
    samples[:, 1] .= x

    for i in 2:nsteps
        # Perform an involutive mcmc step
        x = mcmc_step(prob, K, x)
        samples[:, i] .= x
    end

    return samples
end

export mcmc_sampler, mcmc_step

# there are some weird flow types 
# this will influence how we compute the density and so on
abstract type AbstractFlowType end

# time-inhomogeneous mixflow with IRF but simulate the inverse (linear density cost, cant do trajectory sampling)
struct RandomInverseMixFlow <: AbstractFlowType 
    flow_length::Int
end

"""
log_density_ratio_flow( 
    flow::AbstractFlowType, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer, 
    x, v, uv, ua
)

Compute ℓqN(x) - ℓπ(x). Can be used for computing log_density_flow, IS weights, and elbo.
- For measure-preserving transformation T, we can use the property that any measure preserving map T has jacobian π(x)/π(T_inv x).
- For non-measure-preserving flows, we just compute ℓqN via change of var formula.
"""
function log_density_ratio_flow end
_log_importance_weight(args...) = -log_density_ratio_flow(args...)

function log_density_flow(
    flow::AbstractFlowType, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer,
    x, v, uv, ua,
)
    ℓπ = logpdf_aug_target(prob, K, x, v)
    logw = log_density_ratio_flow(flow, prob, K, mixer, x, v, uv, ua) 
    return logw + ℓπ
end

# iid sample from the flow distribution
iid_sample(
    flow::AbstractFlowType, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer, nsample::Int
) = [iid_sample(flow, prob, K, mixer) for _ in 1:nsample]
    
include("flow/irf_fwd_mixflow.jl")
include("flow/irf_bwd_mixflow.jl")
include("flow/deterministic_mixflow.jl")
include("flow/ensemble_irf_flow.jl")

export _log_importance_weight
export DeterministicMixFlow, IRFMixFlow, BackwardIRFMixFlow
export EnsembleIRFFlow


include("particles.jl")
export Particles

"""
Output Particles that stores iid samples of x, corresponding log importance weights, and other diagnostics.
"""
function mixflow(
    flow::AbstractFlowType, prob::MixFlowProblem, K::InvolutiveKernel, mixer::Union{M, StructArray{M}},
    nsample::Int; 
    show_report::Bool = true,
) where {M<:AbstractUnifMixer}
    dims = LogDensityProblems.dimension(prob)
    samples = zeros(dims, nsample)
    ℓws = zeros(nsample)
    @showprogress @threads for i in 1:nsample
        x, v, uv, ua = iid_sample(flow, prob, K, mixer) 
        samples[:, i] .= x
        # compupte log is weight
        ℓw = _log_importance_weight(flow, prob, K, mixer, x, v, uv, ua)
        ℓws[i] = ℓw
    end
    output = Particles(flow, samples, ℓws)
    report(output, show_report)
    return output
end

export mixflow

# function _log_importance_weight(
#     flow::AbstractFlowType, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer, 
#     x, v, uv, ua,
# )
#     # el = logpdf_aug_target(prob, K, x, v) - log_density_flow(flow, prob, K, mixer, x, v, uv, ua)
#     logw = log_density_ratio_flow(flow, prob, K, mixer, x, v, uv, ua) 
#     return -logw
# end

# function _elbo_batch(
#     flow::AbstractFlowType, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer, 
#     samples::Vector{Tuple{X, V, UV, UA}},
# ) where {X, V, UV, UA}

#     nsample = length(samples)
#     els = zeros(nsample)
#     @threads for i in 1:nsample
#         els[i] = _log_importance_weight(flow, prob, K, mixer, samples[i]...)
#     end
#     return mean(els)
# end

# function elbo(
#     flow::AbstractFlowType, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer, 
#     nsample::Int,
# )
#     samples = iid_sample(flow, prob, K, mixer, nsample)
#     els = zeros(nsample)
#     @threads for i in 1:nsample
#         els[i] = _log_importance_weight(flow, prob, K, mixer, samples[i]...)
#     end
#     return mean(els)
# end

end
