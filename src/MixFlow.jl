module MixFlow

using LinearAlgebra, Distributions, Random, StatsBase, ProgressMeter
using LogExpFunctions, IrrationalConstants, SpecialFunctions, StatsFuns
using LogDensityProblems, ADTypes

using Base.Threads: @threads


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

logdensity_reference(prob::MixFlowProblem) = LogDensityProblems.logdensity(prob.reference)
logdensity_target(prob::MixFlowProblem) = LogDensityProblems.logdensity(prob.target)

function iid_sample_reference end
function iid_sample end

export logdensity_reference, logdensity_target, MixFlowProblem
export iid_sample_reference, iid_sample

include("uniform_mixer.jl")
export AbstractUnifMixer, ErgodicShift, RandomShift 



abstract type InvolutiveKernel end

function logpdf_aug_target end

function forward end
function inverse end

function forward_with_logdetjac end
function inverse_with_logdetjac end



function sample_trajectory end

function logpdf_mixflow end
function logpdf_intermediate end
function elbo end

include("rwmh1d.jl")
export forward, inverse, logpdf_mixflow, logpdf_intermediate, logpdf_last


end
