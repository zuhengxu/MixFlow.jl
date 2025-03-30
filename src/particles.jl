# code modified from https://github.com/alexandrebouchard/sais-gpu/blob/main/utils.jl

"""
logweights are assumed to be ℓπ(xi) - ℓq_flow(xi), the importance sampling log weights
"""
function log_normalize_weights(log_weights::AbstractVector{T}) where T
    ℓW = log_weights .- LogExpFunctions.logsumexp(log_weights)
    return ℓW
end

function ess_from_logweights(logw::AbstractVector{T}) where T
    ℓW = log_normalize_weights(logw)
    return exp(-LogExpFunctions.logsumexp(2 * ℓW))
end

function log_normalization_constant(logw::AbstractVector{T}) where T
    nparticles = length(logw)
    return LogExpFunctions.logsumexp(logw) - log(nparticles)
end

"""
compute sum_i exp(lw_i) fi
"""
expectation_from_logweights(ℓw_norm::AbstractVector{T}, fs::AbstractVector{T}) where {T} =
    dot(exp.(ℓw_norm), fs)

function normalize!(weights) 
    s = sum(weights)
    weights .= weights ./ s 
    return s
end

function exp_normalize(log_weights)
    m = maximum(log_weights)
    ws = exp.(log_weights .- m) 
    return m + log(normalize!(ws)) # return log normalized weights
end 
ess(probabilities::AbstractVector) = 1 / sum(abs2, probabilities)


struct Particles
    states # iid samples (D × N) from the flow
    probabilities
    log_weights
    logZ
    elbo
    ess
end

n_particles(p::Particles) = length(p.log_weights)
function Particles(states, log_weights)
    el = -mean(log_weights)
    prs = exp_normalize(log_weights)
    ess = ess(prs)
    logZ = log_normalization_constant(log_weights)
    return Particles(states, prs, log_weights, logZ, el, ess)
end

integrate(f::Function, p::Particles) =
    sum(1:n_particles(p)) do i 
        state = @view p.states[:, i] 
        f(state) * p.probabilities[i]
    end

∫ = integrate 

Statistics.mean(μ::Particles) = ∫(x -> x, μ) 
function Statistics.var(μ::Particles) 
    m = Statistics.mean(μ)
    return ∫(x -> (x - m).^2, μ)
end
Statistics.std(μ::Particles) = sqrt.(var(μ))
