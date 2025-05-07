using Pigeons: header, render_report_cell, hr

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
    normalize!(ws)
    return ws
end 
_ess(probabilities::AbstractVector) = 1 / sum(abs2, probabilities)

"""
Struct that stores collection of iid particles and also diagnostic information wrt weights

Notice that logweights are assumed to be ℓπ(xi) - ℓq_flow(xi), the importance sampling log weights, 
while the _log_density_ratio_flow function computes ℓq_flow(xi) - ℓπ(xi)
"""
struct Particles
    flow
    states # iid samples (D × N) from the flow
    probabilities
    log_weights
    logZ
    elbo
    ess
end

n_particles(p::Particles) = length(p.log_weights)
function Particles(flow::AbstractFlowType, states::AbstractMatrix{T}, log_weights::AbstractVector{T}) where {T}
    el = mean(log_weights)
    prs = exp_normalize(log_weights)
    eff_sample_size = _ess(prs)
    logZ = log_normalization_constant(log_weights)
    return Particles(flow, states, prs, log_weights, logZ, el, eff_sample_size)
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


#######################
# fancy print out
#######################
all_reports() = [  
        # header with    # lambda expression used to 
        # width of 9     # compute that report item
        "    T     "   => a -> a.flow.flow_length,
        "n_ensemble"   => a -> hasfield(typeof(a.flow), :num_flows) ? a.flow.num_flows : 1,
        "    N     "   => a -> n_particles(a.particles),
        # "  time(s) "   => a -> a.full_timing, 
        # "  allc(B) "   => a -> a.timing.bytes,
        "   ess    "   => a -> a.ess,
        "   elbo   "   => a -> a.elbo,
        "log(Z₁/Z₀)"   => a -> a.logZ,
    ]

function report(a::Particles)
    reports = reports_available(a)
    header(reports) 
    println(
        join(
            map(
                pair -> render_report_cell(pair[2], a),
                reports),
            " "
        ))
    hr(reports, "─")
    return nothing
end
report(a::Particles, show_report::Bool) = show_report ? report(a) : nothing

function reports_available(a::Particles)
    result = Pair[] 
    for pair in all_reports() 
        try 
            (pair[2])(a) 
            push!(result, pair)
        catch 
            # some recorder has not been used, skip
        end
    end
    return result
end

# process_result(a::Particles, method::String) = begin 
#     log_normalizer = hasfield(typeof(a), :logZ) ? a.logZ : a.particles.log_normalization
#     Dict(
#         "method" => method,
#         "T" => length(a.schedule),
#         "N" => n_particles(a.particles),
#         "time(s)" => a.full_timing,
#         # "allc(B)" => a.timing.bytes,
#         "ess" => ess(a.particles),
#         "elbo" => a.particles.elbo,
#         "Λ" => isnothing(a.barriers) ? "NA" : a.barriers.globalbarrier,
#         "log(Z₁/Z₀)"   => log_normalizer,
#     )
# end


