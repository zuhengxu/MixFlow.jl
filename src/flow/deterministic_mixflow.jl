# typical mixflow from time homogenous mapping
struct DeterministicMixFlow <: AbstractFlowType 
    flow_length::Int
end

function iid_sample(flow::DeterministicMixFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer)
    T = rand(0:flow.flow_length) 
    x0, v0, uv0, ua0 = _rand_joint_reference(prob, K)  
    return forward_T_step(prob, K, mixer, x0, v0, uv0, ua0, T) 
end

function trajectory_sample(flow::DeterministicMixFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer)
    samples = []
    sample = _rand_joint_reference(prob, K)  
    # step 0
    push!(samples, map(copy, sample))
    for _ in 1:flow.flow_length
        sample = forward(prob, K, mixer, sample...)
        push!(samples, sample)
    end
    return samples
end

function log_density_ratio_flow(
    flow::DeterministicMixFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer, 
    x, v, uv, ua,
)
    T = flow.flow_length
    # ℓπ = logpdf_aug_target(prob, K, x, v)
    ℓs = zeros(T+1)

    # the zero-th step
    lr0 = _log_density_ratio(prob, x)
    ℓs[1] = lr0

    for t in 1:T
        x, v, uv, ua, _ = inverse(prob, K, mixer, x, v, uv, ua, t)
        # here we use the property that any measure preserving map has jacobian π(x)/π(T_inv x)
        ℓr = _log_density_ratio(prob, x) 
        ℓs[t+1] = ℓr
    end
    return logsumexp(ℓs) - log(T+1) 
end

# For non-measure-preserving flows, we need to compute the density incrementally
function log_density_ratio_flow(
    flow::DeterministicMixFlow, prob::MixFlowProblem, K::uncorrectHMC, mixer::AbstractUnifMixer, 
    x, v, uv, ua,
)
    T = flow.flow_length
    ℓπ = logpdf_aug_target(prob, K, x, v)

    logJ = 0.0
    ℓs = zeros(T+1)

    # the zero-th step
    ℓ0 = logpdf_aug_reference(prob, K, x, v)
    ℓs[1] = ℓ0

    for t in 1:T
        x, v, uv, ua, _, logjac = inverse_with_logdetjac(prob, K, mixer, x, v, uv, ua, t)
        logJ += logjac
        ℓ = logpdf_aug_reference(prob, K, x, v) + logJ
        ℓs[t+1] = ℓ
    end
    ℓflow = logsumexp(ℓs) - log(T+1) 
    return ℓflow - ℓπ
end

function log_density_ratio_flow_sweep(
    flow::DeterministicMixFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer, 
    x, v, uv, ua,
)
    T = flow.flow_length
    ℓs = zeros(T+1)

    # the zero-th step
    lr0 = _log_density_ratio(prob, x)
    ℓs[1] = lr0

    for t in 1:T
        x, v, uv, ua, _ = inverse(prob, K, mixer, x, v, uv, ua, t)
        # here we use the property that any measure preserving map has jacobian π(x)/π(T_inv x)
        ℓr = _log_density_ratio(prob, x) 
        ℓs[t+1] = ℓr
    end
    return logmeanexp_sweep(ℓs) 
end

function log_density_ratio_flow_sweep(
    flow::DeterministicMixFlow, prob::MixFlowProblem, K::uncorrectHMC, mixer::AbstractUnifMixer, 
    x, v, uv, ua,
)
    T = flow.flow_length
    ℓπ = logpdf_aug_target(prob, K, x, v)

    logJ = 0.0
    ℓs = zeros(T+1)

    # the zero-th step
    ℓ0 = logpdf_aug_reference(prob, K, x, v)
    ℓs[1] = ℓ0

    for t in 1:T
        x, v, uv, ua, _, logjac = inverse_with_logdetjac(prob, K, mixer, x, v, uv, ua, t)
        logJ += logjac
        ℓ = logpdf_aug_reference(prob, K, x, v) + logJ
        ℓs[t+1] = ℓ
    end
    ℓflows = logmeanexp_sweep(ℓs)
    return ℓflows .- ℓπ
end
