# typical mixflow from time homogenous mapping
struct DeterministicMixFlow <: AbstractFlowType 
    flow_length::Int
end

function iid_sample(flow::DeterministicMixFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer)
    T = rand(1:flow.flow_length)
    x0, v0, uv0, ua0 = _rand_joint_reference(prob, K)  
    return forward_T_step(prob, K, mixer, x0, v0, uv0, ua0, T) 
end

function trajectory_sample(flow::DeterministicMixFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer)
    samples = []
    sample = _rand_joint_reference(prob, K)  
    push!(samples, map(copy, sample))
    for i in 1:flow.flow_length
        sample = forward(prob, K, mixer, sample...)
        push!(samples, map(copy, sample))
    end
    return samples
end

function log_density_flow(
    flow::DeterministicMixFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer, 
    x, v, uv, ua,
)
    T = flow.flow_length
    logJ = 0.0
    ℓs = []
    for t in 1:T
        x, v, uv, ua, _, logjac = inverse_with_logdetjac(prob, K, mixer, x, v, uv, ua, t)
        logJ += logjac
        ℓ = logpdf_aug_reference(prob, K, x, v) + logJ
        push!(ℓs, ℓ)
    end
    return logsumexp(ℓs) - log(T)
end
