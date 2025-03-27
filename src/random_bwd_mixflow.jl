# time-inhomogeneous mixflow with IRF but simulate from past (linear density cost, cant do trajectory sampling)
struct RandomBackwardMixFlow <: AbstractFlowType 
    flow_length::Int
end

function iid_sample(flow::RandomBackwardMixFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer)
    T = rand(1:flow.flow_length) 
    x0, v0, uv0, ua0 = _rand_joint_reference(prob, K)  
    return simulate_from_past_T_step(prob, K, mixer, x0, v0, uv0, ua0, T) 
end


function trajectory_sample(flow::AbstractFlowType, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer)
    @warn "For backward flow, trajectory_sample is of quadratic cost. Use iid_sample instead in practice."
end

function log_density_flow(
    flow::RandomBackwardMixFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer, 
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
