# time-inhomogeneous mixflow with IRF but simulate from past (linear density cost, cant do trajectory sampling)
struct BackwardIRFMixFlow <: AbstractFlowType 
    flow_length::Int
end

function iid_sample(flow::BackwardIRFMixFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer)
    T = rand(0:flow.flow_length) 
    x0, v0, uv0, ua0 = _rand_joint_reference(prob, K)  
    return simulate_from_past_T_step(prob, K, mixer, x0, v0, uv0, ua0, T)     
end

# function trajectory_sample(flow::AbstractFlowType, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer)
#     @warn "For backward flow, trajectory_sample is of quadratic cost. Use iid_sample instead in practice."
# end

function log_density_ratio_flow(
    flow::BackwardIRFMixFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer, 
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
        # this is much more stable as we avoid avaluating density of vdist in intermediate steps
        ℓr = _log_density_ratio(prob, x) 
        ℓs[t+1] = ℓr 
    end
    return logsumexp(ℓs) - log(T+1) 
end

function log_density_ratio_flow_sweep(
    flow::BackwardIRFMixFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer, 
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
        # this is much more stable as we avoid avaluating density of vdist in intermediate steps
        ℓr = _log_density_ratio(prob, x) 
        ℓs[t+1] = ℓr 
    end
    return logmeanexp_sweep(ℓs) 
end


