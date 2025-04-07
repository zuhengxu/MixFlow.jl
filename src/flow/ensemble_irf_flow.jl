# M short runs, no mix
struct EnsembleIRFFlow <: AbstractFlowType 
    flow_length::Int
    num_flows::Int # M
end

function iid_sample(flow::EnsembleIRFFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::ErgodicShift)
    T, nchains = flow.flow_length, flow.num_flows
    x0, v0, uv0, ua0 = _rand_joint_reference(prob, K)  
    return forward_T_step(prob, K, mixer, x0, v0, uv0, ua0, T) 
end

function iid_sample(flow::EnsembleIRFFlow, prob::MixFlowProblem, K::InvolutiveKernel, EMixers::StructArray{ErgodicShift})
    T, nchains = flow.flow_length, flow.num_flows
    x0, v0, uv0, ua0 = _rand_joint_reference(prob, K)  
    m = rand(1:nchains) # random pick one chain
    return forward_T_step(prob, K, EMixers[m], x0, v0, uv0, ua0, T) 
end

function log_density_ratio_flow(
    flow::EnsembleIRFFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::ErgodicShift,
    x, v, uv, ua,
)   
    T = flow.flow_length
    x_inv, _, _, _ = inverse_T_step(prob, K, mixer, x, v, uv, ua, T)
    return _log_density_ratio(prob, x_inv)
end

function log_density_ratio_flow(
    flow::EnsembleIRFFlow, prob::MixFlowProblem, K::InvolutiveKernel, EMixers::StructArray{ErgodicShift},
    x, v, uv, ua,
)   
    ℓs = map(mixer -> log_density_ratio_flow(flow, prob, K, mixer, x, v, uv, ua), EMixers)
    return logsumexp(ℓs) - log(flow.num_flows)
end
