# no mix, single flow, but wont converge
struct IRFFlow <: AbstractFlowType
    flow_length::Int
end

function iid_sample(
    flow::IRFFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer
)
    x0, v0, uv0, ua0 = _rand_joint_reference(prob, K)  
    return forward_T_step(prob, K, mixer, x0, v0, uv0, ua0, flow.flow_length) 
end

function log_density_ratio_flow(
    flow::IRFFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer,
    x, v, uv, ua,
)
    T = flow.flow_length
    x_inv, _, _, _ = inverse_T_step(prob, K, mixer, x, v, uv, ua, T)
    return _log_density_ratio(prob, x_inv)
end

# M short runs, no mix
struct EnsembleIRFFlow <: AbstractFlowType 
    flow_length::Int
    num_flows::Int # M
end

function iid_sample(flow::EnsembleIRFFlow, prob::MixFlowProblem, K::InvolutiveKernel, EMixers::StructArrays{ErgodicShift})
    T, nchains = flow.flow_length, flow.num_flows
    x0, v0, uv0, ua0 = _rand_joint_reference(prob, K)  
    m = rand(1:nchains) # random pick one chain
    mixer = EMixers[m]
    return forward_T_step(prob, K, mixer, x0, v0, uv0, ua0, T) 
end

# function log_density_ratio_flow(
#     flow::EnsembleIRFFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::StructArrays{ErgodicShift},
#     x, v, uv, ua,
# )   

# end
