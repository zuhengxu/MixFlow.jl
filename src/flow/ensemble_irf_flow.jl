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
    # just to make sure that we have the right number of ensembles
    @assert length(EMixers) == flow.num_flows

    ℓs = map(mixer -> log_density_ratio_flow(flow, prob, K, mixer, x, v, uv, ua), EMixers)
    return logsumexp(ℓs) - log(flow.num_flows)
end

function log_density_ratio_flow_sweep_ensemble(
    flow::EnsembleIRFFlow, prob::MixFlowProblem, K::InvolutiveKernel, EMixers::StructArray{ErgodicShift},
    x, v, uv, ua,
)   
    # just to make sure that we have the right number of ensembles
    @assert length(EMixers) == flow.num_flows

    ℓs = map(mixer -> log_density_ratio_flow(flow, prob, K, mixer, x, v, uv, ua), EMixers)
    return logmeanexp_sweep(ℓs)
end

# function log_density_ratio_flow_sweep_both(
#     flow::EnsembleIRFFlow, prob::MixFlowProblem, K::InvolutiveKernel, Emixers::StructArray{ErgodicShift}, 
#     x, v, uv, ua,
# )
#     T = flow.flow_length
#     B = flow.num_flows
#     ℓs = zeros(B, T+1)

#     # the zero-th step
#     lr0 = _log_density_ratio(prob, x)
#     ℓs[:, 1] .= lr0

#     for b in 1:B
#         local mixer = Emixers[b]
#         for t in 1:T
#             # backward process for the inverse
#             # this results in a quadratic cost in density evluation
#             xt, _, _, _ = inverse_T_step(prob, K, mixer, x, v, uv, ua, t)
#             # here we use the property that any measure preserving map T has jacobian π(x)/π(T_inv x)
#             # this is much more stable as we avoid avaluating density of vdist in intermediate steps
#             ℓr = _log_density_ratio(prob, xt) 
#             ℓs[b, t+1] = ℓr
#         end
#     end

#     ℓs_ensemble_sweep = mapslices(logmeanexp_sweep, ℓs, dims=1)
#         # vec(logsumexp(ℓs, dims=1)) .- log(B)
#     return ℓs_ensemble_sweep
# end


