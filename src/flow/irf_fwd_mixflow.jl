# time-inhomogeneous mixflow with IRF (quadrtic density cost)
struct IRFMixFlow <: AbstractFlowType 
    flow_length::Int
end

function iid_sample(flow::IRFMixFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer)
    T = rand(0:flow.flow_length) 
    x0, v0, uv0, ua0 = _rand_joint_reference(prob, K)  
    return forward_T_step(prob, K, mixer, x0, v0, uv0, ua0, T) 
end

function log_density_ratio_flow(
    flow::IRFMixFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer, 
    x, v, uv, ua,
)
    T = flow.flow_length
    ℓs = zeros(T+1)

    # the zero-th step
    lr0 = _log_density_ratio(prob, x)
    ℓs[1] = lr0

    for t in 1:T
        # backward process for the inverse
        # this results in a quadratic cost in density evluation
        xt, _, _, _ = inverse_T_step(prob, K, mixer, x, v, uv, ua, t)
        # here we use the property that any measure preserving map T has jacobian π(x)/π(T_inv x)
        # this is much more stable as we avoid avaluating density of vdist in intermediate steps
        ℓr = _log_density_ratio(prob, xt) 
        ℓs[t+1] = ℓr
    end
    return logsumexp(ℓs) - log(T+1)
end



# function log_density_flow(
#     flow::IRFMixFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer, 
#     x, v, uv, ua,
# )
#     T = flow.flow_length
#     ℓπ = logpdf_aug_target(prob, K, x, v)
#     ℓs = []

#     # the zero-th step
#     lr0 = _log_density_ratio(prob, x)
#     push!(ℓs, lr0)

#     for t in 1:T
#         # backward process for the inverse
#         # this results in a quadratic cost in density evluation
#         for _ in t:1
#             x, v, uv, ua, _ = inverse(prob, K, mixer, x, v, uv, ua, t)
#         end
#         # here we use the property that any measure preserving map T has jacobian π(x)/π(T_inv x)
#         # this is much more stable as we avoid avaluating density of vdist in intermediate steps
#         ℓr = _log_density_ratio(prob, x) 
#         push!(ℓs, ℓr)
#     end
#     return logsumexp(ℓs) - log(T+1) + ℓπ
# end


