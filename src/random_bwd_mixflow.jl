# time-inhomogeneous mixflow with IRF but simulate from past (linear density cost, cant do trajectory sampling)
struct RandomBackwardMixFlow <: AbstractFlowType 
    flow_length::Int
end
function inverse_T_step(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, mixer::AbstractUnifMixer,
    x::AbstractVector{T}, v::AbstractVector{T}, uv::Union{AbstractVector{T}, Nothing}, ua::Union{T,Nothing},
    steps::Int,
) where T 
    for t in steps:-1:1
        x, v, uv, ua, _ = inverse(prob, K, mixer, x, v, uv, ua, t)
    end
    return x, v, uv, ua
end

function simulate_from_past_T_step(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, mixer::AbstractUnifMixer,
    x::AbstractVector{T}, v::AbstractVector{T}, uv::Union{AbstractVector{T}, Nothing}, ua::Union{T,Nothing},
    steps::Int,
) where T  
    for t in steps:-1:1
        x, v, uv, ua, _ = forward(prob, K, mixer, x, v, uv, ua, t)
    end
    return x, v, uv, ua
end

function iid_sample(flow::RandomBackwardMixFlow, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer)
    T = rand(1:flow.flow_length) 
    x0, v0, uv0, ua0 = _rand_joint_reference(prob, K)  
    return simulate_from_past_T_step(prob, K, mixer, x0, v0, uv0, ua0, T) 
end

function trajectory_sample(flow::AbstractFlowType, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer)
    @warn "For backward flow, trajectory_sample is of quadratic cost. Use iid_sample instead in practice."
end

function log_density_flow(flow::AbstractFlowType, prob::MixFlowProblem, K::InvolutiveKernel, x, v, uv, ua)
    nothing
end

function elbo_single(flow::AbstractFlowType, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer, x, v, uv, ua)
    # x, v, uv, ua = iid_sample(flow, prob, K, mixer)
    el = log_density_flow(flow, prob, K, x, v, uv, ua) - logpdf_aug_target(prob, K, x, v)
    return el
end

function elbo(flow::AbstractFlowType, prob::MixFlowProblem, K::InvolutiveKernel, mixer::AbstractUnifMixer, nsample::Int)
    samples = iid_sample(flow, prob, K, mixer, nsample)
    els = map(x -> elbo_single(flow, prob, K, mixer, x...), samples)
    return mean(els)
end

function elbo_trajectory end

