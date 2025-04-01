function forward(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, unif_mixer::AbstractUnifMixer,
    x::AbstractVector{T}, v::AbstractVector{T}, uv::AbstractVector{T}, ua::T,
    t::Int,
) where {T<:Real}
    # refresh uniform aux variables
    uv, ua = update_uniform(unif_mixer, uv, ua, t)

    # involutive mcmc step
    uv_ = _cdf_v_given_x(K, prob, x, v)
    ṽ = _invcdf_v_given_x(K, prob, x, uv)
    # println("uv_", uv_)
    # println("ṽ", ṽ)
    x_, v_ = _involution(K, prob, x, ṽ)
    logr = logpdf_aug_target(prob, K, x_, v_) - logpdf_aug_target(prob, K, x, ṽ)

    if check_acc(ua, logr)
        ua = exp(log(ua) - logr)
        # println("ua", ua)
        acc = true
        return x_, v_, uv_, ua, acc
    else
        acc = false
        return x, ṽ, uv_, ua, acc
    end
end

function inverse(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, unif_mixer::AbstractUnifMixer,
    x_::AbstractVector{T}, v_::AbstractVector{T}, uv::AbstractVector{T}, ua::T,
    t::Int,
) where {T<:Real}
    x, ṽ = _involution(K, prob, x_, v_)
    logr = logpdf_aug_target(prob, K, x_, v_) - logpdf_aug_target(prob, K, x, ṽ)

    loguã = log(ua) + logr

    if loguã > 0
        # reject
        x, ṽ = x_, v_
        acc = false
    else
        # accept
        ua = exp(loguã)
        acc = true
    end

    v = _invcdf_v_given_x(K, prob, x, uv)
    uv = _cdf_v_given_x(K, prob, x, ṽ)

    uv, ua = inv_update_uniform(unif_mixer, uv, ua, t)
    return x, v, uv, ua, acc
end

function forward_with_logdetjac(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, unif_mixer::AbstractUnifMixer,
    x::AbstractVector{T}, v::AbstractVector{T}, uv::AbstractVector{T}, ua::T,
    t::Int,
) where {T <: Real}
    xn, vn, uvn, uan, acc = forward(prob, K, unif_mixer, x, v, uv, ua, t)
    logabsjac = logpdf_aug_target(prob, K, x, v) - logpdf_aug_target(prob, K, xn, vn)
    return xn, vn, uvn, uan, acc, logabsjac
end

function inverse_with_logdetjac(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, unif_mixer::AbstractUnifMixer,
    x_::AbstractVector{T}, v_::AbstractVector{T}, uv::AbstractVector{T}, ua::T, 
    t::Int,
) where {T<:Real}
    x, v, uv, ua, acc = inverse(prob, K, unif_mixer, x_, v_, uv, ua, t)
    logabsjac = logpdf_aug_target(prob, K, x_, v_) - logpdf_aug_target(prob, K, x, v)
    return x, v, uv, ua, acc, logabsjac
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

function forward_T_step(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, mixer::AbstractUnifMixer,
    x::AbstractVector{T}, v::AbstractVector{T}, uv::Union{AbstractVector{T}, Nothing}, ua::Union{T,Nothing},
    steps::Int,
) where {T<:Real}
    for t in 1:steps
        x, v, uv, ua, _ = forward(prob, K, mixer, x, v, uv, ua, t)
    end
    return x, v, uv, ua
end

# function inverse_trajectory(
#     prob::MixFlowProblem, K::MultivariateInvolutiveKernel, mixer::AbstractUnifMixer,
#     x::AbstractVector{T}, v::AbstractVector{T}, uv::Union{AbstractVector{T}, Nothing}, ua::Union{T,Nothing},
#     steps::Int,
# ) where T 
#     sample_path = []
#     for t in steps:-1:1
#         x, v, uv, ua, _ = inverse(prob, K, mixer, x, v, uv, ua, t)
#         push!(sample_path, map(copy, (x, v, uv, ua)))
#     end
#     return sample_path
# end

function forward_trajectory(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, mixer::AbstractUnifMixer,
    x::AbstractVector{T}, v::AbstractVector{T}, uv::Union{AbstractVector{T}, Nothing}, ua::Union{T,Nothing},
    steps::Int,
) where {T<:Real}
    sample_path = []
    @showprogress for t in 1:steps
        push!(sample_path, (x, v, uv, ua))
        x, v, uv, ua, _ = forward(prob, K, mixer, x, v, uv, ua, t)
    end
    push!(sample_path, (x, v, uv, ua))
    return sample_path
end

function forward_trajectory_x(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, mixer::AbstractUnifMixer,
    x::AbstractVector{T}, v::AbstractVector{T}, uv::Union{AbstractVector{T}, Nothing}, ua::Union{T,Nothing},
    steps::Int,
) where {T<:Real}
    sample_path = zeros(T, length(x), steps+1)
    @showprogress for t in 1:steps
        sample_path[:,t] .= x
        x, _, _, _, _ = forward(prob, K, mixer, x, v, uv, ua, t)
    end
    sample_path[:,steps+1] .= x
    return sample_path
end

# also known as the backward process in IRF
function simulate_from_past_T_step(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, mixer::AbstractUnifMixer,
    x::AbstractVector{T}, v::AbstractVector{T}, uv::Union{AbstractVector{T}, Nothing}, ua::Union{T,Nothing},
    steps::Int,
) where {T<:Real}
    for t in steps:-1:1
        x, v, uv, ua, _ = forward(prob, K, mixer, x, v, uv, ua, t)
    end
    return x, v, uv, ua
end

function backward_process_trajectory(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, mixer::AbstractUnifMixer,
    x0::AbstractVector{T}, v0::AbstractVector{T}, uv0::Union{AbstractVector{T}, Nothing}, ua0::Union{T,Nothing},
    steps::Int,
) where {T<:Real}
    sample_path = []
    push!(sample_path, (x0, v0, uv0, ua0))
    @showprogress for t in 1:steps
        x, v, uv, ua = simulate_from_past_T_step(prob, K, mixer, x0, v0, uv0, ua0, t)
        push!(sample_path, (x, v, uv, ua))
    end
    return sample_path
end

function backward_process_trajectory_x(
    prob::MixFlowProblem, K::MultivariateInvolutiveKernel, mixer::AbstractUnifMixer,
    x0::AbstractVector{T}, v0::AbstractVector{T}, uv0::Union{AbstractVector{T}, Nothing}, ua0::Union{T,Nothing},
    steps::Int,
) where {T<:Real}
    sample_path = zeros(T, length(x0), steps+1)
    sample_path[:,1] .= x0
    @showprogress @threads for t in 1:steps
        x, _, _, _ = simulate_from_past_T_step(prob, K, mixer, x0, v0, uv0, ua0, t)
        sample_path[:,t+1] .= x
    end
    return sample_path
end
