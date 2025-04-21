# using Gaussian momentum 
struct uncorrectHMC{T} <: MultivariateInvolutiveKernel
    n_leapfrog::Int        
    ϵ::T
    function uncorrectHMC(n_leapfrog::Int, ϵ::T) where T
        if n_leapfrog <= 1
            throw(ArgumentError("hmc n_leapfrog must be ≥ 1"))
        end
        if ϵ <= 0
            throw(ArgumentError("hmc stepsize ϵ must be positive"))
        end
        new{T}(n_leapfrog, ϵ)
    end
end

_dist_v_given_x(::uncorrectHMC, prob::MixFlowProblem, x) = MvNormal(zeros(length(x)), I)
_rand_v_given_x(::uncorrectHMC, ::MixFlowProblem, x) = randn(length(x))
_rand_v_given_x(::uncorrectHMC, ::MixFlowProblem, x, n::Int) = randn(length(x), n)

function _rand_joint_reference(prob::MixFlowProblem, K::uncorrectHMC)
    x = rand(prob.reference)
    v = _rand_v_given_x(K, prob, x)
    return x, v, nothing, nothing
end

function _leapfrog(∇ℓπ, ϵ::Real, L::Int, x, v)
    v += ϵ/2 .* ∇ℓπ(x) 
    for _ in 1:L - 1
        x += ϵ .* v
        v += ϵ .* ∇ℓπ(x)
    end
    x += ϵ .* v
    v += ϵ/2 .* ∇ℓπ(x)
    return x, v
end

leapfrog(∇ℓπ, K::uncorrectHMC, x, v) = _leapfrog(∇ℓπ, K.ϵ, K.n_leapfrog, x, v)
inv_leapfrog(∇ℓπ, K::uncorrectHMC, x, v) = _leapfrog(∇ℓπ, -K.ϵ, K.n_leapfrog, x, v)

function _refresh_momentum(::uncorrectHMC, S::ErgodicShift, v, t)
    uv = normcdf.(v)
    uv_ = _ergodic_shift.(uv, S.ξs_uv[:, t])
    v_ = norminvcdf.(uv_)

    dist = MvNormal(zeros(length(v)), ones(length(v)))
    logjac = logpdf(dist, v) - logpdf(dist, v_)
    return v_, logjac
end

function _invref_momentum(::uncorrectHMC, S::ErgodicShift, v, t)
    uv = normcdf.(v)
    uv_ = _inv_ergodic_shift.(uv, S.ξs_uv[:, t])
    v_ = norminvcdf.(uv_)

    dist = MvNormal(zeros(length(v)), ones(length(v)))
    logjac = logpdf(dist, v) - logpdf(dist, v_)
    return v_, logjac
end

function forward_with_logdetjac(
    prob::MixFlowProblem, K::uncorrectHMC{T}, unif_mixer::ErgodicShift, 
    x::AbstractVector{T}, v::AbstractVector{T}, ::Nothing, ::Nothing,
    t::Int,
) where {T<:Real}
    ∇ℓπ = Base.Fix1(∇logpdf_target, prob)

    # leapfrog
    x_, v_ = leapfrog(∇ℓπ, K, x, v)
    # momentum refreshment
    v_, logJ = _refresh_momentum(K, unif_mixer, v_, t)
    
    return x_, v_, nothing, nothing, true, logJ
end
forward(
    prob::MixFlowProblem, K::uncorrectHMC{T}, S::ErgodicShift, 
    x::AbstractVector{T}, v::AbstractVector{T}, ::Nothing, ::Nothing,
    t::Int,
) where T = forward_with_logdetjac(prob, K, S, x, v, nothing, nothing, t)[1:5]

function inverse_with_logdetjac(
    prob::MixFlowProblem, K::uncorrectHMC{T}, unif_mixer::ErgodicShift, 
    x_::AbstractVector{T}, v_::AbstractVector{T}, ::Nothing, ::Nothing,
    t::Int,
) where {T<:Real}
    ∇ℓπ = Base.Fix1(∇logpdf_target, prob)

    # momentum refreshment
    v, logJ = _invref_momentum(K, unif_mixer, v_, t)
    # leapfrog
    x, v = inv_leapfrog(∇ℓπ, K, x_, v)

    return x, v, nothing, nothing, true, logJ
end
inverse(
    prob::MixFlowProblem, K::uncorrectHMC{T}, S::ErgodicShift, 
    x_::AbstractVector{T}, v_::AbstractVector{T}, ::Nothing, ::Nothing,
    t::Int,
) where T = inverse_with_logdetjac(prob, K, S, x_, v_, nothing, nothing, t)[1:5]
