# using Gaussian momentum 
struct HMC{T} <: MultivariateInvolutiveKernel
    n_leapfrog::Int        
    ϵ::T
    function HMC(n_leapfrog::Int, ϵ::T) where T
        if n_leapfrog <= 1
            throw(ArgumentError("hmc n_leapfrog must be ≥ 1"))
        end
        if ϵ <= 0
            throw(ArgumentError("hmc stepsize ϵ must be positive"))
        end
        new{T}(n_leapfrog, ϵ)
    end
end

function _dist_v_given_x(::HMC{T}, ::MixFlowProblem, x::AbstractVector{T}) where T
    dim = length(x)
    return MvNormal(zeros(T, dim), I)
end
_rand_v_given_x(::HMC{T}, ::MixFlowProblem, x::AbstractVector{T}) where T = randn(length(x))
_rand_v_given_x(::HMC{T}, ::MixFlowProblem, x::AbstractVector{T}, n::Int) where T = randn(length(x), n)
_cdf_v_given_x(::HMC{T}, ::MixFlowProblem, ::AbstractVector{T}, v::AbstractVector{T}) where T = normcdf.(v)
_invcdf_v_given_x(::HMC{T}, ::MixFlowProblem, ::AbstractVector{T}, uv::AbstractVector{T}) where T = norminvcdf.(uv)

function _rand_joint_reference(prob::MixFlowProblem, K::HMC)
    dim = LogDensityProblems.dimension(prob.target)
    x = rand(prob.reference)
    v = _rand_v_given_x(K, prob, x)
    uv = rand(dim)
    ua = rand()
    return x, v, uv, ua
end

function _involution(K::HMC{T}, prob::MixFlowProblem, x::AbstractVector{T}, v::AbstractVector{T}) where T
    ∇ℓπ = Base.Fix1(∇logpdf_target, prob)
    ϵ = K.ϵ
    x_, v_ = _leapfrog(∇ℓπ, ϵ, K.n_leapfrog, x, v)
    # flip momentum to ensure involutivity
    return x_, -v_
end

