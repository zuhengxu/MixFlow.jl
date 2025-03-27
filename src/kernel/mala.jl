struct MALA{T} <: MultivariateInvolutiveKernel
    ϵ::T
    function MALA(ϵ::T) where T
        if ϵ <= 0
            throw(ArgumentError("mala stepsize ϵ must be positive"))
        end
        new{T}(ϵ)
    end
end

_euler_step(∇ℓπ, ϵ::T, x::AbstractVector{T}) where T = x .+ ϵ .* ∇ℓπ(x) ./ 2

function _dist_v_given_x(K::MALA{T}, prob::MixFlowProblem, x::AbstractVector{T}) where T
    dim = LogDensityProblems.dimension(prob.target)
    ∇ℓπ = Base.Fix1(∇logpdf_target, prob)
    stepsize = K.ϵ
    μ = _euler_step(∇ℓπ, stepsize, x)
    return MvNormal(μ, stepsize*I)
end

_rand_v_given_x(K::MALA{T}, prob::MixFlowProblem, x::AbstractVector{T}) where T = rand(_dist_v_given_x(K, prob, x))
_rand_v_given_x(K::MALA{T}, prob::MixFlowProblem, x::AbstractVector{T}, n::Int) where T = rand(_dist_v_given_x(K, prob, x), n)

function _cdf_v_given_x(
    K::MALA{T}, prob::MixFlowProblem, x::AbstractVector{T}, v::AbstractVector{T}
) where T
    dim = LogDensityProblems.dimension(prob.target)
    ∇ℓπ = Base.Fix1(∇logpdf_target, prob)
    stepsize = K.ϵ
    μ = _euler_step(∇ℓπ, stepsize, x)
    σ = sqrt(stepsize)
    return normcdf.(μ, σ*ones(dim), v)
end

function _invcdf_v_given_x(
    K::MALA{T}, prob::MixFlowProblem, x::AbstractVector{T}, uv::AbstractVector{T}
) where T
    dim = LogDensityProblems.dimension(prob.target)
    ∇ℓπ = Base.Fix1(∇logpdf_target, prob)
    stepsize = K.ϵ
    μ = _euler_step(∇ℓπ, stepsize, x)
    σ = sqrt(stepsize)
    return norminvcdf.(μ, σ*ones(dim), uv)
end


function _rand_joint_reference(prob::MixFlowProblem, K::MALA{T}) where T
    dim = LogDensityProblems.dimension(prob.target)
    x = rand(prob.reference)
    v = _rand_v_given_x(K, prob, x)
    uv = rand(dim)
    ua = rand()
    return x, v, uv, ua
end

# its a metropolis-hastings algorithm with q(v|x) ∼ N(x + ϵ/2 ∇ℓπ(x), ϵI), of which the involution map is the swap
_involution(::MALA{T}, ::MixFlowProblem, x::AbstractVector{T}, v::AbstractVector{T}) where T = (v, x)
