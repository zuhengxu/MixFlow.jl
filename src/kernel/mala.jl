using PDMats: PDiagMat

struct MALA{T} <: MultivariateInvolutiveKernel
    ϵ::T
    σs::PDiagMat{T} # diagonal stds of the proposal
    function MALA(ϵ::T, σs::PDiagMat{T}) where {T}
        if ϵ <= 0
            throw(ArgumentError("mala stepsize ϵ must be positive"))
        end
        new{T}(ϵ, σs)
    end
end

MALA(ϵ::T, σs::AbstractVector{T}) where {T} = MALA(ϵ, PDiagMat(σs))
_euler_step(∇ℓπ, ϵ::T, x::AbstractVector{T}, σs::PDiagMat{T}) where {T} = x .+ ϵ .* σs.diag.^2 .* ∇ℓπ(x) ./ 2

function _dist_v_given_x(K::MALA{T}, prob::MixFlowProblem, x::AbstractVector{T}) where T
    # dim = LogDensityProblems.dimension(prob.target)
    ∇ℓπ = Base.Fix1(∇logpdf_target, prob)
    stepsize = K.ϵ
    μ = _euler_step(∇ℓπ, stepsize, x, K.σs)
    return MvNormal(μ, stepsize.*K.σs.^2)
end

_rand_v_given_x(K::MALA{T}, prob::MixFlowProblem, x::AbstractVector{T}) where T = rand(_dist_v_given_x(K, prob, x))
_rand_v_given_x(K::MALA{T}, prob::MixFlowProblem, x::AbstractVector{T}, n::Int) where T = rand(_dist_v_given_x(K, prob, x), n)

function _cdf_v_given_x(
    K::MALA{T}, prob::MixFlowProblem, x::AbstractVector{T}, v::AbstractVector{T}
) where T
    # dim = LogDensityProblems.dimension(prob.target)
    ∇ℓπ = Base.Fix1(∇logpdf_target, prob)
    stepsize = K.ϵ
    μ = _euler_step(∇ℓπ, stepsize, x, K.σs)
    ϵ_sqrt = sqrt(stepsize)
    return normcdf.(μ, ϵ_sqrt.*K.σs.diag, v)
end

function _invcdf_v_given_x(
    K::MALA{T}, prob::MixFlowProblem, x::AbstractVector{T}, uv::AbstractVector{T}
) where T
    # dim = LogDensityProblems.dimension(prob.target)
    ∇ℓπ = Base.Fix1(∇logpdf_target, prob)
    stepsize = K.ϵ
    μ = _euler_step(∇ℓπ, stepsize, x, K.σs)
    ϵ_sqrt = sqrt(stepsize)
    return norminvcdf.(μ, ϵ_sqrt.*K.σs.diag, uv)
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

# mcmc sampler
function mcmc_step(prob::MixFlowProblem, K::MALA{T}, x::AbstractVector{T}) where T
    #proposal
    v = _rand_v_given_x(K, prob, x)
    ℓ_proposal = logdensity_target(prob, v) + logpdf(_dist_v_given_x(K, prob, v), x)
    ℓ_previous = logdensity_target(prob, x) + logpdf(_dist_v_given_x(K, prob, x), v)
    logr = ℓ_proposal - ℓ_previous
    return check_acc(rand(), logr) ? v : x
end
