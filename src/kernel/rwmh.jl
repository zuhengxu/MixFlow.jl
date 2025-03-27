using PDMats: PDiagMat

struct RWMH{T} <: MultivariateInvolutiveKernel 
    σs::PDiagMat{T} # diagonal stds of the proposal
end

RWMH(σs::AbstractVector{T}) where {T} = RWMH(PDiagMat(σs))
RWMH(dim::Int) = RWMH(PDiagMat(ones(dim)))

_involution(::RWMH{T}, ::MixFlowProblem, x::AbstractVector{T}, v::AbstractVector{T}) where T = (v, x)
_dist_v_given_x(K::RWMH{T}, ::MixFlowProblem, x::AbstractVector{T}) where T = MvNormal(x, K.σs.^2)
_rand_v_given_x(K::RWMH{T}, ::MixFlowProblem, x::AbstractVector{T}) where T = randn(length(x)) .* K.σs.diag .+ x
_rand_v_given_x(K::RWMH{T}, ::MixFlowProblem, x::AbstractVector{T}, n::Int) where T = randn(length(x), n) .* K.σs.diag .+ x

_cdf_v_given_x(
    K::RWMH{T}, ::MixFlowProblem, x::AbstractVector{T}, v::AbstractVector{T}
) where T = normcdf.(x, K.σs.diag, v)

_invcdf_v_given_x(
    K::RWMH{T}, ::MixFlowProblem, x::AbstractVector{T}, uv::AbstractVector{T}
) where T = norminvcdf.(x, K.σs.diag, uv)

# normcdf(μ, σ, x)
# norminvcdf(μ, σ, x)

function _rand_joint_reference(prob::MixFlowProblem, K::RWMH)
    dim = LogDensityProblems.dimension(prob.target)
    x = rand(prob.reference)
    v = _rand_v_given_x(K, prob, x)
    uv = rand(dim)
    ua = rand()
    return x, v, uv, ua
end
