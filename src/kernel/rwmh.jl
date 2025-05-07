struct RWMH{T} <: MultivariateInvolutiveKernel 
    ϵ::T # stepsize
    invdiagM::AbstractVector{T} # diagonal preconditioner
end

_dimension(K::RWMH) = length(K.invdiagM)
_dist_v_given_x(K::RWMH{T}, ::MixFlowProblem, x::AbstractVector{T}) where T = MvNormal(zeros(_dimension(K)), I)
_rand_v_given_x(K::RWMH{T}, ::MixFlowProblem, x::AbstractVector{T}) where T = randn(_dimension(K)) 

_cdf_v_given_x(
    K::RWMH{T}, ::MixFlowProblem, x::AbstractVector{T}, v::AbstractVector{T}
) where T = normcdf.(v)

_invcdf_v_given_x(
    K::RWMH{T}, ::MixFlowProblem, x::AbstractVector{T}, uv::AbstractVector{T}
) where T = norminvcdf.(uv)

_involution(
    K::RWMH{T}, ::MixFlowProblem, x::AbstractVector{T}, v::AbstractVector{T}
) where T = (x .+ K.ϵ .* v .* K.invdiagM, -v)


function _rand_joint_reference(prob::MixFlowProblem, K::RWMH)
    dims = LogDensityProblems.dimension(prob.target)
    x = rand(prob.reference)
    v = _rand_v_given_x(K, prob, x)
    uv = rand(dims)
    ua = rand()
    return x, v, uv, ua
end


function mcmc_step(prob::MixFlowProblem, K::RWMH{T}, x::AbstractVector{T}) where T
    #proposal
    v = _rand_v_given_x(K, prob, x)
    logr = logdensity_target(prob, v) - logdensity_target(prob, x)
    return check_acc(rand(), logr) ? v : x
end
