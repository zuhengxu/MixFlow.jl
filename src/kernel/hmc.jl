# using Gaussian momentum 
struct HMC{T} <: MultivariateInvolutiveKernel
    n_leapfrog::Int        
    ϵ::T
    function HMC(n_leapfrog::Int, ϵ::T) where T
        if n_leapfrog < 1
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


# generating samples from HMC
function advanced_hmc_sampler(
    target_ad, x0::AbstractVector{T}, n_leapfrog::Int, stepsize::Union{Nothing, T}, n_samples::Int, n_adapts::Int, target_acc::T,
) where {T<:Real}
    dims = LogDensityProblems.dimension(target_ad)
    metric = UnitEuclideanMetric(dims)
    hamiltonian = Hamiltonian(metric, target_ad)
    if isnothing(stepsize)
        initial_ϵ = find_good_stepsize(hamiltonian, x0)
    else
        initial_ϵ = stepsize
    end
    integrator = Leapfrog(initial_ϵ)
    kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(n_leapfrog)))
    adaptor = StepSizeAdaptor(target_acc, initial_ϵ)

    samples, stats = sample(
        hamiltonian, kernel, x0, n_samples+n_adapts, adaptor, n_adapts; progress=true
    )
    return reduce(hcat, samples), stats
end


function mcmc_sampler(
    prob::MixFlowProblem, K::HMC, x0::Union{Nothing, Vector{T}}, nsamples::Int;
    init_stepsize::Bool = false, n_adapts::Int = 0, target_acc::T = 0.8,
) where T
    if isnothing(x0)
        x0 = rand(prob.reference)
    end
    target_ad = prob.target
    if init_stepsize
        initial_ϵ = nothing
    else
        initial_ϵ = K.ϵ
    end

    samples, _ =  advanced_hmc_sampler(
        target_ad, x0, K.n_leapfrog, initial_ϵ, nsamples, n_adapts, target_acc
    )
    return n_adapts == 0 ? hcat(x0, samples) : samples
end

    

