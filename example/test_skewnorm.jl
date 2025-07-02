using Random, Distributions
using LinearAlgebra
using LogDensityProblems, LogDensityProblemsAD
using JLD2

using MixFlow 
const MF = MixFlow

include(joinpath(@__DIR__, "julia_env/mfvi.jl"))
include(joinpath(@__DIR__, "julia_env/Model.jl"))
include(joinpath(@__DIR__, "julia_env/evaluation.jl"))
include(joinpath(@__DIR__, "julia_env/stepsize_tuning.jl"))

# learn reference
seed = 1
Random.seed!(seed)

target, dims, ad = _load_skewnormal()
target_ad = ADgradient(ad, target)

batchsize = 10
niters = 100000

reference, _ = mfvi(target; sample_per_iter = batchsize, max_iters = niters, adtype = ad)
prob = MixFlowProblem(reference, target_ad)

name = "skewnormal"

@info "save learned reference to jld"
JLD2.save(
    "$(name)_mfvi.jld2",
    "reference", reference, 
    "prob", prob,
    "niters", niters,
    "batchsize", batchsize,
    "seed", seed,
)

T_tune = 5000

T = 5000

target_rej_rate = 0.8

ϵ, neval = find_stepsize(prob, RWMH, T_tune; target_rej_rate = target_rej_rate, thresh = 0.02, T_check_stab = 5000)

# K = RWMH(ϵ, ones(dims))

# df, output = flow_evaluation(seed, name, flowtype, kernel, T, ϵ; nsample = nsample, nchains = nchains, track_cost = false)

seed = 1
flowtype = MF.BackwardIRFMixFlow
flow_length = 10000
nchains = 30
nsample = 1024
dims = LogDensityProblems.dimension(prob)
kernel_type = MF.RWMH
step_size = 0.05
leapfrog_steps = 100


flow, mixer, nchains = setup_flow(seed, flowtype, flow_length, nchains, dims)
kernel = setup_kernel(kernel_type, step_size, leapfrog_steps, dims)


ldr_sweep = flow isa MF.EnsembleIRFFlow ? MF.log_density_ratio_flow_sweep_ensemble : MF.log_density_ratio_flow_sweep

@info "TV eval using $nsample samples on Ensemble Flow length: $flow_length, nchains: $nchains"
    
# generate samples from the target
xsπ = target_sample(prob.target.ℓ, nsample)
vsπ = reduce(hcat, [MF._rand_v_given_x(kernel, prob, xsπ[:, i]) for i in 1:size(xsπ, 2)])
uvπ = rand(dims, nsample)
uaπ = rand(nsample)

nrows = flow isa MF.EnsembleIRFFlow ? nchains : flow_length + 1
lrs = zeros(nrows, nsample)

@showprogress @threads for i in 1:nsample
    x = xsπ[:, i]
    v = vsπ[:, i]
    uv = kernel isa MF.uncorrectHMC ? nothing : uvπ[:, i]
    ua = kernel isa MF.uncorrectHMC ? nothing : uaπ[i]
    lrs[:, i] .= ldr_sweep(flow, prob, kernel, mixer, x, v, uv, ua)
end

# compute real lds
lds = zeros(nsample)
@showprogress @threads for i in 1:nsample
    x = xsπ[:, i]
    lds[i] = LogDensityProblems.logdensity(prob.target, x)
end



tvs = mean(abs.(expm1.(lrs)), dims = 2) ./ 2

df = DataFrame(
    nensembles = (flow isa MF.EnsembleIRFFlow) ? [1:nchains ;] : 1,
    tv = vec(tvs),
    nparticles = nsample,
    Ts = (flow isa MF.EnsembleIRFFlow) ? flow_length : [1:flow_length+1 ;],
) 



# using Distributions, LinearAlgebra, Random

# struct MvSM{T<:AbstractVector, S<:AbstractMatrix}
#     ξ::T           # location vector
#     Ω::S           # scale matrix (covariance)
#     α::T           # skewness vector
#     d::Int         # dimension
#     mvnorm::MvNormal
# end

# function MvSM(ξ::Vector{<:Real}, Ω::Matrix{<:Real}, α::Vector{<:Real})
#     d = length(ξ)
#     @assert size(Ω,1) == d && size(Ω,2) == d
#     @assert length(α) == d
#     mvn = MvNormal(ξ, Ω)
#     return MvSM(ξ, Ω, α, d, mvn)
# end

# function Distributions.rand(rng::AbstractRNG, dist::MvSM)
#     δ = dist.α ./ sqrt.(1 .+ dist.α.^2)
#     Z = randn(rng)
#     U = randn(rng)
#     z = rand(rng, dist.mvnorm)
#     if U <= dot(δ, (z .- dist.ξ)) / sqrt(dot(δ, dist.Ω * δ))
#         return z
#     else
#         return 2 .* dist.ξ .- z
#     end
# end

# Distributions.rand(dist::MvSM) = rand(Random.GLOBAL_RNG, dist)

# function Distributions.logpdf(dist::MvSM, x::Vector{<:Real})
#     ξ, Ω, α = dist.ξ, dist.Ω, dist.α
#     z = x .- ξ
#     invΩ = inv(Ω)
#     mahal = dot(z, invΩ * z)
#     log_phi = -0.5 * (dist.d * log(2π) + logdet(Ω) + mahal)
#     arg = dot(α, z)
#     return log(2) + log_phi + logcdf(Normal(0, 1), arg)
# end


# ξ = [0.0, 0.0]
# Ω = [1.0 0.5; 0.5 1.0]
# α = [5.0, -2.0]

# dist = MvSM(ξ, Ω, α)

# x = rand(dist)
# logp = logpdf(dist, x)

# dims = 2
