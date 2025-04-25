using Flux

using Bijectors
using Bijectors: partition, PartitionMask

using Random, Distributions, LinearAlgebra
using Functors
using Optimisers, ADTypes
using Mooncake
using NormalizingFlows
using LogExpFunctions
using DataFrames, CSV
using JLD2

using MixFlow


include(joinpath( @__DIR__, "Model.jl"))

@leaf MvNormal # avoid optimizing the init params of MvNormal, which causes issuse

##################################
# define neural spline layer using Bijectors.jl interface
#################################
"""
Neural Rational quadratic Spline layer 

# References
[1] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G., Neural Spline Flows, CoRR, arXiv:1906.04032 [stat.ML],  (2019). 
"""
struct NeuralSplineLayer{T,A<:Flux.Chain} <: Bijectors.Bijector
    dim::Int                # dimension of input
    K::Int                  # number of knots
    n_dims_transferred::Int  # number of dimensions that are transformed
    nn::A   # networks that parmaterize the knots and derivatives
    B::T                    # bound of the knots
    mask::Bijectors.PartitionMask
end

function NeuralSplineLayer(
    dim::T1,                # dimension of input
    hdims::T1,              # dimension of hidden units for s and t
    K::T1,                  # number of knots
    B::T2,                  # bound of the knots
    mask_idx::AbstractVector{<:Int}, # index of dimensione that one wants to apply transformations on
) where {T1<:Int,T2<:Real}
    num_of_transformed_dims = length(mask_idx)
    input_dims = dim - num_of_transformed_dims
    
    # output dim of the NN
    output_dims = (3K - 1)*num_of_transformed_dims
    # one big mlp that outputs all the knots and derivatives for all the transformed dimensions
    nn = mlp3(input_dims, hdims, output_dims)

    mask = Bijectors.PartitionMask(dim, mask_idx)
    return NeuralSplineLayer(dim, K, num_of_transformed_dims, nn, B, mask)
end

@functor NeuralSplineLayer (nn,)

# define forward and inverse transformation
"""
Build a rational quadratic spline from the nn output
Bijectors.jl has implemented the inverse and logabsdetjac for rational quadratic spline

we just need to map the nn output to the knots and derivatives of the RQS
"""
function instantiate_rqs(nsl::NeuralSplineLayer, x::AbstractVector)
    K, B = nsl.K, nsl.B
    nnoutput = reshape(nsl.nn(x), nsl.n_dims_transferred, :)
    ws = @view nnoutput[:, 1:K]
    hs = @view nnoutput[:, (K + 1):(2K)]
    ds = @view nnoutput[:, (2K + 1):(3K - 1)]
    return Bijectors.RationalQuadraticSpline(ws, hs, ds, B)
end

function Bijectors.transform(nsl::NeuralSplineLayer, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(nsl.mask, x)
    # instantiate rqs knots and derivatives
    rqs = instantiate_rqs(nsl, x_2)
    y_1 = Bijectors.transform(rqs, x_1)
    return Bijectors.combine(nsl.mask, y_1, x_2, x_3)
end

function Bijectors.transform(insl::Inverse{<:NeuralSplineLayer}, y::AbstractVector)
    nsl = insl.orig
    y1, y2, y3 = partition(nsl.mask, y)
    rqs = instantiate_rqs(nsl, y2)
    x1 = Bijectors.transform(Inverse(rqs), y1)
    return Bijectors.combine(nsl.mask, x1, y2, y3)
end

function (nsl::NeuralSplineLayer)(x::AbstractVector)
    return Bijectors.transform(nsl, x)
end

# define logabsdetjac
function Bijectors.logabsdetjac(nsl::NeuralSplineLayer, x::AbstractVector)
    x_1, x_2, _ = Bijectors.partition(nsl.mask, x)
    rqs = instantiate_rqs(nsl, x_2)
    logjac = logabsdetjac(rqs, x_1)
    return logjac
end

function Bijectors.logabsdetjac(insl::Inverse{<:NeuralSplineLayer}, y::AbstractVector)
    nsl = insl.orig
    y1, y2, _ = partition(nsl.mask, y)
    rqs = instantiate_rqs(nsl, y2)
    logjac = logabsdetjac(Inverse(rqs), y1)
    return logjac
end

function Bijectors.with_logabsdet_jacobian(nsl::NeuralSplineLayer, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(nsl.mask, x)
    rqs = instantiate_rqs(nsl, x_2)
    y_1, logjac = with_logabsdet_jacobian(rqs, x_1)
    return Bijectors.combine(nsl.mask, y_1, x_2, x_3), logjac
end


##################################
# define affine coupling layer using Bijectors.jl interface
#################################
struct AffineCoupling <: Bijectors.Bijector
    dim::Int
    mask::Bijectors.PartitionMask
    s::Flux.Chain
    t::Flux.Chain
end

# let params track field s and t
@functor AffineCoupling (s, t)

function AffineCoupling(
    dim::Int,  # dimension of input
    hdims::Int, # dimension of hidden units for s and t
    mask_idx::AbstractVector, # index of dimensione that one wants to apply transformations on
)
    cdims = length(mask_idx) # dimension of parts used to construct coupling law
    s = mlp3(cdims, hdims, cdims)
    t = mlp3(cdims, hdims, cdims)
    mask = PartitionMask(dim, mask_idx)
    return AffineCoupling(dim, mask, s, t)
end

function Bijectors.transform(af::AffineCoupling, x::AbstractVector)
    # partition vector using 'af.mask::PartitionMask`
    x₁, x₂, x₃ = partition(af.mask, x)
    y₁ = x₁ .* af.s(x₂) .+ af.t(x₂)
    return Bijectors.combine(af.mask, y₁, x₂, x₃)
end

function (af::AffineCoupling)(x::AbstractArray)
    return Bijectors.transform(af, x)
end

function Bijectors.with_logabsdet_jacobian(af::AffineCoupling, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(af.mask, x)
    y_1 = af.s(x_2) .* x_1 .+ af.t(x_2)
    logjac = sum(log ∘ abs, af.s(x_2))
    return Bijectors.combine(af.mask, y_1, x_2, x_3), logjac
end

function Bijectors.with_logabsdet_jacobian(
    iaf::Inverse{<:AffineCoupling}, y::AbstractVector
)
    af = iaf.orig
    # partition vector using `af.mask::PartitionMask`
    y_1, y_2, y_3 = partition(af.mask, y)
    # inverse transformation
    x_1 = (y_1 .- af.t(y_2)) ./ af.s(y_2)
    logjac = -sum(log ∘ abs, af.s(y_2))
    return Bijectors.combine(af.mask, x_1, y_2, y_3), logjac
end

function Bijectors.logabsdetjac(af::AffineCoupling, x::AbstractVector)
    _, x_2, _ = partition(af.mask, x)
    logjac = sum(log ∘ abs, af.s(x_2))
    return logjac
end

#########################
# some utility functions for creating flows
########################
function mlp3(input_dim::Int, hidden_dims::Int, output_dim::Int; activation=Flux.leakyrelu)
    m = Chain(
        Flux.Dense(input_dim, hidden_dims, activation),
        Flux.Dense(hidden_dims, hidden_dims, activation),
        Flux.Dense(hidden_dims, output_dim),
    )
    return Flux.f64(m) # all uses Float64
end

function create_flow(Ls, q₀)
    ts =  reduce(∘, Ls)
    return Bijectors.transformed(q₀, ts)
end

function create_neural_spline_flow()
    # reference
    q0 = MvNormal(zeros(2), ones(2))

    dims = 2
    hdims = 64
    K = 10
    B = 30
    Ls = [
        NeuralSplineLayer(dims, hdims, K, B, [1]) ∘ NeuralSplineLayer(dims, hdims, K, B, [2]) for
        _ in 1:3
    ]

    flow = create_flow(Ls, q0)
    return flow 
end

function create_real_nvp()
    # reference
    q0 = MvNormal(zeros(2), ones(2))

    dims = 2
    hdims = 32
    Ls = [AffineCoupling(dims, hdims, [1]) ∘ AffineCoupling(dims, hdims, [2]) for _ in 1:3]

    flow = create_flow(Ls, q0)
    return flow 
end

function flow_tv_est(target, flow; nsample = 128)
    Xs = rand(target, nsample)
    ldrs = [
        logpdf(flow, x) - logpdf(target, x) for x in eachcol(Xs)
    ]
    drs = abs.(expm1.(ldrs)) 
    return mean(drs)/2
end

flow_elbo_est(logp, flow; nsample = 128) = NormalizingFlows.elbo(flow, logp, nsample)

_is_nan_or_inf(x) = isnan(x) || isinf(x)

# running function on trained flow
function run_norm_flow(
    seed, name::String, flowname::String, lr; 
    batchsize::Int = 32, niters::Int=100_000, show_progress=true,
    nsample_eval::Int=128,
)
    Random.seed!(seed)
    target = load_model(name)
    logp = Base.Fix1(logpdf, target)

    # create flow
    if flowname == "neural_spline_flow"
        flow = create_neural_spline_flow()
    elseif flowname == "real_nvp"
        flow = create_real_nvp()
    else
        error("flow not defined")
    end
    
    # #############
    # flow train
    # #############
    
    adtype = ADTypes.AutoMooncake(; config = Mooncake.Config())
    checkconv(iter, stat, re, θ, st) = _is_nan_or_inf(stat.loss) || (stat.gradient_norm < 1e-3)

    flow_trained, stats, _ = train_flow(
        elbo,
        flow,
        logp,
        batchsize;
        max_iters=niters,
        optimiser=Optimisers.Adam(lr),
        ADbackend=adtype,
        show_progress=show_progress,
        hasconverged=checkconv,
    )
    @info "Training finished"

    # generate new samples
    ys = rand(flow_trained, nsample_eval)

    # losses = map(x -> x.loss, stats)
    tv = flow_tv_est(target, flow_trained; nsample = nsample_eval)

    logws = map(x -> NormalizingFlows.elbo_single_sample(flow_trained, logp, x), eachcol(ys))
    el = mean(logws)
    logz = MixFlow.log_normalization_constant(logws)
    es = MixFlow.ess_from_logweights(logws)/nsample_eval
    
    # # save the trained flow
    # res_dir = joinpath(@__DIR__, "result/")
    # JLD2.save(
    #     joinpath(res_dir, "$(name)_$(flowname)_$(lr)_$(seed).jld2"),
    #     "flow", flow_trained,
    #     "losses", losses,
    #     "batchsize", batchsize,
    #     "seed", seed,
    # )
    
    return DataFrame(
        tv=tv,
        elbo=el,
        logZ=logz,
        ess=es,
    )
end

# df = run_norm_flow(
#     1, "Funnel", "real_nvp", 1e-3; 
#     batchsize=64, niters=10000, show_progress=true,
#     nsample_eval=512,
# )

# flowname = "neural_spline_flow"
# name = "Banana"
# seed = 1
# lr = 1e-3
# batchsize = 32
# niters = 500

# flow_elbo_est(logp, flow)
