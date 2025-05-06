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
# Real NVP (affine coupling) layer using Bijectors.jl interface
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


function flow_sample_eval(logp, flow; nsample = 128)
    # generate new samples from flow
    try 
        ys = rand(flow, nsample)

        logws = map(x -> NormalizingFlows.elbo_single_sample(flow, logp, x), eachcol(ys))
        el = mean(logws)
        logz = MixFlow.log_normalization_constant(logws)
        es = MixFlow.ess_from_logweights(logws)/nsample
        return el, logz, es
    catch e
        println("Error in flow_sample_eval: $e")
        return NaN, NaN, NaN
    end
end

_is_nan_or_inf(x) = isnan(x) || isinf(x)
