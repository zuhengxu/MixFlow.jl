using Optimisers
using Mooncake
using Bijectors
using Functors
using ADTypes
using LogDensityProblems
using NormalizingFlows

# prevent destrcuture the param
@leaf MvNormal

function mfvi(
    target;
    adtype = AutoMooncake(; config = Mooncake.Config()),
    sample_per_iter = 10,
    max_iters = 10000,
    optimiser = Optimisers.Adam(1e-3),
)
    dim = LogDensityProblems.dimension(target)
    logp = Base.Fix1(LogDensityProblems.logdensity, target)

    q₀ = MvNormal(zeros(dim), I)
    flow =
        Bijectors.transformed(q₀, Bijectors.Shift(zeros(dim)) ∘ Bijectors.Scale(ones(dim)))

    cb(iter, opt_stats, re, θ) = (sample_per_iter = sample_per_iter, ad = adtype)
    checkconv(iter, stat, re, θ, st) = stat.gradient_norm < 1e-3
    flow_trained, stats, _ = NormalizingFlows.train_flow(
        NormalizingFlows.elbo,
        flow,
        logp,
        sample_per_iter;
        max_iters = max_iters,
        optimiser = optimiser,
        ADbackend = adtype,
        show_progress = true,
        callback = cb,
        hasconverged = checkconv,
    )
    return flow_trained, stats
end

# target = load_model("Banana")
# flow, stats = mfvi(target; sample_per_iter = 10, max_iters = 10000)

# function visualize(p::Bijectors.MultivariateTransformed, samples=rand(p, 1000))
#     xrange = range(minimum(samples[1, :]) - 1, maximum(samples[1, :]) + 1; length=100)
#     yrange = range(minimum(samples[2, :]) - 1, maximum(samples[2, :]) + 1; length=100)
#     z = [exp(Distributions.logpdf(p, [x, y])) for x in xrange, y in yrange]
#     fig = contour(xrange, yrange, z'; levels=15, color=:viridis, label="PDF", linewidth=2)
#     scatter!(samples[1, :], samples[2, :]; label="Samples", alpha=0.3, legend=:bottomright)
#     return fig
# end
