using Random, Distributions
using LinearAlgebra
using LogDensityProblems, LogDensityProblemsAD
using JLD2

using MixFlow 
const MF = MixFlow


include(joinpath(@__DIR__, "../../julia_env/Model.jl"))
include(joinpath(@__DIR__, "../../julia_env/flowlayer.jl"))

function run_baseline(
    seed, name::String, lr; 
    batchsize::Int = 64, niters::Int= 50_000, show_progress=true,
    nsample_eval::Int=1024, save_jld::Bool=true,
)
    Random.seed!(seed)

    @info "load model $(name)"
    target, dims, ad = load_model(name)

    @info "learning mfvi for $(name), dims = $(dims)"
    logp = Base.Fix1(LogDensityProblems.logdensity, target)

    q₀ = MvNormal(zeros(dims), I)
    flow =
        Bijectors.transformed(q₀, Bijectors.Shift(zeros(dims)) ∘ Bijectors.Scale(ones(dims)))
    
    cb(iter, opt_stats, re, θ) = (sample_per_iter = batchsize, ad = ad)
    checkconv(iter, stat, re, θ, st) = _is_nan_or_inf(stat.loss) || (stat.gradient_norm < 1e-3)

    time = @elapsed begin
    flow_trained, stats, _ = train_flow(
        NormalizingFlows.elbo,
        flow,
        logp,
        batchsize;
        max_iters=niters,
        optimiser=Optimisers.Adam(lr),
        ADbackend=ad,
        show_progress=show_progress,
        hasconverged=checkconv,
        callback=cb,
    )
    end
    @info "Training finished"

    # if early stop due to NaN or Inf, return NaN for all
    if _is_nan_or_inf(stats[end].loss)
        println("Training failed: loss is NaN or Inf")
        return DataFrame(
            time = NaN,
            elbo = NaN,
            logZ = NaN,
            ess = NaN,
        )
    end

    # losses = map(x -> x.loss, stats)
    # try and if error happens, return NaN
    el, logz, es = flow_sample_eval(logp, flow_trained; nsample = nsample_eval)
    
    # save the trained flow
    if save_jld
        res_dir = joinpath(@__DIR__, "result/")

        if !isdir(res_dir)
            mkdir(res_dir)
        end

        JLD2.save(
            joinpath(res_dir, "$(name)_mfvi_$(lr)_$(seed).jld2"),
            "flow", flow_trained,
            "batchsize", batchsize,
            "seed", seed,
        )
    end
    
    df = DataFrame(
        time = time,
        elbo=el,
        logZ=logz,
        ess=es,
    )

    return df
end
