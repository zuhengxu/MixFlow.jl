include(joinpath(@__DIR__, "../../flowlayer.jl"))

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
    try
        Xs = rand(target, nsample)
        ldrs = [
            logpdf(flow,x) - logpdf(target, x) for x in eachcol(Xs)
        ]
        drs = abs.(expm1.(ldrs)) 
        return mean(drs)/2
    catch e
        println("Error in flow_tv_est: $e")
        return NaN
    end
end

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

    # if early stop due to NaN or Inf, return NaN for all
    if _is_nan_or_inf(stats[end].loss)
        println("Training failed: loss is NaN or Inf")
        return DataFrame(
            tv = NaN,
            elbo = NaN,
            logZ = NaN,
            ess = NaN,
        )
    end

    # losses = map(x -> x.loss, stats)
    # try and if error happens, return NaN
    tv = flow_tv_est(target, flow_trained; nsample = nsample_eval)
    el, logz, es = flow_sample_eval(logp, flow_trained; nsample = nsample_eval)
    
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
#     1, "Funnel", "neural_spline_flow", 1e-3; 
#     batchsize=32, niters=1000, show_progress=true,
#     nsample_eval=512,
# )

# flowname = "neural_spline_flow"
# name = "Funnel"
# seed = 1
# lr = 1e-3
# batchsize = 32
# niters = 1000

# flow_sample_est(logp, flow)
