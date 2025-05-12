using JLD2
using DataFrames, CSV
using Optimisers
using Random

include(joinpath(@__DIR__, "../../julia_env/Model.jl"))
include(joinpath(@__DIR__, "../../julia_env/flowlayer.jl"))

function create_neural_spline_flow(name, nlayers)
    # reference
    q0 = JLD2.load(joinpath(@__DIR__, "../reference/result/$(name)_mfvi.jld2"))["reference"]

    dims = length(q0)
    hdims = min(dims, 64)
    mask_idx1 = 1:2:dims
    mask_idx2 = 2:2:dims

    K = 10
    B = 30
    Ls = [ NeuralSplineLayer(dims, hdims, K, B, mask_idx1) ∘ NeuralSplineLayer(dims, hdims, K, B, mask_idx2) for _ in 1:nlayers ]

    flow = create_flow(Ls, q0)
    return flow 
end

function create_real_nvp(name, nlayers)
    # reference
    q0 = JLD2.load(joinpath(@__DIR__, "../reference/result/$(name)_mfvi.jld2"))["reference"]

    dims = length(q0)
    hdims = min(dims, 64)
    mask_idx1 = 1:2:dims
    mask_idx2 = 2:2:dims

    Ls = [ AffineCoupling(dims, hdims, mask_idx1) ∘ AffineCoupling(dims, hdims, mask_idx2) for _ in 1:nlayers ]

    flow = create_flow(Ls, q0)
    return flow 
end

# nf training
function run_norm_flow(
    seed, name::String, flowname::String, nlayers, lr; 
    batchsize::Int = 32, niters::Int=100_000, show_progress=true,
    nsample_eval::Int=128, save_jld::Bool=true,
)
    Random.seed!(seed)
    target, _, _ = load_model(name)

    # mooncake is much faster for large nn
    ad = AutoMooncake(; config = Mooncake.Config())

    logp = Base.Fix1(LogDensityProblems.logdensity, target)

    # create flow
    if flowname == "neural_spline_flow"
        flow = create_neural_spline_flow(name, nlayers)
    elseif flowname == "real_nvp"
        flow = create_real_nvp(name, nlayers)
    else
        error("flow not defined")
    end
    
    # #############
    # flow train
    # #############
    # stop if nan or inf in training
    checkconv(iter, stat, re, θ, st) = _is_nan_or_inf(stat.loss) || (stat.gradient_norm < 1e-3)

    time_train = @elapsed begin
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
            joinpath(res_dir, "$(name)_$(flowname)_$(nlayers)_$(lr)_$(seed).jld2"),
            "flow", flow_trained,
            "batchsize", batchsize,
            "seed", seed,
        )
    end
    
    df = DataFrame(
        time = time_train,
        elbo=el,
        logZ=logz,
        ess=es,
    )

    return df
end



# target_list = ["TReg", "SparseRegression", "Brownian", "Sonar"]

# for name in target_list
#     @info "Running $name"
#     df = run_norm_flow(
#         1, name, "neural_spline_flow", 3, 1e-3; 
#         batchsize=32, niters=10, show_progress=true,
#         nsample_eval=128,
#     )
# end

# name = "LGCP"
# df = run_norm_flow(
#     1, name, "real_nvp", 5, 1e-4; 
#     batchsize=32, niters=100, show_progress=true,
#     nsample_eval=128,
# )
