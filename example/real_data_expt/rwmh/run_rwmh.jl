using ProgressMeter
using Random, Distributions, Plots
using LinearAlgebra
using Base.Threads: @threads
using LogExpFunctions
using LogDensityProblems, LogDensityProblemsAD
using ADTypes, Mooncake
using DataFrames, CSV
using JLD2

using MixFlow 
const MF = MixFlow

include(joinpath(@__DIR__, "../../julia_env/evaluation.jl"))


function run_simulation(
    seed, name, flowtype, kernel, T, nchains; 
    nsample = 128, target_rej_rate = 0.2, track_cost = true, T_tune = T, 
    save_jld = false
)
    Random.seed!(seed)
    prob, dims = load_prob_with_ref(name)

    if track_cost
        prob = MixFlowProblem(prob.reference, TrackedLogDensityProblem(prob.target))
    end

    ϵ, neval = find_stepsize(prob, kernel, T_tune; target_rej_rate = target_rej_rate, thresh = 0.02, T_check_stab = T)

    # compute cost for tuning stepsize
    cost_tuning = track_cost ? compute_cost(prob.target) : NaN


    K = kernel(ϵ, ones(dims))

    # rej_rate, err = rejection_rate(prob, K, T_check)

    df, output = flow_evaluation(seed, name, flowtype, kernel, T, ϵ; nsample = nsample, nchains = nchains, track_cost = track_cost)
    
    # add cost tuning
    df[!, "cost_tuning"] .= cost_tuning

    jld_pth = joinpath(@__DIR__, "result/")
    if save_jld 
        if !isdir(jld_pth)
            mkpath(jld_pth)
        end
        JLD2.save(
            joinpath(jld_pth, "rwmh_$(name)_$seed.jld2"),
            "output", output,
        )
    end
    return df, output
end


# name = "TReg"
# prob, dims = load_prob_with_ref(name)
# kernel = RWMH
# T_check = 5000
# ϵ, neval = find_stepsize(prob, kernel, T_check; target_rej_rate = 0.2, thresh = 0.02, T_check_stab = T_check)
# K = kernel(ϵ, ones(dims))


# rej_rate, err = rejection_rate(prob, K, T_check)

# flowtype = MF.BackwardIRFMixFlow
# # flowtype = MF.IRFMixFlow
# # flowtype = MF.DeterministicMixFlow
# # flowtype = MF.EnsembleIRFFlow

# df = flow_evaluation(1, name, flowtype, kernel, T_check, ϵ; nsample = 1024, nchains = 30)

# df, output = run_simulation(
#     1, "Banana", MF.EnsembleIRFFlow, RWMH, 5000, 30; target_rej_rate = 0.766, nsample = 128
# )
