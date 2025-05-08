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

include(joinpath(@__DIR__, "../../evaluation.jl"))


function run_simulation(
    seed, name, flowtype, kernel, T, nchains; 
    nsample = 128, target_rej_rate = 0.2, track_cost = true, T_tune = T, 
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

    df = flow_evaluation(seed, name, flowtype, kernel, T, ϵ; nsample = nsample, nchains = nchains)
    
    # add cost tuning
    df[!, "cost_tuning"] .= cost_tuning
    return df
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

# df = run_simulation(
#     1, "SparseRegression", MF.BackwardIRFMixFlow, RWMH, 5000, 30; nsample = 128
# )
