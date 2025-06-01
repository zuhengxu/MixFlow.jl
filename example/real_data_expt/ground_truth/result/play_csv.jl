include("../run_ais.jl")

include(joinpath(@__DIR__, "../../../julia_env/", "utils.jl"))
include(joinpath(@__DIR__, "../../../julia_env/", "plotting.jl"))
using Pipe
using JLD2

function process_jld(pth)
    res = JLD2.load(pth)
    out = res["output"].particles
    logZ = out.log_normalization

    Mean = mean(out)
    Std = std(out)

    return logZ, Mean, Std
end

# load all the results
res_dir = joinpath(@__DIR__, "../ground_truth_res/")
if !isdir(res_dir)
    mkpath(res_dir)
end

for (t, alg) in Iterators.product(real_data_list, ["sais"])
    local logZs = []
    local Means = []
    local Stds = []

    for seed in 1:10
        local pth = "$(t)_$(alg)_$(seed).jld2"
        println("Loading $pth")
        
        local logZ, Mean, Std = process_jld(joinpath(@__DIR__, "../result/", pth))
        push!(logZs, logZ)
        push!(Means, Mean)
        push!(Stds, Std)
    end
    local logZ_med = median(logZs)

    local Means_med = @pipe Means |> 
        reduce(hcat, _) |> 
        median(_; dims=2) |>
        vec(_)

    local Stds_med = @pipe Stds |>
        reduce(hcat, _) |> 
        median(_; dims=2) |>
        vec(_)

    JLD2.save(
        joinpath(res_dir, "$(t)_truth.jld2"),
        "logZ", logZ_med,
        "Mean", Means_med,
        "Std", Stds_med,
    )
end

# t = "LGCP"
# alg = "sais"
# seed = 1

# pth = "$(t)_$(alg)_$(seed).jld2"
# println("Loading $pth")

# res = JLD2.load(pth)

# out = res["output"].particles
# logZ = out.log_normalization

# Mean = mean(out)
# Std = std(out)

