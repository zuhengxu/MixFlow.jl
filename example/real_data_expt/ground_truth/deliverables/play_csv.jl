include("../run_ais.jl")

include(joinpath(@__DIR__, "../../../julia_env/", "utils.jl"))
include(joinpath(@__DIR__, "../../../julia_env/", "plotting.jl"))
using Pipe

# cat all tv csvs
# dirs = _find_dir("LGCP")
# dfs = [_read_csv_prefix(d; append=()) for d in dirs]

# df_lgcp = _hcat_all_dfs(dfs)
# df_lgcp.target .= "LGCP"
# df_lgcp.lr .= 0.001
# df_lgcp.batchsize .= 32
# df_lgcp.seed .= [1:5 ;]


# df_mfvi = CSV.read("summary.csv", DataFrame)

# df_mfvi_all = vcat(df_lgcp, df_mfvi, cols= :union)

# df_mfvi_all.flowtype .= "Baseline" 
# df_mfvi_all.kernel .= "MFVI"


# CSV.write("mvfi_all.csv", df_mfvi_all)


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

