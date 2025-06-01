include(joinpath(@__DIR__, "../rwmh/run_rwmh.jl"))
include(joinpath(@__DIR__, "../../julia_env/", "utils.jl"))
include(joinpath(@__DIR__, "../../julia_env/", "plotting.jl"))
using Pipe



# lgcp_dir = joinpath(@__DIR__, "LGCP_csv")

# cat all lgcp rwmh csvs
dirs = _find_dir("nuts")
dfs = []
for pth in dirs
    method, target, seed_s = split(pth, "_")
    seed = parse(Int, split(seed_s, ".")[1])

    df_tmp = _read_csv_prefix(pth; append=())
    df_tmp.target .= name
    df_tmp.seed .= seed
    
    push!(dfs, df_tmp)
end

dfs = vcat(dfs...)

df_n = CSV.read(joinpath(@__DIR__, "../deliverables/nuts.csv"), DataFrame)

df_n_all = vcat(df_n, dfs, cols=:union)

CSV.write(joinpath(@__DIR__, "../deliverables/nuts.csv"), df_n_all)
unique(df_n_all.target)
