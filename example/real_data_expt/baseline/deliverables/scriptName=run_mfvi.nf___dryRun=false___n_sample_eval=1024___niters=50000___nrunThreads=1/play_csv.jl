include(joinpath(@__DIR__, "../../julia_env/", "utils.jl"))
include(joinpath(@__DIR__, "../../julia_env/", "plotting.jl"))

# cat all tv csvs
dirs = _find_dir("LGCP")
dfs = [_read_csv_prefix(d; append=()) for d in dirs]

df_lgcp = _hcat_all_dfs(dfs)
df_lgcp.target .= "LGCP"
df_lgcp.lr .= 0.001
df_lgcp.batchsize .= 32
df_lgcp.seed .= [1:5 ;]


df_mfvi = CSV.read("summary.csv", DataFrame)

df_mfvi_all = vcat(df_lgcp, df_mfvi, cols= :union)

df_mfvi_all.flowtype .= "Baseline" 
df_mfvi_all.kernel .= "MFVI"


CSV.write("mvfi_all.csv", df_mfvi_all)


