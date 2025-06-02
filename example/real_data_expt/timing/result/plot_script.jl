include(joinpath(@__DIR__, "../../../julia_env/", "utils.jl"))
include(joinpath(@__DIR__, "../../../julia_env/", "plotting.jl"))
using Pipe


# flowtypes = ["BackwardIRFMixFlow", "DeterministicMixFlow", "EnsembleIRFFlow"]

# cat all nuts timing
dirs = _find_dir("nuts")
dfs = []
for pth in dirs
    name, method, _, seed_s = split(pth, "_")
    seed = parse(Int, split(seed_s, ".")[1])

    df_tmp = _read_csv_prefix(pth; append=())
    df_tmp.time .= df_tmp.nuts_time
    df_tmp.method .= "NUTS"
    df_tmp.seed .= seed
    
    push!(dfs, df_tmp)
end
dfs = map(identity, dfs)
df_n = _hcat_all_dfs(dfs)
select!(df_n, [:name, :method, :seed, :time])

# cat all irfflow timing
dirs = _find_dir("Flow")
dfs = []
for pth in dirs
    name, flowtype, seed_s = split(pth, "_")
    seed = parse(Int, split(seed_s, ".")[1])

    df_tmp = _read_csv_prefix(pth; append=())
    df_tmp.time .= df_tmp.irfflow_time
    df_tmp.method .= flowtype
    df_tmp.seed .= seed
    
    push!(dfs, df_tmp)
end
dfs = map(identity, dfs)
df_mf = _hcat_all_dfs(dfs)
# remove flow_type and irfflow_time column
select!(df_mf, [:name, :method, :seed, :time])

# rename DeterministicMixFlow to HomogeneousMixFlow
df_mf.method .= replace(df_mf.method, "DeterministicMixFlow" => "HomogeneousMixFlow")

df = vcat(df_mf, df_n, cols=:setequal)


# cat all flow and baseline timing
dirs = _find_dir("vi")
dfs = []
for pth in dirs
    name, method, _ = split(pth, "_")
    seed = 1

    df_tmp = _read_csv_prefix(pth; append=())
    df_tmp.NSF .= df_tmp.normflow_time
    df_tmp.RealNVP .= df_tmp.realnvp_time
    
    push!(dfs, df_tmp)
end
dfs = map(identity, dfs)
df_vi = _hcat_all_dfs(dfs)
select!(df_vi, [:name, :NSF, :RealNVP, :baseline_time, :reference_time])

# for df_vi, change RealNVP to missing for name "SparseRegression" because it's naned in training
df_vi.RealNVP[df_vi.name .== "SparseRegression"] .= NaN

#######################
# start making plots
#######################
targets = unique(df.name)

for name in targets
    ds = df[df.name .== name, :]
    ds_vi = df_vi[df_vi.name .== name, :]
    ref_time = ds_vi.reference_time[1]

    # add reference time to all methods (we didn't record that when running the timing)
    ds.time .= ds.time .+ ref_time

    fg = @df ds boxplot(
        :method, :time,
        markerstrokewidth=0.5, fillalpha=0.8;
        markersize=3,
        ylabel="Time (seconds)",
        yscale = :log10,
        xtickfontsize=10, ytickfontsize=12, yguidefontsize=15,
        legendfontsize=11, titlefontsize=18,
        xrotation=-10,
        label="",
    )

    hline!(ds_vi.baseline_time, color=1, lw=3, linestyle=:dash, label="MFVI")
    try 
        hline!(ds_vi.NSF, color=3, lw=3, linestyle=:dashdot, label="NSF")
    catch
    end


    try
        if !isnan(ds_vi.RealNVP[1])
            hline!(ds_vi.RealNVP, color=4, lw=3, linestyle=:dot, label="RealNVP")
        end
    catch
    end
    plot!(fg, legend=:best)
    plot!(fg, dpi=250, size=(600, 400), margin=5Plots.mm)
    savefig(fg, "../figure/$(name)_cost_no_irfmf.png")
end
