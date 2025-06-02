include(joinpath(@__DIR__, "../plotting.jl"))

# combine tv csvs
df_tv_mf = CSV.read("tv_mixflow_all.csv", DataFrame)

# change flowtype "DeterministicMixFlow" to "HomogeneousMixFlow"
df_tv_mf.flowtype .= replace.(df_tv_mf.flowtype, "DeterministicMixFlow" => "HomogeneousMixFlow")

tv_plot_compare_flowtype(df_tv_mf)

# df_tv_ensemble = CSV.read("tv_ensemble_all.csv", DataFrame)



#####################
# elbo plot
#####################

# read irfflow eval csvs
df = CSV.read("elbo_all.csv", DataFrame)
df.flowtype .= _throw_dot.(df.flowtype) 
df.kernel .= _throw_dot.(df.kernel) 
# change flowtype "DeterministicMixFlow" to "HomogeneousMixFlow"
df.flowtype .= replace.(df.flowtype, "DeterministicMixFlow" => "HomogeneousMixFlow")
# remove the columns :cost
select!(df, Not([:cost]))

# add a column class for df, filling it with "Exact Flow"
df.class .= "OurMethod"

# read the normflow results
df_nf = CSV.read(
    joinpath(
        @__DIR__,
        "../../normflow/deliverables/scriptName=normflow.nf___dryRun=false___n_sample_eval=1024___nrunThreads=2/output/summary.csv",
    ),
    DataFrame,
)

df_nf = _remove_nan(df_nf)
df_nf.class .= "NormalizingFlow"
targets = unique(df.target)

local fg
dpi = 200
figsize = (600, 400)

for t in targets
    println("target: $t")

    t = String(t)
    # get the best step size for each flowtype
    dfnf = df_best_nf_setting(df_nf, t)
    dfnf.kernel .= dfnf.flowtype
    dfnf.flowtype .= "NormFlow"

    df_our = df_our_setting(df, t)

    # combine the two dataframes
    df_combined = vcat(dfnf, df_our, cols=:union)

    fg = @df df_combined groupedboxplot(
        :flowtype, :ess, group = :kernel, yscale = :log10, markerstrokewidth = 0.5, fillalpha = 0.8; 
        markersize = 3,
        # title = "$t ",
        ylabel = "ESS per sample",
        xtickfontsize = 10, ytickfontsize = 12, yguidefontsize = 15,
        legendfontsize = 11, titlefontsize = 18,
        xrotation = 8,
    )
    plot!(fg, legend = :bottomright)
    plot!(fg, dpi = dpi, size = figsize, margin = 5Plots.mm)
    hline!([1.0], color = :black, lw = 2, linestyle = :dot, label = "perfect sample", alpha = 0.8)
    savefig(fg, "figure/$(t)_ess.png")

    fg = @df df_combined groupedboxplot(
        :flowtype, :elbo, group = :kernel, 
        markerstrokewidth = 0.5, fillalpha = 0.8; 
        markersize = 3,
        # ylims = (-10, 10),
        ylabel = "ELBO",
        xtickfontsize = 10, ytickfontsize = 12, yguidefontsize = 15,
        legendfontsize = 11, titlefontsize = 18,
        xrotation = 8,
    )
    plot!(fg, legend = :bottomright)
    plot!(fg, dpi = dpi, size = figsize, margin = 5Plots.mm)
    hline!([0.0], color = :black, lw = 2, linestyle = :dot, label = "true logZ", alpha = 0.8)
    savefig(fg, "figure/$(t)_elbo_raw.png")

    fg = @df df_combined groupedboxplot(
        :flowtype, :elbo, group = :kernel, 
        markerstrokewidth = 0.5, fillalpha = 0.8; 
        markersize = 3,
        ylims = (-40, 20),
        ylabel = "ELBO",
        xtickfontsize = 10, ytickfontsize = 12, yguidefontsize = 15,
        legendfontsize = 11, titlefontsize = 18,
        xrotation = 8,
    )
    plot!(fg, legend = :bottomright)
    plot!(fg, dpi = dpi, size = figsize, margin = 5Plots.mm)
    hline!([0.0], color = :black, lw = 2, linestyle = :dot, label = "true logZ", alpha = 0.8)
    savefig(fg, "figure/$(t)_elbo_restrict.png")

    fg = @df df_combined groupedboxplot(
        :flowtype, :logZ, group = :kernel, 
        markerstrokewidth = 0.5, fillalpha = 0.8; 
        markersize = 3,
        # title = "$t ",
        ylabel = "log normalization constant",
        xtickfontsize = 10, ytickfontsize = 12, yguidefontsize = 15,
        legendfontsize = 11, titlefontsize = 18,
        xrotation = 8,
    )
    plot!(fg, legend = :bottomright)
    plot!(fg, dpi = dpi, size = figsize, margin = 5Plots.mm)
    hline!([0.0], color = :black, lw = 2, linestyle = :dot, label = "true logZ", alpha = 0.8)
    savefig(fg, "figure/$(t)_logz.png")

end
