include("../plotting.jl")
include("../../julia_env/utils.jl")
include("../../julia_env/plotting.jl")


df = CSV.read("elbo_all.csv", DataFrame)

df_nf = CSV.read("../../synthetic_expt/normflow/deliverables/scriptName=normflow.nf___dryRun=false___n_sample_eval=1024___nrunThreads=2/output/summary.csv", DataFrame)
df_nf = _remove_nan(df_nf)
df_nf.class .= "NormalizingFlow"

df.flowtype .= _throw_dot.(df.flowtype) 
df.kernel .= _throw_dot.(df.kernel) 
# remove the columns :cost
select!(df, Not([:cost]))
# add a column class for df, filling it with "Exact Flow"
df.class .= "OurMethod"

function df_best_nf_setting(df_nf, t::String)
    # t = "Banana"
    ds_nf = _subset_expt(df_nf, Dict(:target => t))

    # find lr that gives the lowest TV
    dnf_s = @pipe ds_nf |>
                     groupby(_, [:flowtype, :lr]) |>
                     combine(_, :tv => median, :elbo => median, :logZ => median, :ess => median) |>
                     groupby(_, [:flowtype]) |>
                     combine(_, [:lr, :tv_median] =>
                         ((l, t) -> (best_step_size=l[argmin(t)], tv_min=minimum(t))) =>
                             AsTable)
    @show dnf_s

    # build a dictionary of flowtype and its best step size
    nfts = dnf_s.flowtype
    nfss = dnf_s.best_step_size
    nf_dict = Dict(
        nfts[i] => nfss[i] for i in 1:length(nfts)
    )

    # go back to the original ds_nf and get the rows with flowtype with its best_step_size
    dssg_nf = @pipe groupby(ds_nf, [:flowtype, :lr]) 

    dsnf_final = filter(x -> x.lr[1] == nf_dict[x.flowtype[1]], dssg_nf; ungroup = true)
    return dsnf_final
end

function df_our_setting(df, t::String)

    ds = _subset_expt(df, Dict(:target => t))
    # find lr that gives the highest ess
    ds_s = @pipe ds |>
                     groupby(_, [:kernel, :step_size]) |>
                     combine(_, :elbo => median, :logZ => median, :ess => median) |>
                     groupby(_, [:kernel]) |>
                     combine(_, [:step_size, :ess_median] =>
                         ((l, e) -> (best_step_size=l[argmax(e)], ess_max=maximum(e))) =>
                             AsTable)
    @show ds_s

    ks = ds_s.kernel
    ss = ds_s.best_step_size
    # build a dictionary of kernel and its best step size
    ks_dict = Dict(
        ks[i] => ss[i] for i in 1:length(ks)
    )

    # go back to the original ds and get the rows with kernel with its best_step_size
    dsg = @pipe groupby(ds, [:kernel, :step_size])
    ds_final = filter(x -> x.step_size[1] == ks_dict[x.kernel[1]], dsg; ungroup = true)

    return ds_final
end

targets = unique(df.target)

local fg

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
    plot!(fg, dpi = 600, size = (600, 400), margin = 5Plots.mm)
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
    plot!(fg, dpi = 600, size = (600, 400), margin = 5Plots.mm)
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
    plot!(fg, dpi = 600, size = (600, 400), margin = 5Plots.mm)
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
    plot!(fg, dpi = 600, size = (600, 400), margin = 5Plots.mm)
    hline!([0.0], color = :black, lw = 2, linestyle = :dot, label = "true logZ", alpha = 0.8)
    savefig(fg, "figure/$(t)_logz.png")

end
