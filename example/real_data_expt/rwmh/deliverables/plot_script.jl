include(joinpath(@__DIR__, "../", "run_rwmh.jl"))
include(joinpath(@__DIR__, "../../../julia_env/", "utils.jl"))
include(joinpath(@__DIR__, "../../../julia_env/", "plotting.jl"))
using Pipe

# process nfs
df_nf = CSV.read("nf.csv", DataFrame)
df_nf = _remove_nan(df_nf)
df_nf.kernel .= df_nf.flowtype
df_nf.flowtype .= "NormFlow"


df_mvfi = CSV.read("mvfi_all.csv", DataFrame)
df_rwmh = CSV.read("rwmh_all.csv", DataFrame)
df_rwmh.kernel .= "RWMH"

# rename DeterministicMixFlow to HomogeneousMixFlow
df_rwmh.flowtype .= replace(df_rwmh.flowtype, "DeterministicMixFlow" => "HomogeneousMixFlow")

function df_best_nf_setting(df_nf, t::String)
    # t = "Banana"
    ds_nf = _subset_expt(df_nf, Dict(:target => t))

    # find lr that gives the lowest TV
    dnf_s = @pipe ds_nf |>
                     groupby(_, [:kernel, :nlayer, :lr]) |>
                     combine(_, :elbo => median, :logZ => median, :ess => median) |>
                     groupby(_, [:kernel]) |>
                     combine(_, [:lr, :nlayer, :elbo_median] =>
                             ((l, n, t) -> (best_step_size=l[argmax(t)], best_nlayer=n[argmax(t)] ,elbo_max=maximum(t))) =>
                             AsTable)
    @show dnf_s

    # build a dictionary of flowtype and its best step size
    nfts = dnf_s.kernel
    nfss = dnf_s.best_step_size
    nfnlayer = dnf_s.best_nlayer
    nf_dict = Dict(
        nfts[i] => (nfss[i], nfnlayer[i]) for i in 1:length(nfts)
    )

    # go back to the original ds_nf and get the rows with flowtype with its best_step_size
    dssg_nf = @pipe groupby(ds_nf, [:kernel, :nlayer ,:lr]) 

    dsnf_final = filter(x -> (x.lr[1], x.nlayer[1]) == nf_dict[x.kernel[1]], dssg_nf; ungroup = true)
    return dsnf_final
end

# dsn_tmp = df_best_nf_setting(df_nf, "TReg")

function df_our_setting(df, t::String)
    ds = _subset_expt(df, Dict(:target => t))
    return ds
end

function df_mfvi_setting(df, t::String)
    ds = _subset_expt(df, Dict(:target => t))
    return ds
end

function ground_truth_setting(t::String)
    res = JLD2.load(joinpath(@__DIR__, "ground_truth_res/$(t)_sais.jld2"))
    res["logZ"], res["Mean"], res["Std"]
end

if !isdir("figure")
    mkpath("figure")
end

for t in real_data_list    
    try
        println("target: $t")

        t = String(t)
        # get the best step size for each flowtype
        dfnf = df_best_nf_setting(df_nf, t)
        dfmfvi = df_mfvi_setting(df_mvfi, t)
        df_our = df_our_setting(df_rwmh, t)
        gt_logz = ground_truth_setting(t)[1]

        # combine the two dataframes
        df_combined = vcat(dfmfvi, dfnf, df_our, cols=:union)

        fg = @df df_combined groupedboxplot(
            :flowtype, :ess, group = :kernel, yscale = :log10, markerstrokewidth = 0.5, fillalpha = 0.8; 
            markersize = 3,
            # title = "$t ",
            ylabel = "ESS per sample",
            xtickfontsize = 10, ytickfontsize = 12, yguidefontsize = 15,
            legendfontsize = 11, titlefontsize = 18,
            xrotation = 8,
        )
        plot!(fg, legend = :best)
        plot!(fg, dpi = 600, size = (600, 400), margin = 5Plots.mm)
        hline!([1.0], color = :black, lw = 2, linestyle = :dot, label = "perfect sample", alpha = 0.8)
        savefig(fg, "figure/$(t)_ess.png")

        fg = @df df_combined groupedboxplot(
            :flowtype, :elbo, group = :kernel, 
            markerstrokewidth = 0.5, fillalpha = 0.8; 
            markersize = 3,
            ylabel = "ELBO",
            xtickfontsize = 10, ytickfontsize = 12, yguidefontsize = 15,
            legendfontsize = 11, titlefontsize = 18,
            xrotation = 8,
        )
        plot!(fg, legend = :best)
        plot!(fg, dpi = 600, size = (600, 400), margin = 5Plots.mm)
        hline!([gt_logz], color = :black, lw = 2, linestyle = :dot, label = "true logZ", alpha = 0.8)
        savefig(fg, "figure/$(t)_elbo_raw.png")

        fg = @df df_combined groupedboxplot(
            :flowtype, :elbo, group = :kernel, 
            markerstrokewidth = 0.5, fillalpha = 0.8; 
            markersize = 3,
            ylims = t == "TReg" ? (gt_logz-1, gt_logz+1) : (gt_logz-50, gt_logz+300),
            ylabel = "ELBO",
            xtickfontsize = 10, ytickfontsize = 12, yguidefontsize = 15,
            legendfontsize = 11, titlefontsize = 18,
            xrotation = 8,
        )
        plot!(fg, legend = :best)
        plot!(fg, dpi = 600, size = (600, 400), margin = 5Plots.mm)
        hline!([gt_logz], color = :black, lw = 2, linestyle = :dot, label = "true logZ", alpha = 0.8)
        savefig(fg, "figure/$(t)_elbo_restrict.png")

        fg = @df df_combined groupedboxplot(
            :flowtype, :logZ, group = :kernel, 
            markerstrokewidth = 0.5, fillalpha = 0.8; 
            markersize = 3,
            ylabel = "log normalization constant",
            xtickfontsize = 10, ytickfontsize = 12, yguidefontsize = 15,
            legendfontsize = 11, titlefontsize = 18,
            xrotation = 8,
        )
        plot!(fg, legend = :best)
        plot!(fg, dpi = 600, size = (600, 400), margin = 5Plots.mm)
        hline!([gt_logz], color = :black, lw = 2, linestyle = :dot, label = "true logZ", alpha = 0.8)
        savefig(fg, "figure/$(t)_logz_raw.png")

        fg = @df df_combined groupedboxplot(
            :flowtype, :logZ, group = :kernel, 
            markerstrokewidth = 0.5, fillalpha = 0.8; 
            markersize = 3,
            ylims = t == "TReg" ? (gt_logz-1, gt_logz+1) : (gt_logz-100, gt_logz+500),
            ylabel = "log normalization constant",
            xtickfontsize = 10, ytickfontsize = 12, yguidefontsize = 15,
            legendfontsize = 11, titlefontsize = 18,
            xrotation = 8,
        )
        plot!(fg, legend = :best)
        plot!(fg, dpi = 600, size = (600, 400), margin = 5Plots.mm)
        hline!([gt_logz], color = :black, lw = 2, linestyle = :dot, label = "true logZ", alpha = 0.8)
        savefig(fg, "figure/$(t)_logz_restrict.png")

    catch e
        println("Error in $t: $e")
        continue
    end
end

t = "LGCP"

t = String(t)
# get the best step size for each flowtype
# dfnf = df_best_nf_setting(df_nf, t)
dfmfvi = df_mfvi_setting(df_mvfi, t)
df_our = df_our_setting(df_rwmh, t)
gt_logz = ground_truth_setting(t)[1]

# combine the two dataframes
df_combined = vcat(dfmfvi, df_our, cols=:union)

fg = @df df_combined boxplot(
    :flowtype, :ess, group = :kernel, yscale = :log10, markerstrokewidth = 0.5, fillalpha = 0.8; 
    markersize = 3,
    # title = "$t ",
    ylabel = "ESS per sample",
    xtickfontsize = 10, ytickfontsize = 12, yguidefontsize = 15,
    legendfontsize = 11, titlefontsize = 18,
    xrotation = 8,
)
plot!(fg, legend = :best)
plot!(fg, dpi = 600, size = (600, 400), margin = 5Plots.mm)
hline!([1.0], color = :black, lw = 2, linestyle = :dot, label = "perfect sample", alpha = 0.8)
savefig(fg, "figure/$(t)_ess.png")

fg = @df df_combined boxplot(
    :flowtype, :elbo, group = :kernel, 
    markerstrokewidth = 0.5, fillalpha = 0.8; 
    markersize = 3,
    # ylims = (-10, 10),
    ylabel = "ELBO",
    xtickfontsize = 10, ytickfontsize = 12, yguidefontsize = 15,
    legendfontsize = 11, titlefontsize = 18,
    xrotation = 8,
)
plot!(fg, legend = :best)
plot!(fg, dpi = 600, size = (600, 400), margin = 5Plots.mm)
hline!([gt_logz], color = :black, lw = 2, linestyle = :dot, label = "true logZ", alpha = 0.8)
savefig(fg, "figure/$(t)_elbo_raw.png")

# fg = @df df_combined groupedboxplot(
#     :flowtype, :elbo, group = :kernel, 
#     markerstrokewidth = 0.5, fillalpha = 0.8; 
#     markersize = 3,
#     ylims = (-40, 20),
#     ylabel = "ELBO",
#     xtickfontsize = 10, ytickfontsize = 12, yguidefontsize = 15,
#     legendfontsize = 11, titlefontsize = 18,
#     xrotation = 8,
# )
# plot!(fg, legend = :best)
# plot!(fg, dpi = 600, size = (600, 400), margin = 5Plots.mm)
# hline!([gt_logz], color = :black, lw = 2, linestyle = :dot, label = "true logZ", alpha = 0.8)
# savefig(fg, "figure/$(t)_elbo_restrict.png")

fg = @df df_combined boxplot(
    :flowtype, :logZ, group = :kernel, 
    markerstrokewidth = 0.5, fillalpha = 0.8; 
    markersize = 3,
    # title = "$t ",
    ylabel = "log normalization constant",
    xtickfontsize = 10, ytickfontsize = 12, yguidefontsize = 15,
    legendfontsize = 11, titlefontsize = 18,
    xrotation = 8,
)
plot!(fg, legend = :best)
plot!(fg, dpi = 600, size = (600, 400), margin = 5Plots.mm)
hline!([gt_logz], color = :black, lw = 2, linestyle = :dot, label = "true logZ", alpha = 0.8)
savefig(fg, "figure/$(t)_logz.png")

