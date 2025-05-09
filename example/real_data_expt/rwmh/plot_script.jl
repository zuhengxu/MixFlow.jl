using MixFlow: log_normalization_constant
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

include(joinpath(@__DIR__, "../../Model.jl"))
include(joinpath(@__DIR__, "../../plotting.jl"))

sais_csv = joinpath(@__DIR__, "../ground_truth/deliverables/real_sais.csv")
df_sais = CSV.read(sais_csv, DataFrame)
df_nf = CSV.read(joinpath(@__DIR__, "../normflow/deliverables/50k_no_lgcp/output/summary.csv"), DataFrame)
df = CSV.read(joinpath(@__DIR__, "deliverables/scriptName=run_rwmh.nf___dryRun=false___n_sample=64___nrunThreads=1/output/summary.csv"), DataFrame)

df = _remove_nan(df)
df_sais = _remove_nan(df_sais)
rename!(df_sais, :log_norm_constant => :logZ)
df_nf = _remove_nan(df_nf)

# targets = unique(df.target)
# kernels = unique(df.kernel)
# flowtypes = unique(df.flowtype)

# k = "MF.RWMH"
# metric = :elbo


function make_box_plot(
    ds, ds_sais, ds_nf_best, t::String, m::Symbol; best_lr, best_nlayer, size = (800, 400), kwargs...
)
    selector = Dict(
        :target => t,
    )
    ds = _subset_expt(df, selector)
    ds_sais = _subset_expt(df_sais, selector)
    ds_nf = _subset_expt(df_nf, selector)

    ds_nf_best = _subset_expt(ds_nf, Dict(:lr => best_lr, :nlayer => best_nlayer))
    gt = m == :ess ? [1.0] : ds_sais[!, m]
    gt_label = m == :ess ? "optimal" : "truth"

    @info "$t, $m"
    fg = @df ds boxplot(:flowtype, cols(m), markerstrokewidth = 0.5, fillalpha = 0.8, label = "")
    hline!(gt, color = :green, lw = 3, label = gt_label, linestyle = :dash)
    plot!(legend = :best)
    plot!(;kwargs...)
    title!("$t IRF-Flows")

    fg_nf = @df ds_nf_best boxplot(:flowtype, cols(m), markerstrokewidth = 0.5, fillalpha = 0.8)
    hline!(gt, color = :green, lw = 3, label = gt_label, linestyle = :dash)
    plot!(;kwargs...)
    title!("$t Normalizing Flows")

    fig = plot(fg, fg_nf, layout = (1, 2), dpi = 600, margin = 5Plots.mm, size = size)
    savefig(fig, "figure/$(t)_$m.png")
    return fig
end

name = "Brownian"
best_lr = 0.001
best_nlayer = 3

make_box_plot(
    df, df_sais, df_nf, name, :elbo;
    best_lr = best_lr,
    best_nlayer = best_nlayer,
    ylims = (0, 5),
    ylabel = "ELBO",
    size = (1000, 400),
)
make_box_plot(
    df, df_sais, df_nf, name, :logZ;
    best_lr = best_lr,
    best_nlayer = best_nlayer,
    ylabel = "log normalization constant",
    ylims = (0, 16),
    size = (1000, 400),
)
make_box_plot(
    df, df_sais, df_nf, name, :ess;
    best_lr = best_lr,
    best_nlayer = best_nlayer,
    yscale = :log10,
    ylabel = "Importance Sampling ESS (per particle)",
    size = (1000, 400),
    ylims = (1e-5, 2),
)


name= "Sonar"
best_lr = 0.001
best_nlayer = 3

make_box_plot(
    df, df_sais, df_nf, name, :elbo;
    best_lr = best_lr,
    best_nlayer = best_nlayer,
    ylims = (-135, -100),
    ylabel = "ELBO",
    size = (1000, 400),
)

make_box_plot(
    df, df_sais, df_nf, name, :logZ;
    best_lr = best_lr,
    best_nlayer = best_nlayer,
    ylabel = "log normalization constant",
    ylims = (-140, -25),
    size = (1000, 400),
)

make_box_plot(
    df, df_sais, df_nf, name, :ess;
    best_lr = best_lr,
    best_nlayer = best_nlayer,
    yscale = :log10,
    ylabel = "Importance Sampling ESS (per particle)",
    size = (1000, 400),
    ylims = (1e-5, 2),
)

name = "TReg"
best_lr = 0.001
best_nlayer = 3


make_box_plot(
    df, df_sais, df_nf, name, :elbo;
    best_lr = best_lr,
    best_nlayer = best_nlayer,
    ylabel = "ELBO",
    ylims = (-145.6, -145.3),
    size = (1000, 400),
)

make_box_plot(
    df, df_sais, df_nf, name, :logZ;
    best_lr = best_lr,
    best_nlayer = best_nlayer,
    ylabel = "log normalization constant",
    ylims = (-145.6, -145.3),
    size = (1000, 400),
)

make_box_plot(
    df, df_sais, df_nf, name, :ess;
    best_lr = best_lr,
    best_nlayer = best_nlayer,
    # yscale = :log10,
    ylabel = "Importance Sampling ESS (per particle)",
    size = (1000, 400),
    ylims = (0.8, 1),
)


namea = "LGCP"
best_lr = 0.001
best_nlayer = 3

df_nf_lgcp = CSV.read(joinpath(@__DIR__, "../normflow/deliverables/scriptName=run_nf.nf___dryRun=false___n_sample_eval=1024___nrunThreads=1/output/summary.csv"), DataFrame)
df_nf_lgcp = _remove_nan(df_nf_lgcp)
ds_nf_best_lgcp = _subset_expt(df_nf_lgcp, Dict(:lr => 0.0001, :niters => 10000))


make_box_plot(
    df, df_sais, df_nf, name, :elbo;
    best_lr = best_lr,
    best_nlayer = best_nlayer,
    ylims = (-135, -100),
    ylabel = "ELBO",
    size = (1000, 400),
)

make_box_plot(
    df, df_sais, df_nf, name, :logZ;
    best_lr = best_lr,
    best_nlayer = best_nlayer,
    ylabel = "log normalization constant",
    ylims = (-140, -25),
    size = (1000, 400),
)

make_box_plot(
    df, df_sais, df_nf, name, :ess;
    best_lr = best_lr,
    best_nlayer = best_nlayer,
    yscale = :log10,
    ylabel = "Importance Sampling ESS (per particle)",
    size = (1000, 400),
    ylims = (1e-5, 2),
)



# t = "TReg"
# selector = Dict(
#     :target => t,
# )
# ds = _subset_expt(df, selector)
# ds_sais = _subset_expt(df_sais, selector)
# ds_nf = _subset_expt(df_nf, selector)

# # choose best nf
# fg_nf_all = @df ds_nf groupedboxplot(
#     :flowtype, :elbo, group = (:lr, :nlayer), markerstrokewidth = 0.5, fillalpha = 0.8, 
#     # ylims = (-300, -100)
# )
