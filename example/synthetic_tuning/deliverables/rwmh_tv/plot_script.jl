using Plots, StatsPlots
using CSV, DataFrames

println(@__DIR__)
include(joinpath(@__DIR__, "../../../julia_env/plotting.jl"))

# rwmg tv
df = CSV.read(joinpath(@__DIR__, "output/summary.csv"), DataFrame)

fig_dir = joinpath(@__DIR__, ".")

if !isdir(fig_dir)
    println("Creating figure directory: $fig_dir")
    mkpath(fig_dir)
end

targets = unique(df.target)
kernels = unique(df.kernel)
flowytpes = unique(df.flowtype)

for t in targets
    println("target: $t, stability check")
    local selector = Dict(
        :target => t,
    )
    local ds = _subset_expt(df, selector)

    x_key, y_key, rep_key, group_key = :Ts, :errors, :seed, :kernel

    local xs, y_mat, gs = _process_for_grouped_errorline(
        ds,
        x_key,
        y_key,
        rep_key,
        group_key,
    )

    local fg = StatsPlots.errorline(
        xs, log10.(y_mat),
        label = reshape(gs, 1, length(gs)),
        xlabel = x_key,
        legend = :best;
        errorstyle = :ribbon,
        legendtitle = "Kernel",
        title = "$(t) IRF invertibilty error",
        lw = 3,
        ylabel = "log10(Error)",
    )

    plot!(fg, dpi = 600, margin = 5Plots.mm, size = (500, 400))
    savefig(fg, joinpath(fig_dir, "$(t)__stability.png"))
end

