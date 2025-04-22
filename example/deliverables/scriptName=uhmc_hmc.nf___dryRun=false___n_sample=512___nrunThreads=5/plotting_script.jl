using Plots, StatsPlots
using CSV, DataFrames

println(@__DIR__)
include(joinpath(@__DIR__, "../../plotting.jl"))

fig_dir = joinpath(@__DIR__, "figure/")

if !isdir(fig_dir)
    println("Creating figure directory: $fig_dir")
    mkdir(fig_dir)
end

df = CSV.read(joinpath(@__DIR__, "output/summary.csv"), DataFrame) 

targets = unique(df.target)
kernels = unique(df.kernel)
f = unique(df.flowtype)[1]
color_list = [1:4 ;]

for t in targets
    fig_name = "$(t)__$(_throw_dot(f))"
    println(fig_name)

    local ds = _subset_expt(df, Dict(:target => t, :kernel => kernels[1]))
    local fg = groupederrorline(
        ds, :Ts, :tv, :seed, :step_size;
        mark_nan = true,
        errorstyle = :ribbon,
        legend = :best,
        title = fig_name,
        groupcolor = color_list,
        linestyle = :solid,
        lw = 2,
    )

    local ds1 = _subset_expt(df, Dict(:target => t, :kernel => kernels[2]))
    
    add_groupederrorline!(
        fg,
        ds1, :Ts, :tv, :seed, :step_size;
        mark_nan = true,
        errorstyle = :ribbon,
        label = "",
        groupcolor = color_list,
        linestyle = :dash,
        lw = 2,
    )
    plot!(fg, ylabel = "Total Variation", xlabel = "flow length")
    plot!(fg, [0], [0], linestyle = :dash, label = "uHMC", color = "black")
    plot!(fg, [0], [0], linestyle = :solid, label = "HMC", color = "black")
    plot!(dpi = 600, size = (500, 400), margin = 10Plots.mm)
    savefig(fg, joinpath(fig_dir, fig_name * ".png"))
end
