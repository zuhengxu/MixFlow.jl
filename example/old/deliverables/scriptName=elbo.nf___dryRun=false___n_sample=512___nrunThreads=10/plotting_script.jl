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
flowtypes = unique(df.flowtype) # remmove deterministic flows now--- there is bug
kernels = unique(df.kernel)[1:end-1] # no uncorrectHMC for now

for (t, k, f) in Iterators.product(targets, kernels, flowtypes)
    println("target: $t, kernel: $k, flowtype: $f")
    local selector = Dict(
        :target => t,
        :flowtype => f,
        :kernel => k,
    )
    local ds = _subset_expt(df, selector)

    ps = []
    for metric in [:elbo, :logZ, :ess]
        local fg = groupederrorline(
            ds, :flow_length, metric, :seed, :step_size;
            errorstyle = :ribbon,
            legend = :best,
            legendtitle = "step size",
            title = "$(t)__$(_throw_dot(k))__$(_throw_dot(f))",
        )

        if metric == :elbo
            hline!(fg, [0], color = :red, linestyle = :dash, lw = 2, label = "optimal")
        elseif metric == :logZ
            hline!(fg, [0], color = :red, linestyle = :dash, lw = 2, label = "optimal")
        elseif metric == :ess
            hline!(fg, [512], color = :red, linestyle = :dash, lw = 2, label = "n particles")
        end
        push!(ps, fg)

        # savefig(fg, joinpath(fig_dir, "$(t)__$(_throw_dot(k))__$(_throw_dot(f))_$(metric).png"))
    end
    fg_joined = plot(ps..., layout = (1, 3), dpi = 600, size = (1600, 400), margin = 10Plots.mm)
    savefig(fg_joined, joinpath(fig_dir, "$(t)__$(_throw_dot(k))__$(_throw_dot(f)).png"))
end
