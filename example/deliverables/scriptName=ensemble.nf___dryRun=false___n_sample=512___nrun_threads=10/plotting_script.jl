using Plots, StatsPlots
using CSV, DataFrames

println(@__DIR__)
include(joinpath(@__DIR__, "../../plotting.jl"))

fig_dir = joinpath(@__DIR__, "figure/flowlength/")

if !isdir(fig_dir)
    println("Creating figure directory: $fig_dir")
    mkdir(fig_dir)
end

df = CSV.read(joinpath(@__DIR__, "output/summary.csv"), DataFrame) 
# df = CSV.read("output/summary.csv", DataFrame) 

targets = unique(df.target)
kernels = unique(df.kernel)
nchains = unique(df.nchains)

for (t, k, n) in Iterators.product(targets, kernels, nchains)
    println("target: $t, kernel: $k, nchains: $n")
    local selector = Dict(
        :target => t,
        :nchains => n,
        :kernel => k,
    )
    local ds = _subset_expt(df, selector)

    for metric in [:elbo, :logZ, :ess]
        local fg = groupederrorline(
            ds, :flow_length, metric, :seed, :step_size;
            errorstyle = :ribbon,
            legend = :best,
            legendtitle = "step size",
            title = "$(t)__$(_throw_dot(k))__nchain=$(n)",
        )

        if metric == :elbo
            hline!(fg, [0], color = :red, linestyle = :dash, lw = 2, label = "optimal")
        elseif metric == :logZ
            hline!(fg, [0], color = :red, linestyle = :dash, lw = 2, label = "optimal")
        elseif metric == :ess
            hline!(fg, [512], color = :red, linestyle = :dash, lw = 2, label = "n particles")
        end

        savefig(fg, joinpath(fig_dir, "$(t)__$(_throw_dot(k))__nchain=$(n)_$(metric).png"))
    end
end


# t = targets[1]
# k = kernels[1]
# n = nchains[1] 
# selector = Dict(
#     :target => t,
#     :nchains => n,
#     :kernel => k,
# )
# ds = _subset_expt(df, selector)

# performance over increasing nchains

fig_dir = joinpath(@__DIR__, "figure/nchains/")

if !isdir(fig_dir)
    println("Creating figure directory: $fig_dir")
    mkdir(fig_dir)
end
targets = unique(df.target)
kernels = unique(df.kernel)
flengths = unique(df.flow_length)

for (t, k, l) in Iterators.product(targets, kernels, flengths)
    println("target: $t, kernel: $k, flow_length: $l")
    local selector = Dict(
        :target => t,
        :kernel => k,
        :flow_length => l,
    )
    local ds = _subset_expt(df, selector)

    for metric in [:elbo, :logZ, :ess]
        local fg = groupederrorline(
            ds, :nchains, metric, :seed, :step_size;
            errorstyle = :ribbon,
            legend = :best,
            legendtitle = "step size",
            title = "$(t)__$(_throw_dot(k))__T=$(l)",
        )

        if metric == :elbo
            hline!(fg, [0], color = :red, linestyle = :dash, lw = 2, label = "optimal")
        elseif metric == :logZ
            hline!(fg, [0], color = :red, linestyle = :dash, lw = 2, label = "optimal")
        elseif metric == :ess
            hline!(fg, [512], color = :red, linestyle = :dash, lw = 2, label = "n particles")
        end

        savefig(fg, joinpath(fig_dir, "$(t)__$(_throw_dot(k))__T=$(l)_$(metric).png"))
    end
end

