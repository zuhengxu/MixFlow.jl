#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/compare_traces/trace_plotting.jl")
grouped_ess_plot(
    "output";
    dpi = 600,
    size = (1000, 800),
    margin = 10Plots.mm,
    xtickfontsize = 18, ytickfontsize = 18, yguidefontsize = 18,
)
