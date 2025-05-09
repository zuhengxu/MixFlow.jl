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

include(joinpath(@__DIR__, "../julia_env/Model.jl"))
include(joinpath(@__DIR__, "../julia_env/plotting.jl"))

function tv_plot(
    combined_csvs_folder::String
)
    df = CSV.read(joinpath(combined_csvs_folder, "summary.csv"), DataFrame) 

    if !isdir("figure/")
        mkpath("figure/")
    end

    targets = unique(df.target)
    kernels = unique(df.kernel)
    flowtypes = unique(df.flowtype)

    for (t, k, f) in Iterators.product(targets, kernels, flowtypes)
        println("target: $t, kernel: $k, flowtype: $f")
        fig_name = "figure/$(t)__$(_throw_dot(f))__$(_throw_dot(k))"
        local selector = Dict(
            :target => t,
            :kernel => k,
            :flowtype => f,
        )
        try
        local ds = _subset_expt(df, selector)

        local fg = groupederrorline(
            ds, :Ts, :tv, :seed, :step_size;
            mark_nan = true,
            errorstyle = :ribbon,
            legend = :best,
            title = fig_name,
            linestyle = :solid,
            lw = 2,
        )
        
        plot!(fg, ylims = (0, 1))    # TV is between 0 and 1
        plot!(fg, ylabel = "Total Variation", xlabel = "flow length")
        plot!(fg, dpi = 600, size = (500, 400), margin = 10Plots.mm)
        savefig(fg, fig_name * ".png")
        catch e
            println("Error: $e")
            continue
        end
    end
end
