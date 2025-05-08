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

include(joinpath(@__DIR__, "../mfvi.jl"))
include(joinpath(@__DIR__, "../Model.jl"))
include(joinpath(@__DIR__, "../evaluation.jl"))
include(joinpath(@__DIR__, "../plotting.jl"))

# tv_uhmc = run_tv_sweep(1, "Funnel", MF.DeterministicMixFlow, MF.uncorrectHMC, 10, 0.1; nsample = 4)
# tv_hmc = run_tv_sweep(1, "Funnel", MF.DeterministicMixFlow, MF.HMC, 10, 0.1; nsample = 4)


function uhmc_hmc_tv_plot(
    combined_csvs_folder::String
)
    df = CSV.read(joinpath(combined_csvs_folder, "summary.csv"), DataFrame) 

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
        savefig(fg, fig_name * ".png")
    end
end
