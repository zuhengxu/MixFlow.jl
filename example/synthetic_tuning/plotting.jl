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

function tv_plot(df::DataFrame)

    targets = unique(df.target)
    kernels = unique(df.kernel)
    flowtypes = unique(df.flowtype)

    for (t, k, f) in Iterators.product(targets, kernels, flowtypes)
        println("target: $t, kernel: $k, flowtype: $f")
        fig_name = "$(t)__$(_throw_dot(f))__$(_throw_dot(k))"
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

tv_plot(combined_csvs_folder::String) = begin
    df = CSV.read(joinpath(combined_csvs_folder, "summary.csv"), DataFrame) 
    tv_plot(df)
end


function ensemble_flowlength_plot(
    df::DataFrame, metric::Symbol; 
    create_fig_dir=true, fig_dir::String = "figure/flowlength/", 
    kwargs...
)
        
    if create_fig_dir && (!isdir(fig_dir))
        println("Creating figure directory: $fig_dir")
        mkpath(fig_dir)
    end

    targets = unique(df.target)
    kernels = unique(df.kernel)

    for (t, k) in Iterators.product(targets, kernels)
        try
            println("target: $t, kernel: $k")
            local selector = Dict(
                :target => t,
                :kernel => k,
            )
            local ds = _subset_expt(df, selector)

            local nc_mx = maximum(unique(ds.nchains))
            local ds = _subset_expt(ds, Dict(:nchains => nc_mx))

            local fg = groupederrorline(
                ds, :flow_length, metric, :seed, :step_size;
                errorstyle = :ribbon,
                legend = :best,
                legendtitle = "step size",
                title = "$(t)__$(_throw_dot(k))__nchain=$(nc_mx)",
            )
            plot!(fg; kwargs...)
            plot!(fg, dpi = 600, size = (500, 400), margin = 10Plots.mm)
            savefig(fg, joinpath(fig_dir, "$(t)__$(_throw_dot(k))__nchain=$(nc_mx).png"))
        catch e
            println("Error: $e")
            continue
        end
    end
end

function ensemble_nchains_plot(
    df::DataFrame, metric::Symbol; 
    create_fig_dir=true, fig_dir::String = "figure/nchains/", 
    kwargs...
)
        
    if create_fig_dir && (!isdir(fig_dir))
        println("Creating figure directory: $fig_dir")
        mkpath(fig_dir)
    end

    targets = unique(df.target)
    kernels = unique(df.kernel)
    nchains = unique(df.nchains)


    for (t, k) in Iterators.product(targets, kernels)
        try
            println("target: $t, kernel: $k")
            local selector = Dict(
                :target => t,
                :kernel => k,
            )
            local ds = _subset_expt(df, selector)

            local fl_mx = maximum(unique(ds.flow_length))
            local ds = _subset_expt(ds, Dict(:flow_length => fl_mx))
            println("max flow length: ", fl_mx)

            local fg = groupederrorline(
                ds, :nchains, metric, :seed, :step_size;
                errorstyle = :ribbon,
                legend = :best,
                legendtitle = "step size",
                title = "$(t)__$(_throw_dot(k))__T=$(fl_mx)",
            )
            plot!(fg; kwargs...)
            plot!(fg, dpi = 600, size = (500, 400), margin = 10Plots.mm)
            savefig(fg, joinpath(fig_dir, "$(t)__$(_throw_dot(k))__nchain=$(n).png"))
        catch e
            println("Error: $e")
            continue
        end
    end
end

# df = CSV.read(joinpath(@__DIR__, "deliverables/tv_ensemble.csv"), DataFrame) 
# ensemble_flowlength_plot(
#     df, :tv; 
#     create_fig_dir=true, fig_dir="figure/flowlength/", 
#     ylabel="Total Variation", xlabel="flow length",
#     ylims=(0, 1),
# )

# ensemble_nchains_plot(
#     df, :tv; 
#     create_fig_dir=true, fig_dir="figure/nchains/", 
#     ylabel="Total Variation", xlabel="Number of Ensembles",
#     ylims=(0, 1),
# )

# plots comparing three flows
function tv_plot_compare_flowtype(df::DataFrame)

    targets = unique(df.target)
    kernels = unique(df.kernel)
    flowtypes = unique(df.flowtype)

    for (t, k) in Iterators.product(targets, kernels)
        println("target: $t, kernel: $k")
        fig_name = "$(t)__$(_throw_dot(k))"
        local selector = Dict(
            :target => t,
            :kernel => k,
        )
        try
            local ds = _subset_expt(df, selector)
            Tm = maximum(unique(ds.Ts))
            local dsm = _subset_expt(ds, Dict(:Ts => Tm))
            local dsg = groupby(dsm, [:step_size])
            local dsc = combine(dsg, :tv => median)
            # find the :step_size with the largest median tv
            sort!(dsc, :tv_median)
            # find the :step_size with the largest median tv
            s = dsc[1, :step_size]

            local dss = _subset_expt(ds, Dict(:step_size => s))

            local fg = groupederrorline(
                dss, :Ts, :tv, :seed, :flowtype;
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

# df = CSV.read("/home/zuheng/Research/MixFlow.jl/example/synthetic_tuning/deliverables/tv_mixflow.csv", DataFrame)
# df_rwmh = CSV.read("/home/zuheng/Research/MixFlow.jl/example/synthetic_tuning/deliverables/scriptName=tv.nf___dryRun=false___n_sample=64___nrunThreads=1/output/summary.csv", DataFrame)

# # delect rows with flowtype == "rwmh"
# df_no_rwmh = df[df.kernel .!= "MF.RWMH", :]
# hcat(df_no_rwmh, df_rwmh)

# names(df_no_rwmh)
# names(df_rwmh)
# ddf = vcat(df_no_rwmh, df_rwmh, cols=:union)
# #turn missing values into 1


# tv_plot_compare_flowtype(df)

# ds = tv_plot_compare_flowtype(df)
# dsg = groupby(ds, [:step_size])
# dsc = combine(dsg, :tv => median)
# # find the :step_size with the largest median tv
# sort!(dsc, :tv_median)

function best_step_size(df::DataFrame, t, k)
    ds = _subset_expt(df, Dict(:target => t, :kernel => k))
    Tm = maximum(unique(ds.Ts))
    dsm = _subset_expt(ds, Dict(:Ts => Tm))
    dsg = groupby(dsm, [:step_size])
    dsc = combine(dsg, :tv => median)
    # find the :step_size with the largest median tv
    sort!(dsc, :tv_median)
    s = dsc[1, :step_size]
    return s, Tm
end
