using Random, Distributions
using LinearAlgebra
using Base.Threads: @threads
using JLD2
using LogExpFunctions
using LogDensityProblems, LogDensityProblemsAD
using Bijectors
using DataFrames, CSV
using MCMCChains
using StatsPlots, Plots

using MixFlow 
using MixFlow: _rand_joint_reference, _log_density_ratio

const MF = MixFlow

include(joinpath(@__DIR__, "../utils.jl"))
include(joinpath(@__DIR__, "../plotting.jl"))

function chain_from_combine_csvs( 
    combined_csvs_folder::String,
    target::String,
    kernel_str::String, 
    trace_type::String;
    compute_ess = false,
)
    df = CSV.read(joinpath(combined_csvs_folder, "summary.csv"), DataFrame)

    selector = Dict(
        :target => target, 
        :kernel => kernel_str,
        :tracetype => trace_type,
    )
    # groupby then iter over groupby and put in 3way array
    ds = _subset_expt(df, selector)
    dgs = @pipe ds |> 
            select(_, [:iter, :d1, :d2, :dr, :seed]) |>
            groupby(_, [:seed])

    Cs = zeros(size(dgs[1], 1), 3, length(dgs))
    for (i, d) in enumerate(dgs)
        Cs[:, :, i] .= Array(d[:, 2:end-1])
    end
    # xs = [1:size(dds, 1) ;]
    chn = Chains(Cs, [:d1, :d2, :dr])

    if compute_ess
        # Compute ESS for each chain, each col is [:d1, :d2, :dr, :min_d]
        Es = zeros(size(chn, 3), size(chn, 2)+1)
        for i in 1:size(chn, 3)
            local E = ess(chn[:, :, i]; relative= true, autocov_method = FFTAutocovMethod(), maxlag = typemax(Int))
            local e_vec = E[:, 2]
            Es[i, 1:end-1] .= e_vec
            Es[i, end] = minimum(e_vec[1:2])
        end

        println("chn size: ", size(chn))

        # put Es in a dataframe
        df_ess = DataFrame(
            :seed => [1:size(chn, 3) ;],
            :d1 => Es[:, 1],
            :d2 => Es[:, 2],
            :dr => Es[:, 3],
            :min_d => Es[:, end]
        )
    else
        df_ess = nothing
    end
    return Chains(Cs, [:d1, :d2, :dr]), df_ess
end


# function get_ess_from_combined_csv(
#     combined_csvs_folder::String,
#     seed,
#     target::String,
#     kernel_str::String, 
#     trace_type::String,
# )

#     df = CSV.read(joinpath(combined_csvs_folder, "summary.csv"), DataFrame)

#     selector = Dict(
#         :target => target, 
#         :kernel => kernel_str,
#         :tracetype => trace_type,
#         :seed => seed,
#     )
#     # groupby then iter over groupby and put in 3way array
#     ds = _subset_expt(df, selector)
#     dtmp = select(ds, [:iter, :d1, :d2, :dr])
#     
#     Cs = zeros(size(dtmp, 1), 3, 1)

#     # just to make sure iters are sorted
#     sort!(dtmp, :iter)
#     Cs[:, :, 1] .= Array(dtmp[:, 2:end])
#     chn = Chains(Cs, [:d1, :d2, :dr])
#     
#     ess_vec = zeros(4)
#     E = ess(chn; relative= true, autocov_method = FFTAutocovMethod(), maxlag = typemax(Int))
#     e_vec = E[:, 2]
#     ess_vec[1:end-1] .= e_vec
#     ess_vec[end] = minimum(e_vec[1:2])

#     ess_df = DataFrame(
#         # :tracetype => trace_type,
#         # :target => target, 
#         # :kernel => kernel_str,
#         :d1 => ess_vec[1],
#         :d2 => ess_vec[2],
#         :dr => ess_vec[3],
#         :min_d => ess_vec[end]
#     )
#     return ess_df
# end

function _get_ess(
    vec_d1, vec_d2, vec_dr
)
    d1_ess = ess(vec_d1; relative= true, autocov_method = FFTAutocovMethod(), maxlag = typemax(Int))
    d2_ess = ess(vec_d2; relative= true, autocov_method = FFTAutocovMethod(), maxlag = typemax(Int))
    dr_ess = ess(vec_dr; relative= true, autocov_method = FFTAutocovMethod(), maxlag = typemax(Int))
    mind_ess = minimum([d1_ess, d2_ess, dr_ess])
    return (
        d1 = d1_ess,
        d2 = d2_ess,
        dr = dr_ess,
        min_d = mind_ess,
    )
end


function get_ess_from_combined_csv(
    combined_csvs_folder::String,
)
    df = CSV.read(joinpath(combined_csvs_folder, "summary.csv"), DataFrame)
    df_gdf = groupby( df,
        [:target, :kernel, :tracetype, :seed],
    )
    df_e = combine(df_gdf, 
            [:d1, :d2, :dr] => 
                   ((d1, d2, dr) -> _get_ess(d1, d2, dr)) => 
            AsTable)

    return df_e
end

function grouped_ess_plot(
    combined_csvs_folder::String;
    dpi = 600,
    size = (1000, 800),
    margin = 10Plots.mm,
    kargs...
)

    df_e = get_ess_from_combined_csv(combined_csvs_folder)
    df_e = _remove_nan(df_e)

    tagets = unique(df_e.target)
    for t in tagets
        selector = Dict(
            :target => t, 
        )
        df_e_sub = _subset_expt(df_e, selector)
        df_e_sub = _remove_nan(df_e_sub)

        include_legend = t == "Banana" ? true : false

        # density ratio ess plot
        fig_name = "$(t)_density_ratio_ess"
        fg = @df df_e_sub groupedboxplot(:kernel, :dr, group = :tracetype, yscale = :log10, markerstrokewidth = 0.5, fillalpha = 0.8)

        plot!(fg, ylabel = "ESS per iter", title = "$t density ratio ESS", legend = include_legend)
        plot!(fg; dpi = dpi, size = size, margin = margin, kargs...)
        savefig(fg, fig_name * ".png")
        

        # min ess plot
        fig_name = "$(t)_min_ess"
        fg = @df df_e_sub groupedboxplot(:kernel, :min_d, group = :tracetype, yscale = :log10, markerstrokewidth = 0.5, fillalpha = 0.8)

        plot!(fg, ylabel = "ESS per iter", title = "$t min ESS", legend = include_legend)
        plot!(fg; dpi = dpi, size = size, margin = margin, kargs...)
        savefig(fg, fig_name * ".png")
    end
end

# turn "MF.HMC" to "HMC"
function _get_kernel_name(str::String)
    name = split(str, ".", limit = 2)[2]
    return name
end

function trace_meanplot(
    combined_csvs_folder::String,
    target::String,
    kernel_str::String;
    plot_kwargs...
)
    df = CSV.read(joinpath(combined_csvs_folder, "summary.csv"), DataFrame)
    trace_types = unique(df.tracetype)

    for t in trace_types
        fg_name = "$(target)_$(_get_kernel_name(kernel_str))_$(t)"
        chn, _ = chain_from_combine_csvs(
            combined_csvs_folder,
            target,
            kernel_str,
            t;
            compute_ess = false,
        )
        fg = meanplot(chn)
        plot!(
            fg,
            plot_title = fg_name; 
            plot_kwargs...,
        )
        savefig(fg, fg_name * ".png")
    end 
end

# trace_meanplot(
#     ".",
#     "Cross", 
#     "MF.RWMH";
#     dpi = 1000,
#     margin = 5Plots.mm,
#     xguidefontsize = 18,
#     yguidefontsize = 18,
# ) 



# chn, _ = chain_from_combine_csvs(
#     ".",
#     "Cross",
#     "MF.RWMH",
#     "fwd_homo"; 
#     compute_ess = false,
# )
# fg = plot(chn)
# savefig(fg, "../traceplot/Cross_homo_trace.png")

# fg1 = meanplot(chn)
# plot!(fg1, dpi = 1000, margin = 5Plots.mm, xguidefontsize = 18, yguidefontsize = 18, plot_title = "Banana MALA bwd_inv_IRF", legend = false)
