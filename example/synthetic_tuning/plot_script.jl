include("plotting.jl")

# existing ensemble plot
# df = CSV.read(joinpath(@__DIR__, "deliverables/tv_ensemble.csv"), DataFrame) 
# ensemble_flowlength_plot(
#     df, :tv; 
#     create_fig_dir=true, fig_dir="figure/flowlength/", 
#     ylabel="Total Variation", xlabel="flow length",
#     ylims=(0, 1),
#     margin = 5Plots.mm
# )

# ensemble_nchains_plot(
#     df, :tv; 
#     create_fig_dir=true, fig_dir="figure/nchains/", 
#     ylabel="Total Variation", xlabel="Number of Ensembles",
#     ylims=(0, 1),
#     margin = 5Plots.mm
# )



df = CSV.read("deliverables/tv_mixflow.csv", DataFrame)
df_rwmh = CSV.read("deliverables/rwmh_tv/output/summary.csv", DataFrame)
df_ensemble = CSV.read(joinpath(@__DIR__, "deliverables/tv_ensemble.csv"), DataFrame) 

k = "MF.RWMH"
unique(df_rwmh, :flowtype)

unique(df_ensemble.target)
nrows(df_ensemble)


# # delect rows with flowtype == "rwmh"
df_no_rwmh = df[df.kernel .!= "MF.RWMH", :]

names(df_no_rwmh)
names(df_rwmh)
ddf = vcat(df_no_rwmh, df_rwmh, cols=:union)
# for the rows with nensembles == missing, impute it with 1
ddf.nensembles[ismissing.(ddf.nensembles)] .= 1

CSV.write("deliverables/tv_mixflow_all.csv", ddf)
CSV.write("deliverables/tv_ensemble_all.csv", df_ensemble)

ds = _subset_expt(ddf, Dict(:kernel => "MF.MALA", :target => "Banana"))
unique(ds, :step_size)


# ds = _subset_expt(df, Dict(:kernel => "MF.MALA", :target => "Banana"))

# tv_plot_compare_flowtype(df)


# ds = tv_plot_compare_flowtype(df)
# dsg = groupby(ds, [:step_size])
# dsc = combine(dsg, :tv => median)
# # find the :step_size with the largest median tv
# sort!(dsc, :tv_median)


# combine tv csvs
df_tv_mf = CSV.read("deliverables/tv_mixflow_all.csv", DataFrame)
df_tv_ensemble = CSV.read("deliverables/tv_ensemble_all.csv", DataFrame)

# # change the column name :nensemble to :nensembles
# rename!(df_tv_ensemble, :nensemble => :nensembles)
# unique(df_tv_ensemble, :kernel)
# unique(df_tv_ensemble.nchains)
#
# df_tv_mf.flowtype .= _throw_dot.(df_tv_mf.flowtype)
# df_tv_mf.kernel .= _throw_dot.(df_tv_mf.kernel)

# CSV.write("deliverables/tv_mixflow_all.csv", df_tv_mf)


# df_tv_ensemble.kernel .= _throw_dot.(df_tv_ensemble.kernel)
# CSV.write("deliverables/tv_ensemble_all.csv", df_tv_ensemble)

# df_tv_all = vcat(df_tv_mf, df_tv_ensemble, cols=:union)
# CSV.write("deliverables/tv_all.csv", df_tv_all)


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
                lw = 2,
            )
            plot!(fg; kwargs...)
            plot!(fg, dpi = 600, size = (500, 400))
            savefig(fg, joinpath(fig_dir, "$(t)__$(_throw_dot(k))__nchain=$(nc_mx).png"))
        catch e
            println("Error: $e")
            continue
        end
    end
end

function tv_plot_compare_all_flow(df::DataFrame, df_en::DataFrame)

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


            local nc_mx = maximum(unique(ds_en.nchains))
            s_en = k == "HMC" ? 0.1 : 1.0
            local ds_en = _subset_expt(ds_en, Dict(:nchains => nc_mx, :step_size => s_en))
            println("nchains: $nc_mx, step_size: $s_en")
            add_groupederrorline!(fg, ds_en, :flow_length, :tv, :seed, :flowtype;
                errorstyle = :ribbon,
                linestyle = :solid,
                lw = 2,
                label = "EnsembleIRF",
            )
            # return ds_en

            
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

tv_plot_compare_all_flow(df_tv_mf, df_tv_ensemble)

# ds_en = tv_plot_compare_all_flow(df_tv_mf, df_tv_ensemble)
# unique(ds_en, :step_size)

# ds = _subset_expt(df_tv_ensemble, Dict(:kernel => "HMC", :target => "Banana", :flowtype => "EnsembleIRFFlow"))
# unique(ds, :step_size)
