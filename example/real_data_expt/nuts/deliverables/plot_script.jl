include(joinpath(@__DIR__, "../../julia_env/", "Model.jl"))
include(joinpath(@__DIR__, "../../julia_env/", "utils.jl"))
include(joinpath(@__DIR__, "../../julia_env/", "plotting.jl"))


df_rwmh = CSV.read("deliverables/rwmh_res/output/summary.csv", DataFrame)
# rename DeterministicMixFlow to HomogeneousMixFlow
df_rwmh.method .= replace(df_rwmh.method, "DeterministicMixFlow" => "HomogeneousMixFlow")
df_nut = CSV.read("deliverables/nuts.csv", DataFrame)

# for  t in real_data_list
for t in real_data_list
    try
        t = String(t)
        ds = _subset_expt(df_rwmh, Dict(:target => t))
        ds_nuts = _subset_expt(df_nut, Dict(:target => t))
        ds_combine = vcat(ds_nuts, ds, cols=:union)
        sort!(ds_combine, [:method], rev=true)

        fg = @df ds_combine boxplot(
            :method, :max_abs_err_mean,
            colorgroup=:method,
            markerstrokewidth=0.5, fillalpha=0.8;
            markersize=3,
            # title = "$t ",
            ylabel="maximal coord abs. error",
            xtickfontsize=10, ytickfontsize=12, yguidefontsize=15,
            legendfontsize=11, titlefontsize=18,
            xrotation=-10,
            label=""
        )
        plot!(fg, legend=:best)
        plot!(fg, dpi=200, size=(600, 400), margin=5Plots.mm)
        savefig(fg, "figure/$(t)_mean_error.png")

        fg = @df ds_combine boxplot(
            :method, :max_abs_err_std,
            colorgroup=:method,
            markerstrokewidth=0.5, fillalpha=0.8;
            markersize=3,
            # title = "$t ",
            ylabel="maximal coord abs. error",
            xtickfontsize=10, ytickfontsize=12, yguidefontsize=15,
            legendfontsize=11, titlefontsize=18,
            xrotation=-10,
            label=""
        )
        plot!(fg, legend=:best)
        plot!(fg, dpi=200, size=(600, 400), margin=5Plots.mm)
        savefig(fg, "figure/$(t)_std_error.png")
    catch e
        println("Error in $t: ", e)
        continue
    end
end
