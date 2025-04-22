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
include(joinpath(@__DIR__, "../plotting.jl"))


function run_tv(
    seed, name::String, flowtype, T::Int, kernel_type, step_size; 
    nsample = 512, leapfrog_steps=50,
) 
    flow = flowtype(T)

    Random.seed!(seed)

    vi_res = JLD2.load(
        joinpath(@__DIR__, "../syn_mfvi_fit/result/$(name)_mfvi.jld2"),
    )
    prob = vi_res["prob"]

    dims = LogDensityProblems.dimension(prob)
    mixer = ErgodicShift(dims, T)
        
    if kernel_type == MF.HMC
        kernel = MF.HMC(leapfrog_steps, step_size)
    elseif kernel_type == MF.uncorrectHMC
        kernel = MF.uncorrectHMC(leapfrog_steps, step_size)
    else 
        kernel = kernel_type(step_size, ones(dims))
    end

    xsπ = rand(prob.target.ℓ, nsample) 
    vsπ = reduce(hcat, [MF._rand_v_given_x(kernel, prob, xsπ[:, i]) for i in 1:size(xsπ, 2)])
    uvπ = rand(dims, nsample)
    uaπ = rand(nsample)
    
    lrs = zeros(T+1, nsample)
    @showprogress @threads for i in 1:nsample
        x = xsπ[:, i]
        v = vsπ[:, i]
        uv = kernel_type == MF.uncorrectHMC ? nothing : uvπ[:, i]
        ua = kernel_type == MF.uncorrectHMC ? nothing : uaπ[i]
        lrs[:, i] .= MF.log_density_ratio_flow_sweep(flow, prob, kernel, mixer, x, v, uv, ua)
    end
    
    tvs = mean(abs.(expm1.(lrs)), dims = 2) ./ 2

    df = DataFrame(
        tv = vec(tvs),
        Ts = [1:T+1 ;],
        nparticles = nsample,
    ) 
    return df
end

# df = run_tv(1, "Banana", MF.DeterministicMixFlow, 10, MF.RWMH, 0.1; nsample = 16)
# df = run_tv(1, "Funnel", MF.DeterministicMixFlow, 30, MF.HMC, 0.1; nsample = 512)


function tv_plot(
    combined_csvs_folder::String
)
    df = CSV.read(joinpath(combined_csvs_folder, "summary.csv"), DataFrame) 

    targets = unique(df.target)
    kernels = unique(df.kernel)
    f = unique(df.flowtype)[1] # we only run the homoflow

    for (t, k) in Iterators.product(targets, kernels)
        println("target: $t, kernel: $k, flowtype: $f")
        fig_name = "$(t)__$(_throw_dot(f))__$(_throw_dot(k))"
        local selector = Dict(
            :target => t,
            :kernel => k,
        )
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

        plot!(fg, ylabel = "Total Variation", xlabel = "flow length")
        plot!(fg, dpi = 600, size = (500, 400), margin = 10Plots.mm)
        savefig(fg, fig_name * ".png")
    end
end
