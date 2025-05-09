using AdaptiveAIS
using Bijectors
using DataFrames, CSV
using JLD2

include(joinpath(@__DIR__, "../../julia_env/Model.jl"))


# some util functions for adative ais
AdaptiveAIS.iid_sample(rngs, p0::Bijectors.MultivariateTransformed, N) = rand(p0, N)
AdaptiveAIS.iid_sample(rng, p0::Bijectors.MultivariateTransformed) = rand(p0)

# AdaptiveAIS.iid_sample(rng, dist::ContinuousDistribution, n::Int) = rand(dist, n)
# AdaptiveAIS.iid_sample(rng, dist::ContinuousDistribution) = rand(dist)

function run_sais(
    seed, name, nptls, nrounds; 
    tk = RWMH_sweep([10^p for p in -5:0.2:1], I, 1), stepsize = 0.005, save_jld = false,
)
    Random.seed!(seed)

    p1, dims, ad = load_model(name)
    # load mfvi reference
    p0 = JLD2.load(joinpath(@__DIR__, "../reference/result/$(name)_mfvi.jld2"))["reference"]
    # p0 = MvNormal(zeros(dims), I)
    L = LinearPath()

    ais_prob = AISProblem(p0, p1, L)

    MD = MirrorDescent(stepsize = stepsize, max_Δ = 0.1, max_T = Inf)
    a_md = ais(ais_prob, MD; N = nptls, transition_kernel = tk, show_report = true)

    S = SAIS(FixedSchedule(a_md.schedule), nrounds)
    a_res = ais(ais_prob, S; N = nptls, transition_kernel = tk, show_report = true)

    if save_jld
        res_dir = joinpath(@__DIR__, "result/")
        if !isdir(res_dir)
            mkpath(res_dir)
        end

        JLD2.save(
            joinpath(res_dir, "$(name)_sais_$(seed).jld2"),
            "output", a_res,
            "prob", ais_prob,
        )
    end

    df = DataFrame(
        target = name,
        nrounds = nrounds,
        sched_length = length(a_res.schedule),
        N = AdaptiveAIS.n_particles(a_res.particles),
        time = a_res.full_timing.time,
        allc = a_res.timing.bytes,
        ess = AdaptiveAIS.ess(a_res.particles),
        elbo = a_res.particles.elbo,
        GCB = isnothing(a_res.barriers) ? "NA" : a_res.barriers.globalbarrier,
        log_norm_constant = a_res.particles.log_normalization,
        norm_constant = exp(a_res.particles.log_normalization),
    )
    return df
end



# nptls = 2^11
# nrounds = 5

# for name in ["TReg", "SparseRegression", "Brownian", "Sonar", "LGCP"]
#     df = run_sais(1, name, nptls, nrounds; save_jld = true)
#     CSV.write(
#         joinpath(@__DIR__, "result", "$(name)_sais.csv"),
#         df,
#         writeheader = true,
#     )
# end
