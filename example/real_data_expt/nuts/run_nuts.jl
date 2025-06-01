using AdvancedHMC, ADTypes, DifferentiationInterface, Mooncake, Zygote
using AbstractMCMC

using LogDensityProblems
using LinearAlgebra

using JLD2
using DataFrames, CSV
using Optimisers
using Random

include(joinpath(@__DIR__, "../../julia_env/Model.jl"))
include(joinpath(@__DIR__, "../../julia_env/evaluation.jl"))

function ground_truth_setting(t::String)
    res = JLD2.load(joinpath(@__DIR__, "../rwmh/deliverables/ground_truth_res/$(t)_sais.jld2"))
    res["logZ"], res["Mean"], res["Std"]
end

function run_nuts(seed, name; n_samples::Int = 2_000)

    Random.seed!(seed)
    prob, dims = load_prob_with_ref(name)
    # load preconfigured ad

    ad = load_model(name)[3]
    target_ad = LogDensityProblemsAD.ADgradient(ad, prob.target)

    # 2. Wrap the log density function and specify the AD backend.
    #    This creates a callable struct that computes the log density and its gradient.
    ℓπ = TrackedLogDensityProblem(target_ad)

    model = AdvancedHMC.LogDensityModel(ℓπ)

    # Set the number of samples to draw and warmup iterations
    n_adapts = Int(n_samples / 2)

    # Define a Hamiltonian system
    metric = DiagEuclideanMetric(dims)
    hamiltonian = Hamiltonian(metric, model)

    initial_θ = rand(prob.reference)

    # Define a leapfrog solver, with the initial step size chosen heuristically
    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
    integrator = Leapfrog(initial_ϵ)

    # Define an HMC sampler with the following components
    #   - multinomial sampling scheme,
    #   - generalised No-U-Turn criteria, and
    #   - windowed adaption for step-size and diagonal mass matrix
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    # Run the sampler to draw samples from the specified Gaussian, where
    #   - `samples` will store the samples
    #   - `stats` will store diagnostic statistics for each sample
    samples, stats = sample(
        hamiltonian, kernel, initial_θ, n_samples, adaptor, n_adapts; progress=true
    )

    cost = compute_cost(ℓπ)

    Mean = mean(samples)
    Std = std(samples)

    return cost, Mean, Std
end

# gt_logz, gt_Mean, gt_Std = ground_truth_setting(name)

max_abs_err(v1, v2) = maximum(abs.(v1 .- v2))

function df_nuts(seed, name; n_samples::Int = 10_000)
    cost, Mean, Std = run_nuts(seed, name; n_samples=n_samples)    
    _, gt_Mean, gt_Std = ground_truth_setting(name)
    
    m_er = max_abs_err(Mean, gt_Mean)
    s_er = max_abs_err(Std, gt_Std)

    df = DataFrame(
        method = "NUTS",
        cost = cost,
        mean = Mean,
        std = Std,
        max_abs_err_mean = m_er,
        max_abs_err_std = s_er,
    )
    return df 
end


rwmh_dir = joinpath(@__DIR__, "../rwmh/result/")

function df_rwmh_is(seed, name, flowtype)
    pth = joinpath(rwmh_dir, "rwmh_$(name)_$(flowtype)_$(seed).jld2")
    out = JLD2.load(pth)["output"]
    _, gt_Mean, gt_Std = ground_truth_setting(name)

    Mean = mean(out)
    Std = std(out)
    
    m_er = max_abs_err(Mean, gt_Mean)
    s_er = max_abs_err(Std, gt_Std)
    df = DataFrame(
        method = "$(flowtype)",
        mean = Mean,
        std = Std,
        max_abs_err_mean = m_er,
        max_abs_err_std = s_er,
    )
    return df
end

# name = "Sonar"
# seed = 15
# flowtype = BackwardIRFMixFlow

# df_rwmh_is(seed, name, flowtype)

name = "LGCP"
nsample = 5000

for seed in 1:10
    df = df_nuts(seed, name; n_samples=5000)
    CSV.write(
        joinpath(@__DIR__, "results/nuts_$(name)_$(seed).csv"),
        df,
        writeheader=true,
        append=false,
    )
end


