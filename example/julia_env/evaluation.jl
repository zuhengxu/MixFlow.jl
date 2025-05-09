using ProgressMeter
using Random, Distributions
using LinearAlgebra
using Base.Threads: @threads
using LogExpFunctions
using LogDensityProblems, LogDensityProblemsAD
using ADTypes, Mooncake
using DataFrames, CSV
using JLD2

using MixFlow 
const MF = MixFlow

include(joinpath(@__DIR__, "mfvi.jl"))
include(joinpath(@__DIR__, "Model.jl"))

function run_tv_sweep(
    seed, name::String, flowtype, kernel_type, flow_length::Int, step_size; 
    nsample::Int = 1024, leapfrog_steps::Int=10, nchains::Int = 10 
)
    Random.seed!(seed)

    prob, dims = load_synthetic_prob_with_reference(name)
    flow, mixer, nchains = setup_flow(seed, flowtype, flow_length, nchains, dims)
    kernel = setup_kernel(kernel_type, step_size, leapfrog_steps, dims)


    ldr_sweep = flow isa MF.EnsembleIRFFlow ? MF.log_density_ratio_flow_sweep_ensemble : MF.log_density_ratio_flow_sweep

    if (kernel_type == MF.uncorrectHMC) && !(flow isa MF.DeterministicMixFlow)
        println("no this combo")
        return nothing
    end

    @info "TV eval using $nsample samples on Ensemble Flow length: $flow_length, nchains: $nchains"
    
    # generate samples from the target
    xsπ = rand(prob.target.ℓ, nsample) 
    vsπ = reduce(hcat, [MF._rand_v_given_x(kernel, prob, xsπ[:, i]) for i in 1:size(xsπ, 2)])
    uvπ = rand(dims, nsample)
    uaπ = rand(nsample)
    
    nrows = flow isa MF.EnsembleIRFFlow ? nchains : flow_length + 1
    lrs = zeros(nrows, nsample)

    @showprogress @threads for i in 1:nsample
        x = xsπ[:, i]
        v = vsπ[:, i]
        uv = kernel isa MF.uncorrectHMC ? nothing : uvπ[:, i]
        ua = kernel isa MF.uncorrectHMC ? nothing : uaπ[i]
        lrs[:, i] .= ldr_sweep(flow, prob, kernel, mixer, x, v, uv, ua)
    end
    
    tvs = mean(abs.(expm1.(lrs)), dims = 2) ./ 2

    df = DataFrame(
        nensembles = (flow isa MF.EnsembleIRFFlow) ? [1:nchains ;] : 1,
        tv = vec(tvs),
        nparticles = nsample,
        Ts = (flow isa MF.EnsembleIRFFlow) ? flow_length : [1:flow_length+1 ;],
    ) 
    return df
end


function flow_evaluation(
    seed, name::String, flowtype, kernel_type, T::Int, step_size; 
    nsample = 1024, leapfrog_steps=50, nchains = 10, 
)

    Random.seed!(seed)

    prob, dims = load_prob_with_ref(name)

    flow, mixer, nchains = setup_flow(seed, flowtype, T, nchains, dims)
    kernel = setup_kernel(kernel_type, step_size, leapfrog_steps, dims)
    

    if (kernel_type == MF.uncorrectHMC) && !(flow isa MF.DeterministicMixFlow)
        println("no this combo")
        return DataFrame( nensembles = NaN, logZ = NaN, elbo = NaN, ess = NaN, nparticles = NaN) 
    end

    output = MF.mixflow(flow, prob, kernel, mixer, nsample)

    df = DataFrame(
        cost = is_tracked(prob.target) ? compute_cost(prob.target) : NaN,
        nensembles = nchains, 
        logZ = output.logZ,
        elbo = output.elbo,
        ess = output.ess/nsample,  # ess per sample
        nparticles = nsample,
    )
    return df
end

function load_prob_with_ref(name)
    if name ∈ synthetic_list
        return load_synthetic_prob_with_reference(name)
    elseif name ∈ real_data_list 
        return load_real_prob_with_reference(name)
    else
        throw(ArgumentError("Unknown problem name: $name or mfvi reference does not exist yet"))
    end
end


function load_synthetic_prob_with_reference(name)
    vi_res = JLD2.load(
        joinpath(@__DIR__, "../synthetic_expt/reference/result/$(name)_mfvi.jld2"),
    )
    prob = vi_res["prob"]
    dims = LogDensityProblems.dimension(prob)
    return prob, dims
end

function load_real_prob_with_reference(name)
    ref = JLD2.load(joinpath(@__DIR__, "../real_data_expt/reference/result/$(name)_mfvi.jld2"))
    prob = ref["prob"]
    dims = LogDensityProblems.dimension(prob)
    return prob, dims
end


function setup_flow(seed, flowtype, T, nchains, dims)
    flow = (flowtype <: MF.EnsembleIRFFlow) ? flowtype(T, nchains) : flowtype(T)

    if flow isa MF.EnsembleIRFFlow
        mixer = EnsembleRandomShift(dims, T, nchains) 
    elseif flow isa MF.DeterministicMixFlow
        mixer = ErgodicShift(dims, T)
        nchains = 1
    else
        mixer = RandomShift(dims, T)
        nchains = 1
    end

    return flow, mixer, nchains
end

function setup_kernel(kernel_type, step_size, leapfrog_steps, dims)
    if kernel_type == MF.HMC
        kernel = MF.HMC(leapfrog_steps, step_size)
    elseif kernel_type == MF.uncorrectHMC
        kernel = MF.uncorrectHMC(leapfrog_steps, step_size)
    else 
        kernel =  kernel_type(step_size, ones(dims))
    end
end


function check_error(prob, K, mixer, T::Int)
    x0, v0, uv0, ua0 = MF._rand_joint_reference(prob, K)
    x, v, uv, ua = x0, v0, uv0, ua0

    rejs_fwd = []
    for t in 1:T
        x, v, uv, ua, acc = MF.forward(prob, K, mixer, x, v, uv, ua, t)
        if !acc 
            push!(rejs_fwd, t)
        end
    end

    rejs_inv = []
    for t in T:-1:1
        x, v, uv, ua, acc = MF.inverse(prob, K, mixer, x, v, uv, ua, t)
        if !acc
            push!(rejs_inv, t)
        end
    end

    errsq = sum(abs2, x - x0) + sum(abs2, v - v0) + sum(abs2, uv - uv0) + sum(abs2, ua - ua0)
    err = sqrt(errsq)
    return err, rejs_fwd, sort(rejs_inv)
end

function check_error(prob, K, mixer, Ts::AbstractVector{Int})
    stats = []
    for T in Ts
        err, _, _ = check_error(prob, K, mixer, T)
        push!(stats, err)
    end
    return map(identity, stats)
end

function stability_eval(
    seed::Int, name::String, K::MultivariateInvolutiveKernel; 
    type::String="irf", Ts::Vector{Int} = vcat([10, 20, 50], 100:100:1200)
)
    Random.seed!(seed)
    prob, dims = load_prob_with_ref(name)

    T_max = maximum(Ts)

    if type == "homogeneous"
        mixer = ErgodicShift(dims, T_max)
    elseif type == "irf"
        mixer = RandomShift(dims, T_max)
    else
        error("type must be either homogeneous or irf")
    end

    err = check_error(prob, K, mixer, Ts)
    df = DataFrame(
        id = seed,
        Ts = Ts,
        errors = err,
    )
    return df
end


function rejection_rate(prob, K, T)
    dims = LogDensityProblems.dimension(prob)
    mixer = RandomShift(dims, T)
    err, rejsfwd, _ = check_error(prob, K, mixer, T)
    rej_rate = length(rejsfwd) / T
    return rej_rate, err
end


function bijection_search(
    f, l, r;
    target_rej_rate = 0.766,
    max_iter = 20,
    stop_criterion = (x) -> 0.73 < f(x) < 0.79
)
    for i in 1:max_iter
        x = (l + r) / 2
        if stop_criterion(x)
            return x, i
        end

        if f(x) < target_rej_rate
            l = x
        else
            r = x
        end
    end

    println("stop criterion not met")
    return NaN, max_iter
end

function find_stepsize(
    prob, kernel, T; 
    l = 0.0001, r = 10.0, max_iter = 100, target_rej_rate = 0.75,
    thresh = 0.05, T_check_stab = 5000
)
    stop_range_lower = target_rej_rate - thresh
    stop_range_upper = target_rej_rate + thresh

    # find the stepsize that gives the target rejection rate
    # using binary search
    dims = LogDensityProblems.dimension(prob)
    f = ϵ -> rejection_rate(prob, kernel(ϵ, ones(dims)), T)[1]
    stop_criterion = (x) -> stop_range_lower < f(x) < stop_range_upper

    s, neval = bijection_search(
        f, l, r;
        target_rej_rate = target_rej_rate,
        max_iter = max_iter,
        stop_criterion = stop_criterion,
    )
    
    println("done bijection search, stepsize: $s, neval: $neval")
    
    err = rejection_rate(prob, kernel(s, ones(dims)), T_check_stab)[2]
    neval += 1
    stable_cond = (err < 1e-3)
    while !stable_cond
        s /= 2
        err = rejection_rate(prob, kernel(s, ones(dims)), T_check_stab)[2]
        neval += 1
        stable_cond = (err < 1e-3)
    end
    println("done stability check, stepsize: $s, neval: $neval")
    return s, neval
end


# name = "Sonar"
# prob, dims = load_prob_with_ref(name)
# kernel = RWMH
# T_check = 5000
# ϵ, neval = find_stepsize(prob, kernel, 2000; target_rej_rate = 0.2, thresh = 0.02, T_check_stab = T_check)
# K = kernel(ϵ, ones(dims))


# rej_rate, err = rejection_rate(prob, K, T_check)

# flowtype = MF.BackwardIRFMixFlow
# # flowtype = MF.IRFMixFlow
# # flowtype = MF.DeterministicMixFlow
# # flowtype = MF.EnsembleIRFFlow

# df = flow_evaluation(1, name, flowtype, kernel, T_check, ϵ; nsample = 64, nchains = 30)

# T = 3000
# df = run_tv_sweep(2, name, flowtype, kernel, T, ϵ; nsample = 64, nchains = 30)

