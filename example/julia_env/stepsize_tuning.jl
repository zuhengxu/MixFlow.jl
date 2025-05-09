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

