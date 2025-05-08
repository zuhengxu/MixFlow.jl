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
        joinpath(@__DIR__, "synthetic_expt/reference/result/$(name)_mfvi.jld2"),
    )
    prob = vi_res["prob"]
    dims = LogDensityProblems.dimension(prob)
    return prob, dims
end

function load_real_prob_with_reference(name)
    ref = JLD2.load(joinpath(@__DIR__, "real_data_expt/reference/result/$(name)_mfvi.jld2"))
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


name = "Banana"
prob, dims = load_prob_with_ref(name)

kernel = MF.RWMH
ϵ = 0.25
T = 1000
K = kernel(ϵ, ones(dims))

stability_eval(1, name, kernel(ϵ, ones(dims)); Ts = vcat([10, 20, 50], 100:100:1200))

# look error and rejection rate
mixer = RandomShift(dims, T)
err, rejsfwd, rejsinv = check_error(prob, K, mixer, T)
# plot(rejsfwd, label = "forward rejection")
# plot!(rejsinv, label = "inverse rejection")

# flowtype = MF.BackwardIRFMixFlow
# flowtype = MF.IRFMixFlow
# flowtype = MF.DeterministicMixFlow
flowtype = MF.EnsembleIRFFlow

df = flow_evaluation(2, name, flowtype, kernel, T, ϵ; nsample = 256, nchains = 30)

df = run_tv_sweep(2, name, flowtype, kernel, T, ϵ; nsample = 64, nchains = 30)





# Es = [0.01, 0.03, 0.1, 1, 0.1, 0.03, 0.01]
# kernel = HMCmultiple(Es, dims)

# T = 100
# flow = MF.BackwardIRFMixFlow(T)
# mixer = RandomShift(dims, T)

# flow = MF.DeterministicMixFlow(T)
# mixer = ErgodicShift(dims, T)

# T = 3000
# nchains = 200
# mixer = EnsembleRandomShift(dims, T, nchains)
# flow = EnsembleIRFFlow(T, nchains)

# kernel = MF.RWMH(0.5, ones(dims))


