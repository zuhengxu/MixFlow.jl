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

function run_tv_sweep(
    seed, name::String, flowtype, kernel_type, flow_length::Int, step_size; 
    nsample::Int = 1024, leapfrog_steps::Int=10, nchains::Int = 10 
)
    Random.seed!(seed)

    prob, dims = load_synthetic_prob(name)
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
        Ts = (flow isa MF.EnsembleIRFFlow) ? flow_length : [1:T+1 ;],
    ) 
    return df
end


function run_elbo(
    seed, name::String, flowtype, kernel_type, T::Int, step_size; 
    nsample = 1024, leapfrog_steps=50, nchains = 10,
)

    Random.seed!(seed)

    prob, dims = load_synthetic_prob(name)
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

function load_synthetic_prob(name)
    vi_res = JLD2.load(
        joinpath(@__DIR__, "../syn_mfvi_fit/result/$(name)_mfvi.jld2"),
    )
    prob = vi_res["prob"]
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


name = "Banana"
vi_res = JLD2.load(
    joinpath(@__DIR__, "../syn_mfvi_fit/result/$(name)_mfvi.jld2"),
)
prob = vi_res["prob"]

dims = LogDensityProblems.dimension(prob)

kernel = MF.RWMH
ϵ = 0.25
T = 10000

# look error and rejection rate
err, rejsfwd, rejsinv = check_error(prob, MF.RWMH(ϵ, ones(dims)), T)
plot(rejsfwd, label = "forward rejection")
plot!(rejsinv, label = "inverse rejection")

# flowtype = MF.BackwardIRFMixFlow
# flowtype = MF.IRFMixFlow
# flowtype = MF.DeterministicMixFlow
flowtype = MF.EnsembleIRFFlow

df = run_elbo(2, name, flowtype, T, kernel, ϵ; nsample = 256, nchains = 30)

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


