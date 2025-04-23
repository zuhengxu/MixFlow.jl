#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/compare_traces/traces.jl")

# get configurations
seed = 24
name = "Cross"
tracetype = "mcmc"
kernel_type = MF.HMC

# run simulation
df = run_traces(seed, name, kernel_type, tracetype)

# store output
mkdir("seed=24___tracetype=mcmc___target=Cross___kernel=MF.HMC")
CSV.write("seed=24___tracetype=mcmc___target=Cross___kernel=MF.HMC/summary.csv", df)
