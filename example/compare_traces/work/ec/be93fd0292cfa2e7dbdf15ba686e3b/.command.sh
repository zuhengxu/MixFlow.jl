#!/usr/bin/env julia --threads=5 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/compare_traces/traces.jl")

# get configurations
seed = 8
name = "Banana"
tracetype = "mcmc"
kernel_type = MF.HMC

# run simulation
df = run_traces(seed, name, kernel_type, tracetype)

# store output
mkdir("seed=8___tracetype=mcmc___target=Banana___kernel=MF.HMC")
CSV.write("seed=8___tracetype=mcmc___target=Banana___kernel=MF.HMC/summary.csv", df)
