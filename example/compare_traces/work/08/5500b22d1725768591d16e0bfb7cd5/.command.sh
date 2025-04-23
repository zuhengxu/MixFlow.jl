#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/compare_traces/traces.jl")

# get configurations
seed = 13
name = "WarpedGaussian"
tracetype = "fwd_homo"
kernel_type = MF.HMC

# run simulation
df = run_traces(seed, name, kernel_type, tracetype)

# store output
mkdir("seed=13___tracetype=fwd_homo___target=WarpedGaussian___kernel=MF.HMC")
CSV.write("seed=13___tracetype=fwd_homo___target=WarpedGaussian___kernel=MF.HMC/summary.csv", df)
