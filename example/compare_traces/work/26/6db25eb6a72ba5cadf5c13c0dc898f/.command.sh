#!/usr/bin/env julia --threads=5 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/compare_traces/traces.jl")

# get configurations
seed = 2
name = "Cross"
tracetype = "bwd_irf"
kernel_type = MF.HMC

# run simulation
df = run_traces(seed, name, kernel_type, tracetype)

# store output
mkdir("seed=2___tracetype=bwd_irf___target=Cross___kernel=MF.HMC")
CSV.write("seed=2___tracetype=bwd_irf___target=Cross___kernel=MF.HMC/summary.csv", df)
