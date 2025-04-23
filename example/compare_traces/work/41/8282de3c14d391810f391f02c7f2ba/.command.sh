#!/usr/bin/env julia --threads=5 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/compare_traces/traces.jl")

# get configurations
seed = 1
name = "Funnel"
tracetype = "bwd_inv_irf"
kernel_type = MF.HMC

# run simulation
df = run_traces(seed, name, kernel_type, tracetype)

# store output
mkdir("seed=1___tracetype=bwd_inv_irf___target=Funnel___kernel=MF.HMC")
CSV.write("seed=1___tracetype=bwd_inv_irf___target=Funnel___kernel=MF.HMC/summary.csv", df)
