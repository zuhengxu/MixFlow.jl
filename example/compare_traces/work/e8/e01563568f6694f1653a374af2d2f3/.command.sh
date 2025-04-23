#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/compare_traces/traces.jl")

# get configurations
seed = 4
name = "Funnel"
tracetype = "fwd_irf"
kernel_type = MF.HMC

# run simulation
df = run_traces(seed, name, kernel_type, tracetype)

# store output
mkdir("seed=4___tracetype=fwd_irf___target=Funnel___kernel=MF.HMC")
CSV.write("seed=4___tracetype=fwd_irf___target=Funnel___kernel=MF.HMC/summary.csv", df)
