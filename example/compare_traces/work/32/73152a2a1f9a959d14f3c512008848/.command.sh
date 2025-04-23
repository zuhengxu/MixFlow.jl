#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/compare_traces/traces.jl")

# get configurations
seed = 20
name = "Banana"
tracetype = "fwd_irf"
kernel_type = MF.HMC

# run simulation
df = run_traces(seed, name, kernel_type, tracetype)

# store output
mkdir("seed=20___tracetype=fwd_irf___target=Banana___kernel=MF.HMC")
CSV.write("seed=20___tracetype=fwd_irf___target=Banana___kernel=MF.HMC/summary.csv", df)
