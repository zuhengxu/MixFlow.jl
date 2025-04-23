#!/usr/bin/env julia --threads=5 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/compare_traces/traces.jl")

# get configurations
seed = 6
name = "Banana"
tracetype = "bwd_irf"
kernel_type = MF.MALA

# run simulation
df = run_traces(seed, name, kernel_type, tracetype)

# store output
mkdir("seed=6___tracetype=bwd_irf___target=Banana___kernel=MF.MALA")
CSV.write("seed=6___tracetype=bwd_irf___target=Banana___kernel=MF.MALA/summary.csv", df)
