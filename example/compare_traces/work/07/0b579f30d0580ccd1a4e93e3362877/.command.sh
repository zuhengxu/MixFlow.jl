#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/compare_traces/traces.jl")

# get configurations
seed = 21
name = "Funnel"
tracetype = "bwd_inv_irf"
kernel_type = MF.MALA

# run simulation
df = run_traces(seed, name, kernel_type, tracetype)

# store output
mkdir("seed=21___tracetype=bwd_inv_irf___target=Funnel___kernel=MF.MALA")
CSV.write("seed=21___tracetype=bwd_inv_irf___target=Funnel___kernel=MF.MALA/summary.csv", df)
