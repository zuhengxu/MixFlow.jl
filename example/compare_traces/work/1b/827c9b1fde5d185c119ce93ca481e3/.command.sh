#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/compare_traces/traces.jl")

# get configurations
seed = 12
name = "Cross"
tracetype = "inv_irf"
kernel_type = MF.MALA

# run simulation
df = run_traces(seed, name, kernel_type, tracetype)

# store output
mkdir("seed=12___tracetype=inv_irf___target=Cross___kernel=MF.MALA")
CSV.write("seed=12___tracetype=inv_irf___target=Cross___kernel=MF.MALA/summary.csv", df)
