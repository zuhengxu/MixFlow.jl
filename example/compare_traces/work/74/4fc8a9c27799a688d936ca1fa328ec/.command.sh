#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/compare_traces/traces.jl")

# get configurations
seed = 1
name = "WarpedGaussian"
tracetype = "bwd_irf"
kernel_type = MF.RWMH

# run simulation
df = run_traces(seed, name, kernel_type, tracetype)

# store output
mkdir("seed=1___tracetype=bwd_irf___target=WarpedGaussian___kernel=MF.RWMH")
CSV.write("seed=1___tracetype=bwd_irf___target=WarpedGaussian___kernel=MF.RWMH/summary.csv", df)
