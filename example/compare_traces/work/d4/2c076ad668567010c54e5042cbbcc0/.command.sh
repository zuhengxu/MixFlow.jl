#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/compare_traces/traces.jl")

# get configurations
seed = 15
name = "Banana"
tracetype = "bwd_inv_irf"
kernel_type = MF.RWMH

# run simulation
df = run_traces(seed, name, kernel_type, tracetype)

# store output
mkdir("seed=15___tracetype=bwd_inv_irf___target=Banana___kernel=MF.RWMH")
CSV.write("seed=15___tracetype=bwd_inv_irf___target=Banana___kernel=MF.RWMH/summary.csv", df)
