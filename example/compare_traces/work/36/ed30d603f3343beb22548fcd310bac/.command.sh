#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/compare_traces/traces.jl")

# get configurations
seed = 28
name = "Banana"
tracetype = "fwd_homo"
kernel_type = MF.MALA

# run simulation
df = run_traces(seed, name, kernel_type, tracetype)

# store output
mkdir("seed=28___tracetype=fwd_homo___target=Banana___kernel=MF.MALA")
CSV.write("seed=28___tracetype=fwd_homo___target=Banana___kernel=MF.MALA/summary.csv", df)
