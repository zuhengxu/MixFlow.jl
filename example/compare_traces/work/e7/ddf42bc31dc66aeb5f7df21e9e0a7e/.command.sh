#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/compare_traces/traces.jl")

# get configurations
seed = 27
name = "Funnel"
tracetype = "fwd_homo"
kernel_type = MF.RWMH

# run simulation
df = run_traces(seed, name, kernel_type, tracetype)

# store output
mkdir("seed=27___tracetype=fwd_homo___target=Funnel___kernel=MF.RWMH")
CSV.write("seed=27___tracetype=fwd_homo___target=Funnel___kernel=MF.RWMH/summary.csv", df)
