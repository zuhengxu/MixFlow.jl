#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/compare_traces/traces.jl")

# get configurations
seed = 4
name = "Banana"
tracetype = "mcmc"
kernel_type = MF.RWMH

# run simulation
df = run_traces(seed, name, kernel_type, tracetype)

# store output
mkdir("seed=4___tracetype=mcmc___target=Banana___kernel=MF.RWMH")
CSV.write("seed=4___tracetype=mcmc___target=Banana___kernel=MF.RWMH/summary.csv", df)
