#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/synthetic_tuning/metric_mixflow.jl")

# get configurations
seed = 2
name = "Banana"
flowtype = MF.IRFMixFlow
kernel = MF.RWMH
step_size = 1.0
flow_length = 2000

# run simulation
df = run_tv(seed, name, flowtype, flow_length, kernel, step_size; nsample = 64)

# store output
mkdir("seed=2___target=Banana___flowtype=MF.IRFMixFlow___kernel=MF.RWMH___step_size=1.0___flow_length=2000")
CSV.write("seed=2___target=Banana___flowtype=MF.IRFMixFlow___kernel=MF.RWMH___step_size=1.0___flow_length=2000/summary.csv", df)
