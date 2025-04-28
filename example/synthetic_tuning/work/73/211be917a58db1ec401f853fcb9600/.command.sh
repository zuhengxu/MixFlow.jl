#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/synthetic_tuning/metric_mixflow.jl")

# get configurations
seed = 2
name = "Cross"
flowtype = MF.BackwardIRFMixFlow
kernel = MF.RWMH
step_size = 0.2
flow_length = 2000

# run simulation
df = run_tv(seed, name, flowtype, flow_length, kernel, step_size; nsample = 64)

# store output
mkdir("seed=2___target=Cross___flowtype=MF.BackwardIRFMixFlow___kernel=MF.RWMH___step_size=0.2___flow_length=2000")
CSV.write("seed=2___target=Cross___flowtype=MF.BackwardIRFMixFlow___kernel=MF.RWMH___step_size=0.2___flow_length=2000/summary.csv", df)
