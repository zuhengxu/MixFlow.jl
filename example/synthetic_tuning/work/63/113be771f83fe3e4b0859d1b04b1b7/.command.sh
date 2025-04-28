#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/synthetic_tuning/metric_mixflow.jl")

# get configurations
seed = 3
name = "WarpedGaussian"
flowtype = MF.BackwardIRFMixFlow
kernel = MF.RWMH
step_size = 0.05
flow_length = 2000

# run simulation
df = run_tv(seed, name, flowtype, flow_length, kernel, step_size; nsample = 64)

# store output
mkdir("seed=3___target=WarpedGaussian___flowtype=MF.BackwardIRFMixFlow___kernel=MF.RWMH___step_size=0.05___flow_length=2000")
CSV.write("seed=3___target=WarpedGaussian___flowtype=MF.BackwardIRFMixFlow___kernel=MF.RWMH___step_size=0.05___flow_length=2000/summary.csv", df)
