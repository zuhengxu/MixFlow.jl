#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/synthetic_tuning/metric_mixflow.jl")

# get configurations
seed = 25
name = "Cross"
flowtype = MF.BackwardIRFMixFlow
kernel = MF.HMC
step_size = 0.1
flow_length = 200

# run simulation
df = run_tv(seed, name, flowtype, flow_length, kernel, step_size; nsample = 64)

# store output
mkdir("seed=25___target=Cross___flowtype=MF.BackwardIRFMixFlow___kernel=MF.HMC___step_size=0.1___flow_length=200")
CSV.write("seed=25___target=Cross___flowtype=MF.BackwardIRFMixFlow___kernel=MF.HMC___step_size=0.1___flow_length=200/summary.csv", df)
