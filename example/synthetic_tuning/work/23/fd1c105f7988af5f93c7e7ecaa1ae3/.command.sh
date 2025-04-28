#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/synthetic_tuning/metric_mixflow.jl")

# get configurations
seed = 7
name = "Cross"
flowtype = MF.DeterministicMixFlow
kernel = MF.HMC
step_size = 0.01
flow_length = 200

# run simulation
df = run_tv(seed, name, flowtype, flow_length, kernel, step_size; nsample = 64)

# store output
mkdir("seed=7___target=Cross___flowtype=MF.DeterministicMixFlow___kernel=MF.HMC___step_size=0.01___flow_length=200")
CSV.write("seed=7___target=Cross___flowtype=MF.DeterministicMixFlow___kernel=MF.HMC___step_size=0.01___flow_length=200/summary.csv", df)
