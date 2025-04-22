#!/usr/bin/env julia --threads=5 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/synthetic_tuning/tuning.jl")

# get configurations
seed = 1
name = "Cross"
flowtype = MF.DeterministicMixFlow
kernel = MF.MALA
step_size = 0.1
flow_length = 300

# run simulation
df = run_tv(seed, name, flowtype, flow_length, kernel, step_size; nsample = 1024)

# store output
mkdir("seed=1___target=Cross___flowtype=MF.DeterministicMixFlow___kernel=MF.MALA___step_size=0.1___flow_length=300")
CSV.write("seed=1___target=Cross___flowtype=MF.DeterministicMixFlow___kernel=MF.MALA___step_size=0.1___flow_length=300/summary.csv", df)
