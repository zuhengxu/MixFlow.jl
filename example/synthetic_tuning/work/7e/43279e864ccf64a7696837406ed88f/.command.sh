#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/synthetic_tuning/metric_mixflow.jl")

# get configurations
seed = 1
name = "Funnel"
flowtype = MF.BackwardIRFMixFlow
kernel = MF.MALA
step_size = 0.05
flow_length = 2000

# run simulation
df = run_tv(seed, name, flowtype, flow_length, kernel, step_size; nsample = 64)

# store output
mkdir("seed=1___target=Funnel___flowtype=MF.BackwardIRFMixFlow___kernel=MF.MALA___step_size=0.05___flow_length=2000")
CSV.write("seed=1___target=Funnel___flowtype=MF.BackwardIRFMixFlow___kernel=MF.MALA___step_size=0.05___flow_length=2000/summary.csv", df)
