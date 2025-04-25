#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/synthetic_tuning/metric_mixflow.jl")

# get configurations
seed = 20
name = "Banana"
flowtype = MF.BackwardIRFMixFlow
kernel = MF.HMC
step_size = 0.2
flow_length = 150

# run simulation
df = run_elbo(seed, name, flowtype, flow_length, kernel, step_size; nsample = 64)

# store output
mkdir("seed=20___kernel=MF.HMC___step_size=0.2___flow_length=150___target=Banana___flowtype=MF.BackwardIRFMixFlow")
CSV.write("seed=20___kernel=MF.HMC___step_size=0.2___flow_length=150___target=Banana___flowtype=MF.BackwardIRFMixFlow/summary.csv", df)
