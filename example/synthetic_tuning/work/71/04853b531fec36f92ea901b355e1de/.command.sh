#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/synthetic_tuning/metric_mixflow.jl")

# get configurations
seed = 11
name = "WarpedGaussian"
flowtype = MF.BackwardIRFMixFlow
kernel = MF.HMC
step_size = 0.1
flow_length = 50

# run simulation
df = run_elbo(seed, name, flowtype, flow_length, kernel, step_size; nsample = 64)

# store output
mkdir("seed=11___kernel=MF.HMC___step_size=0.1___flow_length=50___target=WarpedGaussian___flowtype=MF.BackwardIRFMixFlow")
CSV.write("seed=11___kernel=MF.HMC___step_size=0.1___flow_length=50___target=WarpedGaussian___flowtype=MF.BackwardIRFMixFlow/summary.csv", df)
