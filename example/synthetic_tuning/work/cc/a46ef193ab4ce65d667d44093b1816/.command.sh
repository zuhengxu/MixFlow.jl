#!/usr/bin/env julia --threads=5 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/synthetic_tuning/tuning.jl")

# get configurations
seed = 1
name = "Banana"
flowtype = MF.DeterministicMixFlow
kernel = MF.RWMH
step_size = 1.0
flow_length = 1500

# run simulation
df = run_tv(seed, name, flowtype, flow_length, kernel, step_size; nsample = 1024)

# store output
mkdir("seed=1___target=Banana___flowtype=MF.DeterministicMixFlow___kernel=MF.RWMH___step_size=1.0___flow_length=1500")
CSV.write("seed=1___target=Banana___flowtype=MF.DeterministicMixFlow___kernel=MF.RWMH___step_size=1.0___flow_length=1500/summary.csv", df)
