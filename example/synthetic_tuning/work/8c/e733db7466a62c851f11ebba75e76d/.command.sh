#!/usr/bin/env julia --threads=5 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/synthetic_tuning/tuning.jl")

# get configurations
seed = 4
name = "Banana"
flowtype = MF.DeterministicMixFlow
kernel = MF.MALA
step_size = 0.2
flow_length = 3000

# run simulation
df = run_tv(seed, name, flowtype, flow_length, kernel, step_size; nsample = 512)

# store output
mkdir("seed=4___target=Banana___flowtype=MF.DeterministicMixFlow___kernel=MF.MALA___step_size=0.2___flow_length=3000")
CSV.write("seed=4___target=Banana___flowtype=MF.DeterministicMixFlow___kernel=MF.MALA___step_size=0.2___flow_length=3000/summary.csv", df)
