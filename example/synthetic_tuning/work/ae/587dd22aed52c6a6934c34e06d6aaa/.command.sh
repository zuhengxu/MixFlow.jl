#!/usr/bin/env julia --threads=10 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/synthetic_tuning/tuning.jl")

# get configurations
seed = [1, 2, 3, 4, 5]
name = "Banana"
flowtype = MF.DeterministicMixFlow
kernel = MF.MALA
step_size = 0.05
flow_length = 3000

# run simulation
df = run_tv(seed, name, flowtype, flow_length, kernel, step_size; nsample = 8)

# store output
mkdir("seed=[1, 2, 3, 4, 5]___target=Banana___flowtype=MF.DeterministicMixFlow___kernel=MF.MALA___step_size=0.05___flow_length=3000")
CSV.write("seed=[1, 2, 3, 4, 5]___target=Banana___flowtype=MF.DeterministicMixFlow___kernel=MF.MALA___step_size=0.05___flow_length=3000/summary.csv", df)
