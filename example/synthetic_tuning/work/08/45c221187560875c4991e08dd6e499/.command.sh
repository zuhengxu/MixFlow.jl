#!/usr/bin/env julia --threads=5 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/synthetic_tuning/tuning.jl")

# get configurations
seed = 4
name = "Funnel"
flowtype = MF.DeterministicMixFlow
kernel = MF.RWMH
step_size = 0.5
flow_length = 3000

# run simulation
df = run_tv(seed, name, flowtype, flow_length, kernel, step_size; nsample = 512)

# store output
mkdir("seed=4___target=Funnel___flowtype=MF.DeterministicMixFlow___kernel=MF.RWMH___step_size=0.5___flow_length=3000")
CSV.write("seed=4___target=Funnel___flowtype=MF.DeterministicMixFlow___kernel=MF.RWMH___step_size=0.5___flow_length=3000/summary.csv", df)
