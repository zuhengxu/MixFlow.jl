#!/usr/bin/env julia --threads=5 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/synthetic_tuning/tuning.jl")

# get configurations
seed = 1
name = "Funnel"
flowtype = MF.DeterministicMixFlow
kernel = MF.MALA
step_size = 0.2
flow_length = 1500

# run simulation
df = run_tv(seed, name, flowtype, flow_length, kernel, step_size; nsample = 1024)

# store output
mkdir("seed=1___target=Funnel___flowtype=MF.DeterministicMixFlow___kernel=MF.MALA___step_size=0.2___flow_length=1500")
CSV.write("seed=1___target=Funnel___flowtype=MF.DeterministicMixFlow___kernel=MF.MALA___step_size=0.2___flow_length=1500/summary.csv", df)
