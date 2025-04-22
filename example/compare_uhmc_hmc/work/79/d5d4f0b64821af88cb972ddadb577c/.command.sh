#!/usr/bin/env julia --threads=5 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/compare_uhmc_hmc/uhmc_hmc.jl")

# get configurations
seed = 3
name = "Funnel"
flowtype = MF.DeterministicMixFlow
kernel = MF.HMC
step_size = 0.1
flow_length = 300

# run simulation
df = run_tv(seed, name, flowtype, flow_length, kernel, step_size; nsample = 512)

# store output
mkdir("seed=3___target=Funnel___flowtype=MF.DeterministicMixFlow___kernel=MF.HMC___step_size=0.1___flow_length=300")
CSV.write("seed=3___target=Funnel___flowtype=MF.DeterministicMixFlow___kernel=MF.HMC___step_size=0.1___flow_length=300/summary.csv", df)
