#!/usr/bin/env julia --threads=1 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/norm_flow.jl")

# get configurations
seed = 1
name = "Banana"
flowtype = "real_nvp"
niters = 100
bs = 64 
lr = 1e-3

# run simulation
df = run_norm_flow(
    seed, name, flowtype, lr; 
    batchsize=bs, niters=niters, show_progress=false,
    nsample_eval=8,
)

# store output
mkdir("seed=1___target=Banana___flowtype=real_nvp___lr=1e-3___batchsize=64___niters=100")
CSV.write("seed=1___target=Banana___flowtype=real_nvp___lr=1e-3___batchsize=64___niters=100/summary.csv", df)
