#!/usr/bin/env julia --threads=2 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/norm_flow.jl")

# get configurations
seed = 4
name = "WarpedGaussian"
flowtype = "real_nvp"
niters = 50000
bs = 64 
lr = 1e-3

# run simulation
df = run_norm_flow(
    seed, name, flowtype, lr; 
    batchsize=bs, niters=niters, show_progress=false,
    nsample_eval=512,
)

# store output
mkdir("seed=4___target=WarpedGaussian___flowtype=real_nvp___lr=1e-3___batchsize=64___niters=50000")
CSV.write("seed=4___target=WarpedGaussian___flowtype=real_nvp___lr=1e-3___batchsize=64___niters=50000/summary.csv", df)
