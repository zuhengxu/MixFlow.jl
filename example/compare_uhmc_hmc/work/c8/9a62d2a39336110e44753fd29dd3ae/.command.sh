#!/usr/bin/env julia --threads=5 --project=/home/zuheng/Research/MixFlow.jl/example

include("/home/zuheng/Research/MixFlow.jl/example/compare_uhmc_hmc/uhmc_hmc.jl")
uhmc_hmc_tv_plot("output")
