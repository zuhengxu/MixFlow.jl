include(jointpath(@__DIR__, "run_mfvi.jl"))

for name in ["TReg", "SparseRegression", "Brownian", "Sonar", "LGCP"]
    get_vi_reference(1, name; batchsize = 10, niters = 100_000)
end
