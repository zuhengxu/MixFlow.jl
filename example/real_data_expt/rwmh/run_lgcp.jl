include("run_rwmh.jl")


nrep = 32
nsample = 64
nchains = 30

for name in ["LGCP"]
    for flowtype in [MF.BackwardIRFMixFlow, MF.DeterministicMixFlow, MF.EnsembleIRFFlow]
        for kernel in [RWMH]
            for T in [5000]
                for seed in 1:nrep
                    println("Running simulation with seed: $seed, name: $name, flowtype: $flowtype, kernel: $kernel, T: $T")
                    df, _ = run_simulation(seed, name, flowtype, kernel, T, nchains; nsample = nsample, save_jld = true, track_cost = true)  
                    
                    # make a dir named LGCP_csv/
                    # save csv in LGCP_csv/
                    if !isdir("LGCP_csv/")
                        mkpath("LGCP_csv/")
                    end
                    # save df in the dir in csv
                    CSV.write(joinpath("LGCP_csv/", "rwmh_$(name)_$(flowtype)_$seed.csv"), df)
                end
            end
        end
    end
end