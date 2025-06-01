include(joinpath(@__DIR__, "../../julia_env/evaluation.jl"))

n_sample_eval = 64

# run tv evaluation 
for name in ["Banana", "Funnel", "WarpedGaussian", "Cross"]
    for flowtype in [MF.DeterministicMixFlow, MF.BackwardIRFMixFlow, MF.IRFMixFlow]
        for kernel in [MF.HMC]
            for flowlength in [300]
                for step_size in [0.01, 0.05, 0.1, 0.2]
                    for nchains in [30]
                        for seed in 1:32

                            # run simulation
                            df = run_tv_sweep(seed, name, flowtype, kernel, flowlength, step_size; nsample=n_sample_eval, nchains=nchains)

                            # save simulation results
                            res_dir = joinpath(@__DIR__, "csvs/")
                            if !isdir(res_dir)
                                mkpath(res_dir)
                            end

                            CSV.write(
                                joinpath(res_dir, "$(name)_$(kernel)_$(flowtype)_$(flowlength)_$(step_size)_$(seed).csv"),
                                df,
                            )

                        end
                    end
                end
            end
        end
    end
end
for name in ["Banana", "Funnel", "WarpedGaussian", "Cross"]
    for flowtype in [MF.EnsembleIRFFlow]
        for kernel in [MF.HMC]
            for flowlength in [0, 10, 20, 50, 80, 100, 200, 300]
                for step_size in [0.01, 0.05, 0.1, 0.2]
                    for nchains in [30]
                        for seed in 1:32

                            # run simulation
                            df = run_tv_sweep(seed, name, flowtype, kernel, flowlength, step_size; nsample=n_sample_eval, nchains=nchains)

                            # save simulation results
                            res_dir = joinpath(@__DIR__, "csvs/")
                            if !isdir(res_dir)
                                mkpath(res_dir)
                            end

                            CSV.write(
                                joinpath(res_dir, "$(name)_$(kernel)_$(flowtype)_$(flowlength)_$(step_size)_$(seed).csv"),
                                df,
                            )

                        end
                    end
                end
            end
        end
    end
end


for name in ["Banana", "Funnel", "WarpedGaussian", "Cross"]
    for flowtype in [MF.DeterministicMixFlow, MF.BackwardIRFMixFlow, MF.IRFMixFlow]
        for kernel in [MF.MALA]
            for flowlength in [2000]
                for step_size in [0.05, 0.2, 1.0]
                    for nchains in [30]
                        for seed in 1:32

                            # run simulation
                            df = run_tv_sweep(seed, name, flowtype, kernel, flowlength, step_size; nsample=n_sample_eval, nchains=nchains)

                            # save simulation results
                            res_dir = joinpath(@__DIR__, "csvs/")
                            if !isdir(res_dir)
                                mkpath(res_dir)
                            end

                            CSV.write(
                                joinpath(res_dir, "$(name)_$(kernel)_$(flowtype)_$(flowlength)_$(step_size)_$(seed).csv"),
                                df,
                            )

                        end
                    end
                end
            end
        end
    end
end
for name in ["Banana", "Funnel", "WarpedGaussian", "Cross"]
    for flowtype in [MF.EnsembleIRFFlow]
        for kernel in [MF.MALA]
            for flowlength in [0, 10, 20, 50, 80, 100, 200, 300]
                for step_size in [0.05, 0.2, 1.0]
                    for nchains in [30]
                        for seed in 1:32

                            # run simulation
                            df = run_tv_sweep(seed, name, flowtype, kernel, flowlength, step_size; nsample=n_sample_eval, nchains=nchains)

                            # save simulation results
                            res_dir = joinpath(@__DIR__, "csvs/")
                            if !isdir(res_dir)
                                mkpath(res_dir)
                            end

                            CSV.write(
                                joinpath(res_dir, "$(name)_$(kernel)_$(flowtype)_$(flowlength)_$(step_size)_$(seed).csv"),
                                df,
                            )

                        end
                    end
                end
            end
        end
    end
end


for name in ["Banana", "Funnel", "WarpedGaussian", "Cross"]
    for flowtype in [MF.DeterministicMixFlow, MF.BackwardIRFMixFlow, MF.IRFMixFlow]
        for kernel in [MF.RWMH]
            for flowlength in [4000]
                for step_size in [0.05, 0.2, 1.0]
                    for nchains in [30]
                        for seed in 1:32

                            # run simulation
                            df = run_tv_sweep(seed, name, flowtype, kernel, flowlength, step_size; nsample=n_sample_eval, nchains=nchains)

                            # save simulation results
                            res_dir = joinpath(@__DIR__, "csvs/")
                            if !isdir(res_dir)
                                mkpath(res_dir)
                            end

                            CSV.write(
                                joinpath(res_dir, "$(name)_$(kernel)_$(flowtype)_$(flowlength)_$(step_size)_$(seed).csv"),
                                df,
                            )

                        end
                    end
                end
            end
        end
    end
end


for name in ["Banana", "Funnel", "WarpedGaussian", "Cross"]
    for flowtype in [MF.EnsembleIRFFlow]
        for kernel in [MF.RWMH]
            for flowlength in [0, 10, 20, 50, 80, 100, 200, 300]
                for step_size in [0.05, 0.2, 1.0]
                    for nchains in [30]
                        for seed in 1:32

                            # run simulation
                            df = run_tv_sweep(seed, name, flowtype, kernel, flowlength, step_size; nsample=n_sample_eval, nchains=nchains)

                            # save simulation results
                            res_dir = joinpath(@__DIR__, "csvs/")
                            if !isdir(res_dir)
                                mkpath(res_dir)
                            end

                            CSV.write(
                                joinpath(res_dir, "$(name)_$(kernel)_$(flowtype)_$(flowlength)_$(step_size)_$(seed).csv"),
                                df,
                            )

                        end
                    end
                end
            end
        end
    end
end


# get elbo, ess, logz results
for name in ["Banana", "Funnel", "WarpedGaussian", "Cross"]
    for flowtype in [MF.DeterministicMixFlow, MF.BackwardIRFMixFlow, MF.EnsembleIRFFlow, MF.IRFMixFlow]
        for kernel in [MF.RWMH]
            for flow_length in [4000]
                for step_size in [1.0]
                    for nchains in [30]
                        for seed in 1:32

                            if flowtype == MF.EnsembleIRFFlow
                                flowlength = 3000
                            end
                            # run simulation

                            df, _ = flow_evaluation(
                                seed, name, flowtype, kernel, flow_length, step_size;
                                nsample=n_sample_eval, nchains=nchains,
                                track_cost=false,
                            )

                            # save simulation results
                            res_dir = joinpath(@__DIR__, "elbos/")
                            if !isdir(res_dir)
                                mkpath(res_dir)
                            end

                            CSV.write(
                                joinpath(res_dir, "$(name)_$(kernel)_$(flowtype)_$(flowlength)_$(step_size)_$(seed).csv"),
                                df,
                            )

                        end
                    end
                end
            end
        end

        for kernel in [MF.HMC]
            for flow_length in [200]
                for step_size in [0.1, 0.2]
                    for nchains in [30]
                        for seed in 1:32

                            # run simulation

                            df, _ = flow_evaluation(
                                seed, name, flowtype, kernel, flow_length, step_size;
                                nsample=n_sample_eval, nchains=nchains,
                                track_cost=false,
                            )

                            # save simulation results
                            res_dir = joinpath(@__DIR__, "elbos/")
                            if !isdir(res_dir)
                                mkpath(res_dir)
                            end

                            CSV.write(
                                joinpath(res_dir, "$(name)_$(kernel)_$(flowtype)_$(flowlength)_$(step_size)_$(seed).csv"),
                                df,
                            )

                        end
                    end
                end
            end
        end
    end
end


