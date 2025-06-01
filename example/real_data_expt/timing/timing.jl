using CSV, DataFrames

include(joinpath(@__DIR__, "../reference/run_mfvi.jl"))
include(joinpath(@__DIR__, "../baseline/run_mfvi.jl"))
include(joinpath(@__DIR__, "../normflow/run_nf.jl"))
include(joinpath(@__DIR__, "../nuts/run_nuts.jl"))
include(joinpath(@__DIR__, "../rwmh/run_rwmh.jl"))

function reference_timing(seed, name)
    # get rid of compilation time
    Random.seed!(seed)

    @info "load model $(name)"
    target, dims, ad = load_model(name)

    # get rid of compilation time
    _, _ = mfvi(target; sample_per_iter = 10, max_iters = 10, adtype = ad)
    time = @elapsed begin
        _, _ = mfvi(target; sample_per_iter = 10, max_iters = 10000, adtype = ad)
    end
    return time
end

function baseline_timing(seed, name)
    # get rid of compilation time
    run_baseline(
        seed, name, 1e-5; 
        batchsize=10, niters=5, show_progress=true,
        nsample_eval=8, save_jld=false,
    )

    time = @elapsed begin
        run_baseline(
            seed, name, 1e-5; 
            batchsize=32, niters=1000, show_progress=true,
            nsample_eval=1024, save_jld=false,
        )
    end
    return 50*time
end

function normflow_timing(
    seed, name, flowtype, nlayers
)
    # get rid of compilation time
    run_norm_flow(
        seed, name, flowtype, nlayers, 1e-5; 
        batchsize=10, niters=5, show_progress=true,
        nsample_eval=8, save_jld=false,
    )

    time = @elapsed begin
        run_norm_flow(
            seed, name, flowtype, nlayers, 1e-5; 
            batchsize=32, niters=100, show_progress=true,
            nsample_eval=1024, save_jld=false,
        )
    end
    return 500*time
end

function nuts_timing(
    seed, name
)
    # get rid of compilation time
    run_nuts(seed, name; n_samples=10)

    time = @elapsed begin
        _ = run_nuts(seed, name; n_samples=10_000)
    end
    return time
end

function irfflow_timing(
    seed, name::String, flowtype
)
    nchains = 30
    flowlength = 5000
    kernel = RWMH
    
    # get rid of compilation time
    run_simulation(
        seed, name, flowtype, kernel, 100, 10; 
        nsample = 8, track_cost = false, save_jld = false,
    )

    time = @elapsed begin
        _ = run_simulation(
            seed, name, flowtype, kernel, flowlength, nchains; 
            nsample = 64, track_cost = false, save_jld = false,
        )
    end

    return time
end



# name = "Brownian"
# baseline_timing(1, name)
# normflow_timing(1, name, "neural_spline_flow", 5)
# nuts_timing(1, name)


res_dir = joinpath(@__DIR__, "result/")
if !isdir(res_dir)
    mkpath(res_dir)
end

# all vi timing
for name in ["TReg", "Brownian", "SparseRegression", "LGCP"]

    file_name = joinpath(res_dir, "$(name)_vi_timing.csv")
    if isfile(file_name)
        @info "Skipping $(name), file already exists."
        continue
    end

    @info "Running reference timing for $(name)"
    time_ref = reference_timing(1, name)
    @info "Reference timing for $(name): $time_ref seconds"

    @info "Running baseline timing for $(name)"
    time_baseline = baseline_timing(1, name)
    @info "Baseline timing for $(name): $time_baseline seconds"

    @info "Running NSF timing for $(name)"
    time_nsf = normflow_timing(1, name, "neural_spline_flow", 3)
    @info "NSF timing for $(name): $time_nsf seconds"

    @info "Running real nvp timing for $(name)"
    time_rnvp = normflow_timing(1, name, "real_nvp", 3)
    @info "RealNVP timing for $(name): $time_rnvp seconds"

    df = DataFrame(
        name = name,
        reference_time = time_ref,
        baseline_time = time_baseline,
        normflow_time = time_nsf+time_ref,
        realnvp_time = time_rnvp+time_ref,
    ) 

    CSV.write(file_name, df)
end

# all irfflow timing
for name in ["TReg", "Brownian", "SparseRegression", "LGCP"]
    for flowtype in [BackwardIRFMixFlow, DeterministicMixFlow, EnsembleIRFFlow, IRFMixFlow]
        for seed in 1:10
            @info "Running IRFFlow timing for $(name) with flow type $(flowtype)"
            file_name = joinpath(res_dir, "$(name)_$(flowtype)_$(seed).csv")
            if isfile(file_name)
                @info "Skipping $(name) with flow type $(flowtype), of seed $(seed), file already exists."
                continue
            end

            if flowtype == IRFMixFlow && name == "LGCP"
                @info "Skipping IRFMixFlow for LGCP, as it is too slow."
                continue
            end

            time_irfflow = irfflow_timing(seed, name, flowtype)
            @info "IRFFlow timing for $(name) with flow type $(flowtype): $time_irfflow seconds"
            local df = DataFrame(
                name = name,
                flow_type = flowtype,
                irfflow_time = time_irfflow,
            )
            CSV.write(file_name, df)
        end
    end
end 


# all nuts timing
for name in ["TReg", "Brownian", "SparseRegression", "LGCP"]
    for seed in 1:10
        @info "Running NUTS timing for $(name)"  
        file_name = joinpath(res_dir, "$(name)_nuts_timing_$(seed).csv")
        if isfile(file_name)
            @info "Skipping $(name), file already exists."
            continue
        end

        time_nuts = nuts_timing(seed, name)
        @info "NUTS timing for $(name): $time_nuts seconds"
        local df = DataFrame(
            seed = seed,
            name = name,
            nuts_time = time_nuts,
        )
        CSV.write(file_name, df)
    end
end

