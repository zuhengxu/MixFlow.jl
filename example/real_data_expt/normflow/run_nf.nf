include { crossProduct; filed; deliverables } from '../../nf-nest/cross.nf'
include { instantiate; precompile; activate } from '../../nf-nest/pkg.nf'
include { combine_csvs; } from '../../nf-nest/combine.nf'

params.dryRun = false
params.n_sample_eval = params.dryRun ? 8 : 1024
params.nrunThreads = 1

def julia_env = file("${moduleDir}/../../")
def julia_script = file("${moduleDir}/run_nf.jl")

def variables = [
    seed: 1..10,
    target: ["TReg", "SparseRegression", "Brownian", "Sonar", "LGCP"],
    // flowtype: ["real_nvp", "neural_spline_flow"],
    flowtype: ["real_nvp"],
    nlayer: [3, 5],
    lr: ["1e-3", "1e-4"],
    batchsize: [64],
    niters: [100000],
]

workflow {
    compiled_env = instantiate(julia_env) | precompile
    configs = crossProduct(variables, params.dryRun)
    combined = run_simulation(compiled_env, configs) | combine_csvs
    // plot(compiled_env, plot_script, combined)
   final_deliverable(compiled_env, combined)
}


process run_simulation {
    debug false 
    memory { 4.GB * Math.pow(2, task.attempt-1) }
    time { 5.hour * Math.pow(2, task.attempt-1) } 
    cpus 1 
    errorStrategy { task.attempt < 2 ? 'retry' : 'ignore' } 
    input:
        path julia_env 
        val config 
    output:
        path "${filed(config)}"
    """
    ${activate(julia_env,params.nrunThreads)}

    include("$julia_script")

    # get configurations
    seed = ${config.seed}
    name = "${config.target}"
    nlayer = ${config.nlayer}
    flowtype = "${config.flowtype}"
    niters = ${config.niters}
    bs = ${config.batchsize} 
    lr = ${config.lr}

    # run simulation
    df = run_norm_flow(
        seed, name, flowtype, nlayer, lr; 
        batchsize=bs, niters=niters, show_progress=false,
        nsample_eval=${params.n_sample_eval},
    )
    
    # store output
    mkdir("${filed(config)}")
    CSV.write("${filed(config)}/summary.csv", df)
    """
}


process final_deliverable {
    input:
        path julia_env 
        path combined_csvs_folder 
    output:
        path combined_csvs_folder
    publishDir "${deliverables(workflow, params)}", mode: 'copy', overwrite: true
    """
    ${activate(julia_env)}
    """
}
