include { crossProduct; filed; deliverables } from '../../nf-nest/cross.nf'
include { instantiate; precompile; activate } from '../../nf-nest/pkg.nf'
include { combine_csvs; } from '../../nf-nest/combine.nf'

params.dryRun = false
params.n_sample_eval = params.dryRun ? 8 : 1024
params.niters = params.dryRun ? 10 : 50000
params.nrunThreads = 1

def julia_env = file("${moduleDir}/../../julia_env")
def julia_script = file("${moduleDir}/run_mfvi.jl")

def variables = [
    target: ["TReg", "Sonar", "Brownian", "SparseRegression"],
    lr: ["1e-3"],
    batchsize: [32],
    seed: 1..5,
]

workflow {
    compiled_env = instantiate(julia_env) | precompile
    configs = crossProduct(variables, params.dryRun)
    combined = run_baseline(compiled_env, configs) | combine_csvs
    // plot(compiled_env, plot_script, combined)
   final_deliverable(compiled_env, combined)
}


process run_baseline {
    debug false 
    memory { 5.GB * Math.pow(2, task.attempt-1) }
    time { 24.hour * Math.pow(2, task.attempt-1) } 
    cpus 1 
//    errorStrategy { task.attempt < 2 ? 'retry' : 'ignore' } 
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
    bs = ${config.batchsize} 
    lr = ${config.lr}
    niters = ${params.niters}

    # run simulation
    df = run_baseline(
        seed, name, lr; 
        batchsize=bs, niters=niters, show_progress=false,
        nsample_eval=${params.n_sample_eval},
        save_jld = true
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
