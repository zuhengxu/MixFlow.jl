include { crossProduct; filed; deliverables } from '../../nf-nest/cross.nf'
include { instantiate; precompile; activate } from '../../nf-nest/pkg.nf'
include { combine_csvs; } from '../../nf-nest/combine.nf'

params.dryRun = false
params.nrunThreads = 1
params.n_sample = params.dryRun ? 1000 : 5000

def julia_env = file("${moduleDir}/../../julia_env")
def julia_script = file("${moduleDir}/run_nuts.jl")

def variables = [
    target: ["SparseRegression", "TReg", "Sonar", "Brownian", "LGCP"],
    seed: 1..10,
]

workflow {
    compiled_env = instantiate(julia_env) | precompile
    configs = crossProduct(variables, params.dryRun)
    combined = run_nuts(compiled_env, configs) | combine_csvs
    final_deliverable(compiled_env, combined)
}


process run_nuts {
    debug false 
    memory { 10.GB * Math.pow(2, task.attempt-1) }
    time { 10.hour * Math.pow(2, task.attempt-1) } 
    cpus 1 
    errorStrategy { 'ignore' } 
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
    flowtype = ${config.flowtype}

    # processing rwmh importance sampling 
    df = df_rwmh_is(seed, name, flowtype)

    # run NUTS simulation
    df_n = df_nuts(seed, name)

    # store output
    mkdir("${filed(config)}")
    CSV.write("${filed(config)}/rwmh_is.csv", df)
    CSV.write("${filed(config)}/nuts.csv", df_n)
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

