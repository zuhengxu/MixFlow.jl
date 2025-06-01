include { crossProduct; filed; deliverables } from '../../nf-nest/cross.nf'
include { instantiate; precompile; activate } from '../../nf-nest/pkg.nf'
include { combine_csvs; } from '../../nf-nest/combine.nf'

params.dryRun = false
params.n_sample = params.dryRun ? 8 : 64 
params.nrunThreads = 1

def julia_env = file("${moduleDir}/../../julia_env")
def julia_script = file(moduleDir/'run_rwmh.jl')
// def plot_script = file(moduleDir/'tuning.jl')

def variables = [
    target: ["LGCP"],
    flowtype: ["BackwardIRFMixFlow", "DeterministicMixFlow"],
    kernel: ["MF.RWMH"],
    nchains: [30],
    flow_length: [5000],
    seed: 1..16,
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
    memory { 5.GB * Math.pow(2, task.attempt-1) }
    time { 24.hour * Math.pow(2, task.attempt-1) } 
    cpus 1 
    errorStrategy { task.attempt < 3 ? 'retry' : 'ignore' } 
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
    kernel = ${config.kernel}
    flowtype = ${config.flowtype}
    T = ${config.flow_length}
    nchains = ${config.nchains}

    # run simulation
    df, _ = run_simulation(seed, name, flowtype, kernel, T, nchains; nsample = ${params.n_sample}, save_jld = true)

    # store output
    mkdir("${filed(config)}")
    CSV.write("${filed(config)}/summary.csv", df)
    """
}

process plot {
    input:
        path julia_env 
        path plot_script
        path combined_csvs_folder 
    output:
        path '*.png'
        path combined_csvs_folder
    publishDir "${deliverables(workflow, params)}", mode: 'copy', overwrite: true
    """
    ${activate(julia_env,params.nrunThreads)}

    include("$plot_script")
    tv_plot("$combined_csvs_folder")
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

