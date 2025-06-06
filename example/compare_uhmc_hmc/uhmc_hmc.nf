include { crossProduct; filed; deliverables } from '../nf-nest/cross.nf'
include { instantiate; precompile; activate } from '../nf-nest/pkg.nf'
include { combine_csvs; } from '../nf-nest/combine.nf'

params.dryRun = false
params.n_sample = params.dryRun ? 8 : 256
params.nrunThreads = 5

def julia_env = file("${moduleDir}/../")
def julia_script = file(moduleDir/'uhmc_hmc.jl')
def plot_script = file(moduleDir/'uhmc_hmc.jl')

def variables = [
    seed: 1..5,
    target: ["Banana", "Cross", "Funnel", "WarpedGaussian"], 
    flowtype: ["MF.DeterministicMixFlow"],
    kernel: ["MF.HMC", "MF.uncorrectHMC"],
    step_size: [0.01, 0.05, 0.1, 0.2],
    flow_length: [300],
]

workflow {
    compiled_env = instantiate(julia_env) | precompile
    configs = crossProduct(variables, params.dryRun)
    combined = run_simulation(compiled_env, configs) | combine_csvs
    plot(compiled_env, plot_script, combined)
    final_deliverable(compiled_env, combined)
}


process run_simulation {
    debug true 
    memory { 10.GB * Math.pow(2, task.attempt-1) }
    time { 24.hour * Math.pow(2, task.attempt-1) } 
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
    flowtype = ${config.flowtype}
    kernel = ${config.kernel}
    step_size = ${config.step_size}
    flow_length = ${config.flow_length}

    # run simulation
    df = run_tv_sweep(seed, name, flowtype, kernel, flow_length, step_size; nsample = ${params.n_sample})
    
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
    uhmc_hmc_tv_plot("$combined_csvs_folder")
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
