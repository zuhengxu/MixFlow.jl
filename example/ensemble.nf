include { crossProduct; filed; deliverables } from './nf-nest/cross.nf'
include { instantiate; precompile; activate } from './nf-nest/pkg.nf'
include { combine_csvs; } from './nf-nest/combine.nf'

params.dryRun = false
params.n_sample = params.dryRun ? 8 : 1024

def julia_env = file(moduleDir)
def julia_script = file(moduleDir/'ensemble.jl')

def variables = [
    seed: 1,
    target: ["Banana", "Funnel"], 
    total_cost: [0, 20, 50, 100, 200, 500],
    nchains: [5, 10], 
    kernel: ["MF.HMC", "MF.RWMH", "MF.MALA"],
    step_size: [0.01, 0.05, 0.1],
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
    memory { 2.GB * Math.pow(2, task.attempt-1) }
    time { 1.hour* Math.pow(2, task.attempt-1) } 
    cpus 1
    errorStrategy { task.attempt < 2 ? 'retry' : 'ignore' } 
    input:
        path julia_env 
        val config 
    output:
        path "${filed(config)}"
    """
    ${activate(julia_env)}

    include("$julia_script")

    # get configurations
    seed = ${config.seed}
    name = "${config.target}"
    kernel = ${config.kernel}
    step_size = ${config.step_size}
    total_cost = ${config.total_cost}
    nchains = ${config.nchains}



    # run simulation
    df = run_ensemble(seed, name, total_cost, nchains, kernel, step_size; nsample = ${params.n_sample})
    
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


// process plot {
//     input:
//         path julia_env 
//         path plot_script
//         path combined_csvs_folder 
//     output:
//         path '*.png'
//         path combined_csvs_folder
//     publishDir "${deliverables(workflow, params)}", mode: 'copy', overwrite: true
//     """
//     ${activate(julia_env)}

//     include("$plot_script")
//     create_plots("$combined_csvs_folder")
//     """
// }
