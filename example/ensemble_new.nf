include { crossProduct; filed; deliverables } from './nf-nest/cross.nf'
include { instantiate; precompile; activate } from './nf-nest/pkg.nf'
include { combine_csvs; } from './nf-nest/combine.nf'

params.dryRun = false
params.n_sample = params.dryRun ? 8 : 512
params.nrun_threads = 10

def julia_env = file(moduleDir)
def julia_script = file(moduleDir/'ensemble.jl')

def variables = [
    seed: 1...5,
    target: ["Banana", "Funnel", "Cross", "WarpedGaussian"], 
<<<<<<< HEAD:example/ensemble.nf
    flow_length: [0, 10, 20, 50, 80, 100],
    nchains: [1, 5, 10, 20], 
    kernel: ["MF.HMC", "MF.RWMH", "MF.MALA"],
    step_size: [0.01, 0.05, 0.1, 0.2],
=======
    flow_length: [0, 10, 20, 30],
    nchains: [1, 5, 10, 20, 30], 
    kernel: ["MF.HMC", "MF.RWMH", "MF.MALA"],
    step_size: [0.03, 0.05, 0.1, 0.2],
>>>>>>> 7542a3ceeb4a91764b60a52d7de0b8c92274be18:example/ensemble_new.nf
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
    memory { 16.GB * Math.pow(2, task.attempt-1) }
<<<<<<< HEAD:example/ensemble.nf
    time { 4.hour* Math.pow(2, task.attempt-1) } 
=======
    time { 10.hour* Math.pow(2, task.attempt-1) } 
>>>>>>> 7542a3ceeb4a91764b60a52d7de0b8c92274be18:example/ensemble_new.nf
    cpus 1
    errorStrategy { task.attempt < 2 ? 'retry' : 'ignore' } 
    input:
        path julia_env 
        val config 
    output:
        path "${filed(config)}"
    """
    ${activate(julia_env,params.nrun_threads)}

    include("$julia_script")

    # get configurations
    seed = ${config.seed}
    name = "${config.target}"
    kernel = ${config.kernel}
    step_size = ${config.step_size}
    T = ${config.flow_length}
    nchains = ${config.nchains}

    # run simulation
    df = run_ensemble(seed, name, T, nchains, kernel, step_size; nsample = ${params.n_sample})
    
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
