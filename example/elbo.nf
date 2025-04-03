include { crossProduct; filed; deliverables } from './nf-nest/cross.nf'
include { instantiate; precompile; activate } from './nf-nest/pkg.nf'
include { combine_csvs; } from './nf-nest/combine.nf'

params.dryRun = false
params.n_sample = params.dryRun ? 8 : 512

def julia_env = file(moduleDir)
def julia_script = file(moduleDir/'elbo.jl')

def variables = [
    // seed: 1..10,
    // target: ["Banana", "Funnel", "WarpedGaussian", "Cross"], 
    // flowtype: ["MF.BackwardIRFMixFlow", "MF.DeterministicMixFlow", "MF.IRFMixFlow"],
    // kernel: ["MF.HMC", "MF.uncorrectHMC", "MF.RWMH", "MF.MALA"],
    // step_size: [0.001, 0.005, 0.01, 0.05, 0.1],
    // flow_length: [0, 10, 20, 50, 100, 250, 500],
    seed: 1..3,
    target: ["Banana", "Funnel"],
    flowtype: ["MF.BackwardIRFMixFlow", "MF.DeterministicMixFlow"],
    kernel: ["MF.RWMH", "MF.MALA"],
    step_size: [0.001, 0.005],
    flow_length: [0, 10],
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
    time 600.min // change
    cpus 1 
    memory 2.GB // change
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
    flowtype = ${config.flowtype}
    kernel = ${config.kernel}
    step_size = ${config.step_size}
    flow_length = ${config.flow_length}

    # run simulation
    df = run_elbo(seed, name, flowtype, flow_length, kernel, step_size; nsample = ${params.n_sample})
    
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
