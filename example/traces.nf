include { crossProduct; filed; deliverables } from './nf-nest/cross.nf'
include { instantiate; precompile; activate } from './nf-nest/pkg.nf'
include { combine_csvs; } from './nf-nest/combine.nf'

params.dryRun = false
def julia_env = file(moduleDir)
def julia_script = file(moduleDir/'stability_check.jl')

def variables = [
    target: ["Banana", "Funnel", "WarpedGaussian", "Cross"], 
    kernel: ["HMC", "uncorrectHMC", "RWMH", "MALA"],
]

kernel_string = [
    HMC: "HMC(50, 0.02)",
    MALA: "MALA(0.25, ones(2))", 
    RWMH: "RWMH(0.3, ones(2))", 
]

t_max_string = [
    HMC: 1000,
    MALA: 1500,
    RWMH: 10000,
]

// global constants
// def n_iterations = 100000 // number of GD steps 
// Notes: With n_iterations = 10_000, takes ~24 minutes total 
// with n = 100_000, takes 3h 15 mins total

workflow {
    compiled_env = instantiate(julia_env) | precompile
    configs = crossProduct(variables, params.dryRun)
    combined = run_simulation(compiled_env, configs) 
    // plot(compiled_env, plot_script, combined)
//    final_deliverable(compiled_env, combined)
}


process run_simulation {
    debug false 
    time 300.min // change
    cpus 5
    memory 8.GB // change
    input:
        path julia_env 
        val config 
    output:
        path "${filed(config)}"
    """
    ${activate(julia_env)}

    include("$julia_script")

    # get configurations
    name = "${config.target}"
    kernel = ${kernel_string[config.kernel]}
    T_max = ${t_max_string[config.kernel]}


    # run simulation
    run_traces(name, kernel, T_max)
    """
}


// process final_deliverable {
//     input:
//         path julia_env 
//         path combined_csvs_folder 
//     output:
//         path combined_csvs_folder
//     publishDir "${deliverables(workflow, params)}", mode: 'copy', overwrite: true
//     """
//     ${activate(julia_env)}
//     """
// }


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