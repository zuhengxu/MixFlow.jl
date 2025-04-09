include { crossProduct; filed; deliverables } from './nf-nest/cross.nf'
include { instantiate; precompile; activate } from './nf-nest/pkg.nf'

params.dryRun = false
def julia_env = file(moduleDir)
def julia_script = file(moduleDir/'syn_vi.jl')

def variables = [
    target: ["Banana", "Funnel", "WarpedGaussian", "Cross"], 
]


// global constants
def n_iterations = params.dryRun ? 100 : 100000 // number of GD steps 
def batch_size = 10 // number of samples per batch

workflow {
    compiled_env = instantiate(julia_env) | precompile
    configs = crossProduct(variables, params.dryRun)
    combined = run_mfvi(compiled_env, configs) 
}


process run_mfvi {
    debug false 
    time 600.min // change
    cpus 1
    memory 4.GB // change
    input:
        path julia_env 
        val config 
    """
    ${activate(julia_env)}

    include("$julia_script")

    # get configurations
    name = "${config.target}"
    seed = 1
    niters = ${n_iterations}
    bs = ${batch_size}

    # run simulation
    get_vi_reference(seed, name; batchsize = bs, niters = niters)
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