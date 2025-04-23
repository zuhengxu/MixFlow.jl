include { crossProduct; filed; deliverables } from '../nf-nest/cross.nf'
include { instantiate; precompile; activate } from '../nf-nest/pkg.nf'
include { combine_csvs; } from '../nf-nest/combine.nf'

params.dryRun = false
params.nrunThreads = 1

def julia_env = file("${moduleDir}/../")
def julia_script = file(moduleDir/'traces.jl')
def plot_script = file(moduleDir/'trace_plotting.jl')

def variables = [
    seed : 1..32,
    tracetype: ["mcmc", "fwd_homo", "fwd_irf", "inv_irf", "bwd_irf", "bwd_inv_irf"],
    target: ["Banana", "Funnel", "WarpedGaussian", "Cross"], 
    kernel: ["MF.RWMH", "MF.MALA", "MF.HMC"],
]

workflow {
    compiled_env = instantiate(julia_env) | precompile
    configs = crossProduct(variables, params.dryRun)
    combined = run_simulation(compiled_env, configs) | combine_csvs
    plot(compiled_env, plot_script, combined)
    final_deliverable(compiled_env, combined)
}


process run_simulation {
    debug false 
    memory { 3.GB * Math.pow(2, task.attempt-1) }
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
    tracetype = "${config.tracetype}"
    kernel_type = ${config.kernel}

    # run simulation
    df = run_traces(seed, name, kernel_type, tracetype)

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
    ${activate(julia_env, 5)}

    include("$plot_script")
    grouped_ess_plot(
        "$combined_csvs_folder";
        dpi = 800,
        size = (1000, 800),
        margin = 10Plots.mm,
        xtickfontsize = 18, ytickfontsize = 18, yguidefontsize = 18,
        legendfontsize = 11, titlefontsize = 18,
    )
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