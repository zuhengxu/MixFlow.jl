using LinearAlgebra
using MixFlow

using CSV, DataFrames, Statistics, Random
using Pipe

function _get_kernel_name(K::MixFlow.InvolutiveKernel)
    str = string(typeof(K))
    name = split(str, "{", limit = 2)[1]
    return name
end

function running_mean(xs::Matrix{T}) where T
    cumsum(xs; dims = 2) ./ [1:size(xs, 2) ;]'
end
function running_mean(xs::Vector{T}) where T
    cumsum(xs) ./ [1:length(xs) ;]
end

function running_square(xs::Matrix{T}) where T
    cumsum(xs.^2, dims = 2) ./ [1: size(xs, 2) ;]'
end


# find directories that contain str
function _find_dir(str::String)
    readdir()[contains.(readdir(), str)]
end


function _read_csv_prefix(pfx::String; append=("output", "summary.csv"))
    pth = joinpath(pfx, append...)
    df = CSV.File(pth) |> DataFrame
    return df
end

# vcat all df in dfs if they have the same columns
# and write a csv called tv_mixflow.csv
function _hcat_all_dfs(dfs::Vector{DataFrame})
    # check if all dfs have the same columns
    cols = names(dfs[1])
    for df in dfs
        if names(df) != cols
            error("DataFrames have different columns")
        end
    end

    # hcat all dfs
    df = vcat(dfs...)
    return df
end

