using CSV, DataFrames
using StatsPlots, Plots
using Pipe

using StatsBase, Statistics
using Random, Distributions, LinearAlgebra

_remove_nan(dat::AbstractVector{T}) where {T} = dat[iszero.(isnan.(dat))]

function _remove_nan(df::AbstractDataFrame) 
    mask = map(row -> any(x -> x isa AbstractFloat && (isnan(x) || isinf(x)), row), eachrow(df))
    df = df[.!mask, :] # remove all rows with Inf and NaN
    return df
end

function get_percentiles(dat; p1=25, p2=75)
    n = size(dat, 1)

    plow = zeros(n)
    phigh = zeros(n)

    for i in 1:n
        # dat_remove_nan = (dat[i, :])[iszero.(isnan.(dat[i,:]))]
        dat_remove_nan = _remove_nan(dat[i, :])
        median_remove_nan = median(dat_remove_nan)
        plow[i] = median_remove_nan - percentile(vec(dat_remove_nan), p1)
        phigh[i] = percentile(vec(dat_remove_nan), p2) - median_remove_nan
    end

    return plow, phigh
end

function get_median(dat)
    n = size(dat, 1)
    med = zeros(n)

    for i in 1:n
        # dat_remove_nan = (dat[:, i])[iszero.(isnan.(dat[:, i]))]
        dat_remove_nan = _remove_nan(dat[i, :])
        med[i] = median(dat_remove_nan)
    end

    return med
end

"""
get subdataframe from the combined csv
"""
function _subset_expt(
    df::AbstractDataFrame, 
    expt_keys::Vector{Symbol},
    expt_vals::Vector,
)
    df_g = groupby(df, expt_keys)
    df_s = df_g[(expt_vals...,)]
    return df_s
end

function _subset_expt(df::AbstractDataFrame, selector::Dict)
    expt_keys = collect(keys(selector))
    expt_vals = collect(values(selector))
    df_s = _subset_expt(df, expt_keys, expt_vals)
    return df_s
end

function subset_expt(
    combined_csv_dir::String,
    selector::Dict,
)
    df = CSV.read(combined_csv_dir, DataFrame)
    # df = _remove_nan(df)
    df_s = _subset_expt(df, selector)
    return df_s
end

function _process_nan_for_errorline!(
    ymat::AbstractMatrix{T}
) where T
    # if one row has nan, change the whole row to nan
    for i in 1:size(ymat, 1)
        if any(isnan.(ymat[i, :]))
            ymat[i, :] .= NaN
        end
    end
end

function _process_nan_for_errorline!(
    ymat::AbstractArray{T, 3}
) where T
    @views for i in 1:size(ymat, 3)
        _process_nan_for_errorline!(ymat[:, :, i])
    end
end


"""
process the dataframe to a 3 way array that can be sent to StatsPlots.errorline

input:
- df (DataFrame) - the dataframe for a specific expt to be processed
- x_key (Symbol) - the column name for x-axis
- y_key (Symbol) - the column name for y-axis
- rep_key (Symbol) - the column name for repeated observations
- group_key (Symbol) - the column name for group. This is used to group the data along the 3rd dimension of the output


output:
- x (vector{<:Real}) - the values along the x-axis for each y-point

- y (matrix [x, repeat, group]) - values along y-axis wrt x. The first dimension must be of equal length to that of x.
        The second dimension is treated as the repeated observations and error is computed along this dimension. If the
        matrix has a 3rd dimension this is treated as a new group.
- g (vector) - the group values. The length of this vector must be equal to the 3rd dimension of y.
"""
function _process_for_grouped_errorline(
    df::AbstractDataFrame,          
    x_key::Symbol,
    y_key::Symbol,
    rep_key::Symbol,
    group_key::Symbol,
)
    # first sort x 
    sort!(df, x_key)

    xs = unique(df[!, x_key])
    rs = unique(df[!, rep_key])
    gs = unique(df[!, group_key])

    d_tmp = @pipe df |>
        select(_, [x_key, y_key, rep_key, group_key]) |>
        groupby(_, [x_key, group_key])

    y_mat = zeros(length(xs), length(rs), length(gs))
    
    g_vals = similar(gs)
    for (i, x) in enumerate(xs)
        for (j, g) in enumerate(gs)
            # add catch block in case the rep key doesn;t have same length
            try
                y_mat[i, :, j] .= d_tmp[(x, g)][!, y_key]
            catch e
                y_mat[i, :, j] .= NaN
                local v = d_tmp[(x, g)][!, y_key]
                local n = length(v)
                y_mat[i, 1:n, j] .= v
            end

            if i == 1
                g_vals[j] = g
            end
        end
    end
    # sync all rows with nans, avoiding plotting issues with error bar
    _process_nan_for_errorline!(y_mat)
    return xs, y_mat, g_vals
end 

"""
add a marker to the figure indicating the location of nan happens
ymat has to be processed by _process_nan_for_errorline!
"""
function nan_marks(xs, y_mat::AbstractMatrix)
    x_pos = []
    y_pos = []
    idx = findfirst(isnan, y_mat[:,1]) 
    if idx === nothing
        return nothing, nothing
    end
    push!(x_pos, xs[idx-1])        
    push!(y_pos, mean(y_mat[idx-1, :]))
    return x_pos, y_pos
end

function nan_marks(xs, y_mat::AbstractArray{T, 3}) where T
    x_pos = []
    y_pos = []
    for i in 1:size(y_mat, 3)
        xps, yps = nan_marks(xs, @view(y_mat[:, :, i]))
        if xps === nothing
            continue
        end
        push!(x_pos, xps)
        push!(y_pos, yps)
    end
    if isempty(x_pos)
        return nothing, nothing
    end
    return reduce(vcat, x_pos), reduce(vcat, y_pos)
end

function add_nan_mark_fig!(fg, xs, y_mat; plt_kargs...)
    x_pos, y_pos = nan_marks(xs, y_mat)
    if x_pos !== nothing
        scatter!(
            fg,
            x_pos, y_pos,
            label = "",
            marker = :x,
            markercolor = :red;
            plt_kargs...,
        )
    end
end

function groupederrorline(
    df::AbstractDataFrame,          
    x_key::Symbol,
    y_key::Symbol,
    rep_key::Symbol,
    group_key::Symbol;
    mark_nan = false,
    plt_kargs...
)
    xs, y_mat, gs = _process_for_grouped_errorline(
        df,
        x_key,
        y_key,
        rep_key,
        group_key,
    )

    fg = StatsPlots.errorline(
        xs, y_mat,
        label = reshape(gs, 1, length(gs)),
        xlabel = x_key,
        ylabel = y_key,
        legend = :best;
        plt_kargs...,
    )
    if mark_nan
        add_nan_mark_fig!(fg, xs, y_mat; label = "NaN")
    end
    return fg
end

function add_groupederrorline!(
    fg,
    df::AbstractDataFrame,          
    x_key::Symbol,
    y_key::Symbol,
    rep_key::Symbol,
    group_key::Symbol;
    mark_nan = false,
    plt_kargs...
)
    xs, y_mat, _ = _process_for_grouped_errorline(
        df,
        x_key,
        y_key,
        rep_key,
        group_key,
    )

    StatsPlots.errorline!(
        fg,
        xs, y_mat,
        xlabel = x_key,
        ylabel = y_key;
        plt_kargs...,
    )
    if mark_nan
        add_nan_mark_fig!(fg, xs, y_mat; label = "NaN")
    end
end


# string manipulation
"""
throw away the hanging prefix of the string
"""
_throw_dot(s::AbstractString) = split(s, ".", limit= 2)[end]

# ##########
# # example
# ##########
# # pth = "deliverables/scriptName=elbo.nf___dryRun=false___n_sample=512___nrunThreads=10/"
# # cd(pth)

# df = CSV.read("output/summary.csv", DataFrame) 

# targets = unique(df.target)
# flowtypes = unique(df.flowtype)
# kernels = unique(df.kernel)[1:end-1] # no uncorrectHMC for now

# for (t, k, f) in Iterators.product(targets, kernels, flowtypes)
#     println("target: $t, kernel: $k, flowtype: $f")
#     selector = Dict(
#         :target => t,
#         :flowtype => f,
#         :kernel => k,
#     )
#     ds = _subset_expt(df, selector)

#     for metric in [:elbo, :logZ, :ess]
#         fg = groupederrorline(
#             ds, :flow_length, metric, :seed, :step_size;
#             errorstyle = :ribbon,
#             legend = :best,
#             legendtitle = "step size",
#             title = "$(t)__$(_throw_dot(k))__$(_throw_dot(f))",
#         )

#         if metric == :elbo
#             hline!(fg, [0], color = :red, linestyle = :dash, lw = 2, label = "optimal")
#         elseif metric == :logZ
#             hline!(fg, [0], color = :red, linestyle = :dash, lw = 2, label = "optimal")
#         elseif metric == :ess
#             hline!(fg, [512], color = :red, linestyle = :dash, lw = 2, label = "n particles")
#         end

#         savefig(fg, "$(t)__$(_throw_dot(k))__$(_throw_dot(f))_$(metric).png")
#     end
# end
