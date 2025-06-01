using KernelFunctions
using PDMats


struct LogGaussianCoxProcess{
    Area<:Real,Counts<:AbstractVector{<:Int},GPMean,GPCovChol,LogJac<:Real
}
    area        :: Area
    counts      :: Counts
    gp_mean     :: GPMean
    gp_cov_chol :: GPCovChol
    logjac      :: LogJac
end

function LogDensityProblems.logdensity(prob::LogGaussianCoxProcess, f_white)
    (; area, counts, gp_mean, gp_cov_chol, logjac) = prob
    f    = gp_cov_chol * f_white + gp_mean
    ℓp_f = logpdf(MvNormal(Zeros(length(f)), I), f_white)
    ℓp_y = sum(@. f * counts - area * exp(f))
    return ℓp_f + ℓp_y + logjac
end

function LogDensityProblems.capabilities(::Type{<:LogGaussianCoxProcess})
    return LogDensityProblems.LogDensityOrder{0}()
end

LogDensityProblems.dimension(prob::LogGaussianCoxProcess) = length(prob.gp_mean)

function LogGaussianCoxProcess()
    n_grid_points      = 40
    grid_length        = 1.0
    area               = 1/(n_grid_points^2)
    cell_boundaries_1d = range(0, grid_length; length=n_grid_points + 1)

    data = readdlm(
        joinpath(@__DIR__, "data", "pines.csv"), ',', Float64, '\n'; header=true
    )

    pine_coords   = data[1][:, 2:3]
    pine_coords_x = pine_coords[:, 1]
    pine_coords_y = pine_coords[:, 2]

    hist            = fit(Histogram, (pine_coords_x, pine_coords_y), (cell_boundaries_1d, cell_boundaries_1d))
    counts_2d       = hist.weights
    grid_1d_unit    = Float64.(0:(n_grid_points - 1))
    grid_2d_unit    = collect(Iterators.product(grid_1d_unit, grid_1d_unit))
    coordinates_tup = reshape(grid_2d_unit, :)
    coordinates     = hcat([[coords[1], coords[2]] for coords in coordinates_tup]...)
    counts_1d       = reshape(counts_2d, :)

    σ2     = 1.91
    β      = 1 / 33
    μ0     = log(126) - σ2 / 2
    kernel = σ2 * KernelFunctions.compose(ExponentialKernel(), ScaleTransform(1 / (n_grid_points * β)))

    K       = kernelmatrix(kernel, coordinates; obsdim=2)
    K_chol  = cholesky(K).L
    gp_mean = Fill(μ0, n_grid_points^2)
    logjac  = 0 #-logdet(K_chol)

    return LogGaussianCoxProcess{
        typeof(area),typeof(counts_1d),typeof(gp_mean),typeof(K_chol),typeof(logjac)
    }(
        area, counts_1d, gp_mean, K_chol, logjac
    )
end


# p1 = LogGaussianCoxProcess()
# dims = LogDensityProblems.dimension(p1)
# LogDensityProblems.logdensity(p1, randn(dims))

# target_ad = ADgradient(AutoMooncake(; config = Mooncake.Config()), p1; x = randn(dims))

# l, g = LogDensityProblems.logdensity_and_gradient(target_ad, randn(dims))

function _load_lgcp()
    target = LogGaussianCoxProcess()
    ad = AutoMooncake(; config = Mooncake.Config())
    dims = LogDensityProblems.dimension(target)
    # target_ad = ADgradient(ad, target; x = randn(dims))
    return target, dims, ad 
end


# using BenchmarkTools

# Bl = []
# BBg = []
# Bg = []
# Ds = []
# for name in ["TReg", "Brownian", "SparseRegression", "LGCP"]
#     @info "Loading $name"
#     prob, dims, ad = load_model(name)
#     @info "Loaded $name with dimension $dims"

#     # ad = AutoMooncake(; config = Mooncake.Config())
#     prob_ad = ADgradient(ad, prob; x = randn(dims))

#     bl = @belapsed LogDensityProblems.logdensity(prob, randn(dims))
#     bg = @belapsed LogDensityProblems.logdensity_and_gradient(prob_ad, randn(dims))

#     push!(Ds, dims)
#     push!(Bl, bl)
#     push!(Bg, bg)
#     push!(BBg, bl*dims)
# end

# using Plots
# plot(Ds, Bg./Bl, label="Gradient_time/LogDensity_time", title="Gradient/LogDensity vs Dimension", xlabel="Dimension", ylabel="Ratio")
