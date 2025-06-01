function t_reg_data_load(dat)
    Dat = CSV.read(dat, DataFrame; header = 1)
    Zr, Y = Dat[:, 1:end-1], Dat[:,end]
    Zr[:, 3] .= 140 .- Zr[:,3]
    Zr = log.(Zr)
    N, p = size(Zr)
    Zr = Matrix(Zr)
    # standarize dataset 
    Zr = (Zr .- mean(Zr, dims =1)) ./ std(Zr, dims = 1)
    Y = (Y .- mean(Y))./std(Y)
    # add intercept
    Z = hcat(ones(N), Zr)
    return Matrix(Z), Y, N, p +1
end

##################
# load and process dataset
#################3

struct StudentTRegression{
    Feature<:Matrix{<:Real}, Response<:Vector{<:Real}, Size<:Int, D<:Int
}
    X::Feature
    Y::Response
    N::Size
    d::D
end

function StudentTRegression()
    X, Y, N, d = t_reg_data_load(joinpath(@__DIR__, "data/creatinine.csv"))
    return StudentTRegression{
        typeof(X), typeof(Y), typeof(N), typeof(d)
    }(X, Y, N, d)
end

    
#######################
# models ( t₅(Xβ, 1)-distirbution lin reg with heavy tailed prior, β ∼ Cauchy(0, 1))
#######################
function LogDensityProblems.logdensity(prob::StudentTRegression, β::AbstractVector)
    (; X, Y, N, d) = prob
    
    ℓp_β = -d*log(π) - sum(log1p, β.^2.0)
    #  β:= β₁,...,βₚ,β_{p+1}
    diffs = Y .- X * β
    ℓ_lik = -3.0*sum(log, 5.0 .+ diffs.^2.0) 
    return ℓ_lik + ℓp_β 
end

function LogDensityProblems.logdensity_and_gradient(prob::StudentTRegression, β)
    (; X, Y, N, d) = prob
    
    ℓp_β = -d*log(π) - sum(log1p, β.^2.0)

    #  β:= β₁,...,βₚ,β_{p+1}
    diffs = Y .- X * β
    A = 5.0 .+ diffs.^2.0

    ℓ_lik = -3.0*sum(log, A) 
    ℓ =  ℓ_lik + ℓp_β

    ∇ℓp_β = @. -2.0*β/(1.0 + β^2.0)
    ∇ℓ_lik = X' * (6.0*diffs./A) 
    
    return ℓ, ∇ℓ_lik .+ ∇ℓp_β
end

# use customized adjoint with Zygote, faster than mooncake (707ns vs 2.7μs)
Zygote.@adjoint function LogDensityProblems.logdensity(prob::StudentTRegression, β)
    ℓ, ∇ℓ = LogDensityProblems.logdensity_and_gradient(prob, β)
    lr_logpdf_pullback(x̄) = (nothing,  ∇ℓ* x̄)
    return ℓ, lr_logpdf_pullback
end

Zygote.refresh()

function LogDensityProblems.capabilities(::Type{<:StudentTRegression})
    return LogDensityProblems.LogDensityOrder{1}()
end

LogDensityProblems.dimension(prob::StudentTRegression) = size(prob.X, 2)

# p1 = StudentTRegression()
# dims = LogDensityProblems.dimension(prob)

# ad = AutoMooncake(; config = Mooncake.Config())

# t_ad = ADgradient(ad, p1; x = randn(dims))

# xt = randn(dims)
# l, g = LogDensityProblems.logdensity_and_gradient(t_ad, xt)
# ll, gg = LogDensityProblems.logdensity_and_gradient(p1, xt)

# @btime LogDensityProblems.logdensity_and_gradient(p1, randn(dims))
# @btime LogDensityProblems.logdensity_and_gradient(t_ad, randn(dims))

function _load_t_reg()
    ad = AutoZygote()
    model_ad = StudentTRegression()
    dims = LogDensityProblems.dimension(model_ad)
    return model_ad, dims, ad
end
