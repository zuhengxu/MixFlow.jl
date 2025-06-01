function sparse_reg_data_load(dat)
    Dat = CSV.read(dat, DataFrame; header = 0)
    # turn String31 into Float64
    for c in names(Dat)
        if eltype(Dat[!,c]) != Float64
            Dat[!, c] = parse.(Float64, Dat[!, c])
        end
    end
    Zr, Y = Dat[:, 1:end-1], Dat[:,end]
    N, p = size(Zr)
    Z = hcat(ones(N), Zr)

    # subsample to 50 to make posterior weird looking
    return Matrix(Z[1:50, :]), Y[1:50], N, p
end

struct SparseRegression{
    Feature<:Matrix{<:Real}, Response<:Vector{<:Real}, Size<:Int, D<:Int
}
    fs::Feature
    rs::Response
    N::Size
    d::D
end

function SparseRegression()
    fs, rs, N, p = sparse_reg_data_load(joinpath(@__DIR__, "data/super_st.csv"))
    return SparseRegression{
        typeof(fs), typeof(rs), typeof(N), typeof(p)
    }(fs, rs, N, p+2)
end

LogDensityProblems.dimension(prob::SparseRegression) = prob.d

function LogDensityProblems.logdensity(prob::SparseRegression, z::AbstractVector)
    (; fs, rs, N, d) = prob

    # s := logσ²
    s, β = z[1], @view(z[2:end])
    ℓprior_s = -0.5 * (s^2) - 0.5 * log(2π)

    Tβ = β.^2.0.*[-50.0 -0.005] .- [log(0.1) log(10)]
    ℓprior_β = sum(LogExpFunctions.logsumexp(Tβ; dims =2)) - (d-1)*(0.5*log(2π) +log(2.0))  

    #  β:= β₁,...,βₚ,β_{p+1}
    diffs = rs .- fs * β
    ℓ_lik = -0.5*exp(-s)*sum(abs2, diffs) - 0.5*N*(log(2π) + s)
    return ℓ_lik + ℓprior_β + ℓprior_s
end

function ∇ℓ_prior_β_single(::SparseRegression, β)
    b = exp(-(50.0 - 5e-3)*β^2.0)
    exponent = log1p(1e6*b) - 2.0*log(10.0) - log1p(1e2*b)
    return -β*exp(exponent)
end

function LogDensityProblems.logdensity_and_gradient(prob::SparseRegression, z)
    (; fs, rs, N, d) = prob

    # s := logσ²
    s, β = z[1], @view(z[2:end])
    ℓprior_s = -0.5 * (s^2) - 0.5 * log(2π)

    Tβ = β.^2.0.*[-50.0 -0.005] .- [log(0.1) log(10)]
    ℓprior_β = sum(LogExpFunctions.logsumexp(Tβ; dims =2)) - (d-1)*(0.5*log(2π) +log(2.0))  

    #  β:= β₁,...,βₚ,β_{p+1}
    diffs = rs .- fs * β
    ℓ_lik = -0.5*exp(-s)*sum(abs2, diffs) - 0.5*N*(log(2π) + s)


    a = 0.5 * exp(-s)
    gsl = a * sum(abs2, diffs) - 0.5 * N 
    gβl = fs' * diffs .* 2a
    gs = gsl - s
    gb = gβl .+ ∇ℓ_prior_β_single.(Ref(prob), β)

    return ℓ_lik + ℓprior_β + ℓprior_s, vcat(gs, gb)
end

# p1 = SparseRegression()
# ad = AutoMooncake(; config = Mooncake.Config())
# dims = LogDensityProblems.dimension(p1)
# t_ad = ADgradient(ad, p1; x = randn(dims))

# x = randn(dims)

# l, g = LogDensityProblems.logdensity_and_gradient(t_ad, x)
# ll, gg = LogDensityProblems.logdensity_and_gradient(p1, x)


Zygote.@adjoint function LogDensityProblems.logdensity(prob::SparseRegression, z)
    ℓ, ∇ℓ = LogDensityProblems.logdensity_and_gradient(prob, z)
    lr_logpdf_pullback(x̄) = (nothing, ∇ℓ * x̄)
    return ℓ, lr_logpdf_pullback
end
Zygote.refresh()


function _load_sparse_reg()
    target = SparseRegression()
    dims = LogDensityProblems.dimension(target)
    ad = AutoZygote()
    return target, dims, ad 
end
