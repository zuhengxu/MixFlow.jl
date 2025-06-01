neg_sigmoid(x) = -1.0/(1.0 + exp(-x))

struct LogisticRegression{XT, YT}
    X::XT
    y::YT
end

function LogDensityProblems.logdensity(prob::LogisticRegression, θ::AbstractVector)
    (; X, y) = prob

    ℓp_θ   = mapreduce(normlogpdf, +, θ)

    logits = X*θ
    ℓp_y   = sum(@. logpdf(BernoulliLogit(logits), y))

    ℓp_θ + ℓp_y
end


function LogDensityProblems.logdensity_and_gradient(prob::LogisticRegression, θ)
    (; X, y) = prob
    D = size(X, 2)
    logits = X*θ

    ℓp_θ   = mapreduce(normlogpdf, +, θ)
    ℓp_y   = sum(@. logpdf(BernoulliLogit(logits), y))
    ℓ =  ℓp_θ + ℓp_y

    p = neg_sigmoid.(logits)
    @tullio M[j] := X[n,j]*(p[n] + y[n])
    ∇ℓ = -θ .+ M
    return ℓ, ∇ℓ
end


# use customized adjoint with Zygote, faster than mooncake (13μs vs 19μs)
function _mygrad(prob::LogisticRegression, θ)
    (; X, y) = prob
    D = size(X, 2)

    logits = X*θ
    p = neg_sigmoid.(logits)
    @tullio M[j] := X[n,j]*(p[n] + y[n])
    return -θ .+ M
end

Zygote.@adjoint function LogDensityProblems.logdensity(prob::LogisticRegression, θ)
      lr_logpdf_pullback(x̄) = (nothing, _mygrad(prob, θ) * x̄)
    return LogDensityProblems.logdensity(prob, θ), lr_logpdf_pullback
end
Zygote.refresh()

function LogDensityProblems.capabilities(::Type{<:LogisticRegression})
    return LogDensityProblems.LogDensityOrder{1}()
end

LogDensityProblems.dimension(prob::LogisticRegression) = size(prob.X, 2)

function preprocess_features(X::AbstractMatrix)
    μ = mean(X, dims=1)
    σ = std(X, dims=1)
    σ[σ .<= eps(Float64)] .= 1.0
    X = (X .- μ) ./ σ
    hcat(X, ones(size(X, 1), 1))
end

function LogisticRegressionSonar()
    data   = readdlm(joinpath(@__DIR__, "data/sonar.csv"), ',', Any, '\n')
    X      = convert(Matrix{Float64}, data[:, 1:end-1])
    y      = data[:, end] .== "R"
    X_proc = preprocess_features(X)
    LogisticRegression(X_proc, y)
end

# p1 = LogisticRegressionSonar()
# D = LogDensityProblems.dimension(p1)
# LogDensityProblems.logdensity_and_gradient(p1, randn(D))[2]

function _load_sonar()
    model_ad = LogisticRegressionSonar()
    dims = LogDensityProblems.dimension(model_ad)
    ad = AutoZygote()
    return model_ad, dims, ad
end



