struct BrownianMotion{Y, Idx}
    y       :: Y
    obs_idx :: Idx
end

function LogDensityProblems.capabilities(::Type{<:BrownianMotion})
    return LogDensityProblems.LogDensityOrder{2}()
end

function LogDensityProblems.logdensity(prob::BrownianMotion, θ::AbstractVector)
    (; y, obs_idx) = prob
    x     = @view(θ[1:30])
    α_inn = StatsFuns.softplus(θ[31])
    α_obs = StatsFuns.softplus(θ[32])

    ℓjac_α_inn = loglogistic(α_inn)
    ℓjac_α_obs = loglogistic(α_obs)

    ℓp_α_inn = logpdf(LogNormal(0, 2), α_inn)
    ℓp_α_obs = logpdf(LogNormal(0, 2), α_obs)
    ℓp_x1    = logpdf(Normal(0, α_inn), x[1])
    ℓp_x     = logpdf(MvNormal(@view(x[1:end-1]), α_inn), @view(x[2:end]))
    ℓp_y     = logpdf(MvNormal(@view(x[obs_idx]), α_obs), y)

    ℓp_y + ℓp_x1 + ℓp_x + ℓp_α_inn + ℓp_α_obs + ℓjac_α_inn + ℓjac_α_obs
end

LogDensityProblems.dimension(prob::BrownianMotion) = 32

function BrownianMotion()
    y = [
        0.21592641,
        0.118771404,
        -0.07945447,
        0.037677474,
        -0.27885845,
        -0.1484156,
        -0.3250906,
        -0.22957903,
        -0.44110894,
        -0.09830782,       
        #
        -0.8786016,
        -0.83736074,
        -0.7384849,
        -0.8939254,
        -0.7774566,
        -0.70238715,
        -0.87771565,
        -0.51853573,
        -0.6948214,
        -0.6202789,
    ]
    obs_idx = vcat(1:10, 21:30)
    BrownianMotion(y, obs_idx)
end

function _load_brownian()
    target = BrownianMotion()
    ad = AutoMooncake(; config = Mooncake.Config())
    dims = LogDensityProblems.dimension(target)
    # target_ad = ADgradient(ad, target; x = randn(dims))
    return target, dims, ad 
end
