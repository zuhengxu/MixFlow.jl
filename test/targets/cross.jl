"""
    Cross(μ::Real=2.0, σ::Real=0.15)

2-dimensional Cross distribution

# Reference
[1] Zuheng Xu, Naitong Chen, Trevor Campbell
"MixFlows: principled variational inference via mixed flows."
International Conference on Machine Learning, 2023
"""
Cross() = Cross(2.0, 0.15)
function Cross(μ::T, σ::T) where {T<:Real}
    return MixtureModel([
        MvNormal([zero(μ), μ], [σ, one(σ)]),
        MvNormal([-μ, one(μ)], [one(σ), σ]),
        MvNormal([μ, one(μ)], [one(σ), σ]),
        MvNormal([zero(μ), -μ], [σ, one(σ)]),
    ])
end
