using MixFlow: _ergodic_shift
using Random, Distributions, Plots
using LinearAlgebra
using Base.Threads: @threads
using JLD2
using LogExpFunctions
using LogDensityProblems
using MixFlow

include("Model.jl")
MixFlow.iid_sample(dist::ContinuousDistribution, n::Int) = rand(dist, n)
MixFlow.iid_sample(dist::ContinuousDistribution) = rand(dist)

name = "Mixture1D"
target = load_model(name)
reference = Normal()
prob = MixFlowProblem(reference, target)

K = RWMH1D()
T = 1000
mixer = RandomShift1D(T)

mx = ErgodicShift1D(T)


# ξs = rand(T)

# u0 = rand()


# function bwd_process(u0, ξs, T)
#     u = u0
#     for t in T:-1:1
#         u = _ergodic_shift(u, ξs[t])
#     end
#     return u
# end

# ub = bwd_process(u0, ξs, T)

# ubs = [bwd_process(u0, ξs, t) for t in 1:T]




x0 = randn()
v0 = x0 + randn()
uv0 = rand()
ua0 = rand()

x, v, uv, ua = x0, v0, uv0, ua0

for t in 1:T
    x, v, uv, ua = forward(prob, K, mixer, x, v, uv, ua, t)
end

for t in T:-1:1
    x, v, uv, ua = inverse(prob, K, mixer, x, v, uv, ua, t)
end

[x, v, uv, ua] .- [x0, v0, uv0, ua0]


function backward_process(prob, K, mixer, x0, v0, uv0, ua0, T)
    x, v, uv, ua = x0, v0, uv0, ua0
    for t in T:-1:1
        x, v, uv, ua = forward(prob, K, mixer, x, v, uv, ua, t)
    end
    return x, v, uv, ua
end

x0 = randn()
v0 = x0 + randn()
uv0 = rand()
ua0 = rand()

xs = []
vs = []
uvs = []
uas = []
for i in 1:T
    x, v, uv, ua = backward_process(prob, K, mixer, x0, v0, uv0, ua0, i)
    push!(xs, x)
    push!(vs, v)
    push!(uvs, uv)
    push!(uas, ua)
end


# diaconis example
T = 20
us = rand(T)
as = rand(Bernoulli(0.5), T)

ϕ(x, a, u) = a == 0 ? u*(1-x) : u*x

function ϕ_bwd(x0, as, us, T)
    xs = []
    for t in 1:T
        x = x0
        for i in t:-1:1
            x = ϕ(x, as[i], us[i])
        end
        push!(xs, x)
    end
    return xs
end

xs1 = ϕ_bwd(rand(), as, us, T)
xs2 = ϕ_bwd(rand(), as, us, T)
xs3 = ϕ_bwd(rand(), as, us, T)

plot(xs1, label="1")
plot!(xs2, label="2")
plot!(xs3, label="3")
