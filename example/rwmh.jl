using Random, Distributions, Plots
using LinearAlgebra
using Base.Threads: @threads
using JLD2
using LogExpFunctions
import PlotlyJS as pjs

include("Model.jl")


# the involution map
f(x, v) = (v, x)

check_acc(ua, loga) = log(ua) > loga ? false : true
update_u(u) = (u + π/4) % 1
inv_update_u(u) = (u + 1 - π/4) % 1

function rwmh(logp, x, v, uv, ua)
    uv = update_u(uv)
    ua = update_u(ua)

    # joint logp
    logp_joint(x, v) = logp(x) + logpdf(Normal(x), v)

    ṽ = quantile(Normal(x), uv)
    uv_ = cdf(Normal(x), v)
    x_, v_ = f(x, ṽ)
    logr = logp_joint(x_, v_) - logp_joint(x, ṽ) 
    loga = min(0, logr)

    if check_acc(ua, loga)
        ua = exp(log(ua) - logr) 
        logabsjac = logp_joint(x, v) - logp_joint(x_, v_)
        return x_, v_, uv_, ua, logabsjac
    else
        logabsjac = logp_joint(x, v) - logp_joint(x, ṽ)
        return x, ṽ, uv_, ua, logabsjac
    end
end

function rwmh_inv(logp, x_, v_, uv, ua)
    logp_joint(x, ṽ) = logp(x) + logpdf(Normal(x), ṽ)

    x, ṽ = f(x_, v_)
    logr = logp_joint(x_, v_) - logp_joint(x, ṽ)

    uã = exp(log(ua) + logr)

    if uã > 1
        x, ṽ = x_, v_
    else
        ua = exp(log(ua) + logr)
    end

    v = quantile(Normal(x), uv)
    uv = cdf(Normal(x), ṽ)
    
    uv = inv_update_u(uv)
    ua = inv_update_u(ua)

    logabsjac = logp_joint(x_, v_) - logp_joint(x, v)
    return x, v, uv, ua, logabsjac
end

# gaussian target
name = "Cauchy1D"
target = load_model(name)
ℓπ(x) = logpdf(target, x)

# check invertibility
function inv_error_rwmh(ℓπ, T)
    x0 = randn()
    v0 = randn() + x0
    uv0 = rand()
    ua0 = rand()

    x,v, uv, ua = copy(x0), copy(v0), copy(uv0), copy(ua0)
    for i in 1:T
        x, v, uv, ua, _ = rwmh(ℓπ, x, v, uv, ua)
    end

    for i in 1:T
        x, v, uv, ua, _ = rwmh_inv(ℓπ, x, v, uv, ua)
    end

    err = [x, v, uv, ua] .- [x0, v0, uv0, ua0]
    return norm(err)
end

Ts = 10 .^[1:6 ;]
ers = [inv_error_rwmh(ℓπ, T) for T in Ts]
plot(Ts, ers, xaxis=:log, yaxis=:log, label="inv Error", linewidth=3)
savefig("figure/$(name)_inv_error.png")


# check validity of the forward pass of involutive mcmc
x0 = -3.0
v0 = randn() + x0
uv0 = rand()
ua0 = rand()
x,v, uv, ua = copy(x0), copy(v0), copy(uv0), copy(ua0)

T = 10000
xs = []
for i in 1:T
    x, v, uv, ua, _ = rwmh(ℓπ, x, v, uv, ua)
    push!(xs, x)
end

# mean(xs)

# visualize the histogram and density plot
histogram(xs, nbins=500, label="mcmc sample Histogram", alpha=0.5, normed=true)
plot!(range(-10, 10; length=300), x -> pdf(target, x), label="True density", linewidth=2)
# bound the range of the plot
plot!(xlims=(-10, 10))
# put a vertical line at the x0
vline!([x0], label="Initial x0", color=:red)
savefig("figure/$(name)_sample.png")

# check learned density
function log_density_rwmh(logp, T, x, v, uv, ua)
    # joint logp
    logp_joint(x, v) = logp(x) + logpdf(Normal(x), v)
    x0, v0 = copy(x), copy(v)
    logJ = 0.0
    ℓs = []
    for t in 1:T
        x, v, uv, ua, logjac = rwmh_inv(ℓπ, x, v, uv, ua)
        # println(x, v, logjac)
        # if isnan(x) || isnan(v)
        #     println("nan")
        # end
        logJ += logjac
        l = logp_joint(x, v) + logJ
        push!(ℓs, l)
    end
    # logJ = logp_joint(x0, v0) - logp_joint(x, v) 
    return logsumexp(ℓs) - log(T)
end
# log_density_rwmh(ℓπ, T, x0, v0, uv0, ua0)


xs_eval = [-5:0.2:6 ;]
vs_eval = [-5:0.2:6 ;]

logpdfs_true = zeros(length(xs_eval), length(vs_eval))
@threads for i in 1:length(xs_eval)
    for j in 1:length(vs_eval)
        x, v = xs_eval[i], vs_eval[j]
        logpdfs_true[i, j] = ℓπ(x) + logpdf(Normal(x), v)
    end
end

logpdfs_est = zeros(length(xs_eval), length(vs_eval))
uv0 = rand()
ua0 = rand()
T = 500
@threads for i in 1:length(xs_eval)
    for j in 1:length(vs_eval)
        x, v = xs_eval[i], vs_eval[j]
        logpdfs_est[i, j] = log_density_rwmh(ℓπ, T, x, v, uv0, ua0)
    end
end

if !isdir("result")
    mkdir("result")
end
JLD2.save("result/$(name)_lpdfs.jld2", "target", logpdfs_true, "est", logpdfs_est)


layout = pjs.Layout(
    width=500, height=500,
    scene = pjs.attr(
    xaxis = pjs.attr(showticklabels=false, visible=false),
    yaxis = pjs.attr(showticklabels=false, visible=false),
    zaxis = pjs.attr(showticklabels=false, visible=false),
    ),
    margin=pjs.attr(l=0, r=0, b=0, t=0, pad=0),
    colorscale = "Vird"
)

p_target = pjs.plot(pjs.surface(x=xs_eval, y=vs_eval, z=logpdfs_true, showscale = true), layout)
pjs.savefig(p_target, "figure/$(name)_lpdf.png")

p_est = pjs.plot(pjs.surface(x=xs_eval, y=vs_eval, z=logpdfs_est, showscale = true), layout)
pjs.savefig(p_est, "figure/$(name)_lpdf_est.png")
