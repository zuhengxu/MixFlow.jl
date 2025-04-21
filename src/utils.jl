function logsumexp_scan(X::Vector{T}, Ns::Vector{Int}) where T
    @assert maximum(Ns) == size(X,1)
    L = zeros(size(Ns, 1))
    L[1] = logsumexp(@view(X[1:Ns[1]]))
    @views for i = 2:size(Ns, 1)
        t = logsumexp(X[Ns[i-1]+1:Ns[i]])
        l = logsumexp([t, L[i-1]])
        L[i] = l
    end
    return L
end

function logmeanexp_scan(X::Vector{T}, Ns::Vector{Int}) where T
    Ls = logsumexp_scan(X, Ns)
    return Ls .- log.(Ns)
end

function logsumexp_sweep(X::Vector{T}) where T
```
logsumexp function without memory allocation
adapt from "http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html"
```
    Ls = similar(X)
    alpha = -Inf
    r = 0.0
    for (i, x) in enumerate(X)
        if x <= alpha
            r += exp(x - alpha)
        else
            r *= exp(alpha - x)
            r += 1.0
            alpha = x
        end
        Ls[i] = log(r) + alpha
    end
    return Ls
end

function logmeanexp_sweep(X::AbstractVector{T}) where T
    n = size(X, 1)
    Ls = logsumexp_sweep(X)
    return Ls .- log.([1:n ;])
end
