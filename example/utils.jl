using LinearAlgebra

using MixFlow


function _get_kernel_name(K::MixFlow.InvolutiveKernel)
    str = string(typeof(K))
    name = split(str, "{", limit = 2)[1]
    return name
end

function running_mean(xs::Matrix{T}) where T
    cumsum(xs; dims = 2) ./ [1:size(xs, 2) ;]'
end
function running_mean(xs::Vector{T}) where T
    cumsum(xs) ./ [1:length(xs) ;]
end

function running_square(xs::Matrix{T}) where T
    cumsum(xs.^2, dims = 2) ./ [1: size(xs, 2) ;]'
end

