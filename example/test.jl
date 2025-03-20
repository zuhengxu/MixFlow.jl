using Random, Distributions, Plots
using LinearAlgebra
using Base.Threads: @threads
using JLD2
using LogExpFunctions
using LogDensityProblems
using MixFlow

include("Model.jl")

name = "Cauchy1D"
target = load_model(name)
reference = Normal()
prob = MixFlow.MixFlowProblem(reference, target)


