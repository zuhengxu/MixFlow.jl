#!/usr/bin/env julia --threads=1 --project=example

ENV["JULIA_PKG_PRECOMPILE_AUTO"]=0
using Pkg
Pkg.instantiate()
