#!/usr/bin/env julia --threads=10 --project=combine_env

using Pkg 
Pkg.offline(true) 
Pkg.precompile()
