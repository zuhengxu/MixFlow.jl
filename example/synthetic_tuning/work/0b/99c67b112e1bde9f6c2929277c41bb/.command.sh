#!/usr/bin/env julia --threads=10 --project=example

using Pkg 
Pkg.offline(true) 
Pkg.precompile()
