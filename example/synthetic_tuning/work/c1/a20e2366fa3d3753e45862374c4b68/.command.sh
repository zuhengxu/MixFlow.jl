#!/usr/bin/env julia --threads=1 --project=combine_env

using CSV 
using CombineCSVs

mkdir("output")
combine_csvs(".", "output"; comment = "#")
