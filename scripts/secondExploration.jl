using DrWatson
@quickactivate "HumanLensClassifyingPosteriors" # <- project name
cd(projectdir())
using DataFrames
# using Turing, Distributions
using StatsPlots
using LaTeXStrings
using Random
using Markdown
using Latexify, Plots

unicodeplots()
##
groundTruth = Dict("lenses" => 500.0,"nonlenses" => 2500.0)
groundTruth["ratio"] = groundTruth["lenses"] / groundTruth["nonlenses"]
groundTruth["total"] = groundTruth["lenses"] + groundTruth["nonlenses"]
groundTruth
humans = DataFrame("TP" => [45,140,260,300,490], "TN" => [350, 1500, 1750, 2200, 2490])
humans.FN = groundTruth["lenses"] .- humans.TP
humans.FP = groundTruth["nonlenses"] .- humans.TN
humans.rate = humans.TP ./(humans.TP .+humans.FN)
humans.rateL2NL = (humans.TP .+ humans.FP) ./(humans.FP .+ humans.TN .+ humans.TP .+humans.FN)
humans.likepos = humans.TP ./ groundTruth["lenses"]
humans.likeneg = humans.FN ./ groundTruth["lenses"]
humans.pos = (humans.TP .+ humans.FP) ./ groundTruth["total"]
humans.neg = (humans.TN .+ humans.FN) ./ groundTruth["total"]
humans

posterior = "p(L|{i = y_i}) = \\frac{p(L) ∏_i^N_obs p(i=y_i )}{123}"
m = Math("\\frac{1}{1 + x}");
# posterior = L"p(L|y_i"
# latexify(posterior)
##
plot(framestyle = :none) 
title!(latexify(posterior), titlefontsize=10)
