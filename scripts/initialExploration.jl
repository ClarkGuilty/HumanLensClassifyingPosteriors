using DrWatson
@quickactivate "HumanLensClassifyingPosteriors" # <- project name
cd(projectdir())
using DataFrames
using Turing, Distributions
using StatsPlots
using LaTeXStrings
using Random
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

##

plt = plot()
for i in 1:1:size(humans,1)
    plot!(plt,Beta(humans[i,"TP"],humans[i,"FN"]),
        label="Observer "*string(i)*", a="*string(humans[i,"TP"])*", b="*string(humans[i,"FN"]),
        lw=5,
        xlabel=L"\theta",
        ylabel="Density",
        xlims=(0, 1)
)
end
display(plt)
##

@model function coinflip(y) #Given just the first observer.
    # Our prior belief about the probability that a cutout is a lens.
    p ~ Beta(1, 1)

    # The number of observations.
    N = length(y)
    for n in 1:N
        # Wheter is a lens or not is drawned by a Bernoulli distribution.
        y[n] ~ Bernoulli(p)
    end
end;

#answers of the human classifiers

i=1
data = rand(Bernoulli(humans[1,"rate"]), last(2000))
data = rand(Beta(humans[i,"TP"],humans[i,"FN"]), last(2000))
# Settings of the Hamiltonian Monte Carlo (HMC) sampler.
iterations = 100
ϵ = 0.05
τ = 10

# Start sampling.
chain = sample(coinflip(data), HMC(ϵ, τ), iterations)

# Plot a summary of the sampling process for the parameter p, i.e. the probability of heads in a coin.
histogram(chain[:p])
histogram(data)