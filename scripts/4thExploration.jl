using DrWatson
@quickactivate "HumanLensClassifyingPosteriors" # <- project name
cd(projectdir())
using Turing
using Plots, StatsPlots, Distributions, LaTeXStrings

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

##
# Set the true probability of heads in a coin.
p_true = 0.8

# Iterate from having seen 0 observations to 100 observations.
Ns = 0:1000;
##
# Draw data from a Bernoulli distribution, i.e. draw heads or tails.
Random.seed!(12)
data = rand(Bernoulli(p_true), last(Ns))

# Here's what the first five coin flips look like:
data[1:5]
##
# Our prior belief about the probability of heads in a coin toss.
prior_belief = Beta(1, 1);
plot(prior_belief,
    size = (500, 250),
    title = "Prior belief",
    xlabel = "probability of heads",
    ylabel = "",
    legend = nothing,
    xlim = (0,1),
    fill=0, α=0.3, w=3)
vline!([p_true])

##
# Make an animation.
animation = @gif for (i, N) in enumerate(Ns)

    # Count the number of heads and tails.
    heads = sum(data[1:i-1])
    tails = N - heads

    # Update our prior belief in closed form (this is possible because we use a conjugate prior).
    updated_belief = Beta(prior_belief.α + heads, prior_belief.β + tails)

    # Plotting
    plot(updated_belief,
        size = (500, 250),
        title = "Updated belief after $N observations",
        xlabel = "probability of heads",
        ylabel = "",
        legend = nothing,
        xlim = (0,1),
        fill=0, α=0.3, w=3)
    vline!([p_true])
end

##
@model function coinflip(y)

    # Our prior belief about the probability of heads in a coin.
    p ~ Beta(1, 1)

    # The number of observations.
    N = length(y)
    for n in 1:N
        # Heads or tails of a coin are drawn from a Bernoulli distribution.
        y[n] ~ Bernoulli(p)
    end
end;
##
mean(data)
# Settings of the Hamiltonian Monte Carlo (HMC) sampler.
iterations = 2000
ϵ = 0.05
τ = 10

# Start sampling.
chain = sample(coinflip(data), HMC(ϵ, τ), iterations, progress=false);
##
# Compute the posterior distribution in closed-form.
N = length(data)
heads = sum(data)
updated_belief = Beta(prior_belief.α + heads, prior_belief.β + N - heads)

# Visualize a blue density plot of the approximate posterior distribution using HMC (see Chain 1 in the legend).
p = plot(p_summary, seriestype = :density, xlim = (0,1), legend = :best, w = 2, c = :blue)

# Visualize a green density plot of posterior distribution in closed-form.
plot!(p, range(0, stop = 1, length = 100), pdf.(Ref(updated_belief), range(0, stop = 1, length = 100)),
        xlabel = "probability of heads", ylabel = "", title = "", xlim = (0,1), label = "Closed-form",
        fill=0, α=0.3, w=3, c = :lightgreen)

# Visualize the true probability of heads in red.
vline!(p, [p_true], label = "True probability", c = :red)