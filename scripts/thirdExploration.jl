using DrWatson
@quickactivate "HumanLensClassifyingPosteriors" # <- project name
cd(projectdir())
using Turing

@model dice_throw(y) = begin
    #Our prior belief about the probability of each result in a six-sided dice.
    #p is a vector of length 6 each with probability p that sums up to 1.
    p ~ Dirichlet(6, 1)

    #Each outcome of the six-sided dice has a probability p.
    y ~ filldist(Categorical(p), length(y))
end;