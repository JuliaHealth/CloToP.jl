@info "Installing dependencies"

using Pkg
packages = ["CSV",
            "DataFrames",
            "Flux",
            "HTTP",
            "JLD2",
            "JSON3",
            "MLJ",
            "MLJFlux",
            "Optimisers",
            "Distributions",
            "Plots",
            "StatsBase"]

Pkg.add(packages)
