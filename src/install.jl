@info "Installing dependencies"

using Pkg
packages = ["HTTP",
            "JSON3",
            "CSV",
            "DataFrames",
            "JLD2",
            "Flux",
            "MLJ",
            "MLJFlux",
            "NNlib",
            "Optimisers",
            "Plots",
            "Sockets",
            "StatsBase"]
Pkg.add(packages)

@info "Done"