@info "Installing dependencies"

using Pkg
packages = ["CSV",
            "DataFrames",
            "DSP",
            "FFTW",
            "Flux",
            "HTTP",
            "JLD2",
            "JSON3",
            "MLJ",
            "MLJFlux",
            "NNlib",
            "Optimisers",
            "Plots",
            "ProgressMeter",
            "Random",
            "StatsBase"]
Pkg.add(packages)
