"""
Julia toolbox for predicting clozapine (CLO) and norclozapine (NCLO) blood concentrations

https://clotop.eu
"""
module CloToP

@info "Loading packages"

using Pkg
using CSV
using DataFrames
using JLD2
using MLJ
using MLJFlux
using NNlib
using Flux
using Random
using Plots
using StatsBase

include("predict.jl")

@info "Loading data"

# load training data
if isfile("data/clozapine_test.csv")
    println("Loading: clozapine_test.csv")
    test_data = CSV.read("data/clozapine_test.csv", header=true, DataFrame)
else
    error("File data/clozapine_test.csv cannot be opened!")
    exit(-1)
end

# load models
if isfile("models/norclozapine_regressor_model.jlso")
    println("Loading: norclozapine_regressor_model.jlso")
    nclo_model_regressor = machine("models/norclozapine_regressor_model.jlso")
else
    error("File models/norclozapine_regressor_model.jlso cannot be opened!")
    exit(-1)
end
if isfile("models/clozapine_regressor_model.jlso")
    println("Loading: clozapine_regressor_model.jlso")
    clo_model_regressor = machine("models/clozapine_regressor_model.jlso")
else
    error("File models/clozapine_regressor_model.jlso cannot be opened!")
    exit(-1)
end
if isfile("models/scaler_clo.jld")
    println("Loading: scaler_clo.jld")
    scaler_clo = JLD2.load_object("models/scaler_clo.jld")
else
    error("File models/scaler_clo.jld cannot be opened!")
    exit(-1)
end
if isfile("models/scaler_nclo.jld")
    println("Loading: scaler_nclo.jld")
    scaler_nclo = JLD2.load_object("models/scaler_nclo.jld")
else
    error("File models/scaler_nclo.jld cannot be opened!")
    exit(-1)
end

end # CloToP
