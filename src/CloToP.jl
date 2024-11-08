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

@info "Loading models"

@assert isfile("models/clozapine_regressor_model.jlso") "File models/clozapine_regressor_model.jlso cannot be opened!"
println("Loading: clozapine_regressor_model.jlso")
clo_model_regressor = machine("models/clozapine_regressor_model.jlso")

@assert isfile("models/norclozapine_regressor_model.jlso") "File models/norclozapine_regressor_model.jlso cannot be opened!"
println("Loading: norclozapine_regressor_model.jlso")
nclo_model_regressor = machine("models/norclozapine_regressor_model.jlso")

@assert isfile("models/scaler_clo.jld") "File models/scaler_clo.jld cannot be opened!"
println("Loading: scaler_clo.jld")
scaler_clo = JLD2.load_object("models/scaler_clo.jld")

@assert isfile("models/scaler_nclo.jld") "File models/scaler_nclo.jld cannot be opened!"
println("Loading: scaler_nclo.jld")
scaler_nclo = JLD2.load_object("models/scaler_nclo.jld")

include("predict.jl")

end # CloToP
