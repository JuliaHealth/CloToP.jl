@info "Importing packages.."

using Pkg
# packages = ["CSV", "DataFrames", "JLD2", "Flux", "MLJ", "MLJFlux", "NNlib", "Optimisers", "Plots", "ProgressMeter", "StatsBase"]
# Pkg.add(packages)

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

m = Pkg.Operations.Context().env.manifest
println("       CSV $(m[findfirst(v -> v.name == "CSV", m)].version)")
println("DataFrames $(m[findfirst(v -> v.name == "DataFrames", m)].version)")
println("      JLD2 $(m[findfirst(v -> v.name == "JLD2", m)].version)")
println("       MLJ $(m[findfirst(v -> v.name == "MLJ", m)].version)")
println("   MLJFlux $(m[findfirst(v -> v.name == "MLJFlux", m)].version)")
println("      Flux $(m[findfirst(v -> v.name == "Flux", m)].version)")
println("     NNlib $(m[findfirst(v -> v.name == "NNlib", m)].version)")
println("Optimisers $(m[findfirst(v -> v.name == "Optimisers", m)].version)")
println("     Plots $(m[findfirst(v -> v.name == "Plots", m)].version)")
println(" StatsBase $(m[findfirst(v -> v.name == "StatsBase", m)].version)")
println()

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

vsearch(y::Real, x::AbstractVector) = findmin(abs.(x .- y))[2]

function ctp(patient_data::Vector{<:Real}, scaler_clo, scaler_nclo)

    data_nclo = deepcopy(patient_data)
    # standaridize
    data_nclo[2:5] = StatsBase.transform(scaler_nclo, reshape(data_nclo[2:5], 1, length(data_nclo[2:5])))
    data_nclo[isnan.(data_nclo)] .= 0

    # create DataFrame
    x1 = DataFrame(:male=>data_nclo[1])
    x2 = DataFrame(reshape(data_nclo[2:5], 1, length(data_nclo[2:5])), ["age", "dose", "bmi", "crp"])
    x3 = DataFrame(reshape(data_nclo[6:end], 1, length(data_nclo[6:end])), ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
    data_nclo = Float32.(hcat(x1, x2, x3))
    data_nclo = coerce(data_nclo, :male=>OrderedFactor{2}, :age=>Continuous, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Continuous, :inhibitors_3a4=>Continuous, :substrates_3a4=>Continuous, :inducers_1a2=>Continuous, :inhibitors_1a2=>Continuous, :substrates_1a2=>Continuous)

    # predict
    nclo_level_pred = MLJ.predict(nclo_model_regressor, data_nclo)

    data_clo = deepcopy(patient_data)
    data_clo = vcat(data_clo[1], nclo_level_pred, data_clo[2:end])

    # standardize
    data_clo[2:6] = StatsBase.transform(scaler_clo, reshape(data_clo[2:6], 1, length(data_clo[2:6])))
    data_clo[isnan.(data_clo)] .= 0

    # create DataFrame
    x1 = DataFrame(:male=>data_clo[1])
    x2 = DataFrame(reshape(data_clo[2:6], 1, length(data_clo[2:6])), ["nclo", "age", "dose", "bmi", "crp"])
    x3 = DataFrame(reshape(data_clo[7:end], 1, length(data_clo[7:end])), ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
    data_clo = Float32.(hcat(x1, x2, x3))
    data_clo = coerce(data_clo, :male=>OrderedFactor{2}, :nclo=>Continuous, :age=>Continuous, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Continuous, :inhibitors_3a4=>Continuous, :substrates_3a4=>Continuous, :inducers_1a2=>Continuous, :inhibitors_1a2=>Continuous, :substrates_1a2=>Continuous)

    clo_level_pred = MLJ.predict(clo_model_regressor, data_clo)

    clo_level = round.(Float64(clo_level_pred[1]), digits=1)
    clo_level < 0 && (clo_level = 0)
    nclo_level = round.(Float64(nclo_level_pred[1]), digits=1)[1]
    nclo_level < 0 && (nclo_level = 0)

    if clo_level > 550
        clo_group = 1
    else
        clo_group = 0
    end
    if clo_level > 550 || nclo_level > 270
        clo_group_adj = 1
    else
        clo_group_adj = 0
    end

    return clo_group, clo_group_adj, clo_level, nclo_level
end

function recommended_dose(patient_data::Vector{<:Real}, scaler_clo, scaler_nclo)

    doses = 0:12.5:800
    clo_concentration = zeros(length(doses))
    nclo_concentration = zeros(length(doses))
    clo_group = zeros(Int64, length(doses))
    clo_group_adjusted= zeros(Int64, length(doses))

    for idx in eachindex(doses)
        data = deepcopy(patient_data)
        data = vcat(data[1:2], doses[idx], data[3:end])
        clo_group[idx], clo_group_adjusted[idx], clo_concentration[idx], nclo_concentration[idx] = predict_single_patient(data, scaler_clo, scaler_nclo)
    end

    return collect(doses), clo_concentration, nclo_concentration, clo_group, clo_group_adjusted
end

function dose_range(doses, clo_concentration, nclo_concentration, clo_group, clo_group_adjusted)

    # therapeutic concentration range: 220-550 ng/mL
    # source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10201335/

    if minimum(clo_concentration) < 220
        min_dose_idx = findfirst(x -> x > 220, clo_concentration)
    else
        min_dose_idx = 1
    end

    if maximum(clo_concentration) > 550
        max_dose_idx = findfirst(x -> x > 550, clo_concentration)
    else
        max_dose_idx = length(doses)
    end

    println("Minimum recommended dose: $(doses[min_dose_idx]) mg/day")
    println("Maximum recommended dose: $(doses[max_dose_idx]) mg/day")

    dose_range = (doses[min_dose_idx], doses[max_dose_idx])

    plot(doses, clo_concentration, ylims=(0, 1000), xlims=(0, 800), legend=false, xlabel="dose [mg/day]", ylabel="clozapine concentration [ng/mL]", margins=20Plots.px)
    hline!([220], lc=:green, ls=:dot)
    hline!([550], lc=:red, ls=:dot)
    vline!([doses[min_dose_idx]], lc=:green, ls=:dot)
    vline!([doses[max_dose_idx]], lc=:red, ls=:dot)
end

# male: 0/1
# age: Float
# dose: Float
# bmi: Float
# crp: Float
# inducers_3a4: Int
# inhibitors_3a4: Int
# substrates_3a4: Int
# inducers_1a2: Int
# inhibitors_1a2: Int
# substrates_1a2: Int

pt = [0, 20, 512.5, 27, 10.5, 0, 0, 0, 0, 0, 0]
ctp(pt, scaler_clo, scaler_nclo)

# male: 0/1
# age: Float
# bmi: Float
# crp: Float
# inducers_3a4: Int
# inhibitors_3a4: Int
# substrates_3a4: Int
# inducers_1a2: Int
# inhibitors_1a2: Int
# substrates_1a2: Int

pt = [0, 65, 28, 0.5, 0, 0, 0, 1, 0, 0]
doses, clo_concentration, nclo_concentration, clo_group, clo_group_adjusted, = recommended_dose(pt, scaler_clo, scaler_nclo)
dose_range(doses, clo_concentration, nclo_concentration, clo_group, clo_group_adjusted)
