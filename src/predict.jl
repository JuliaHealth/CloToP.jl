@info "Importing packages.."

using Pkg
# packages = ["CSV", "DataFrames", "JLD2", "MLJ", "MLJDecisionTreeInterface", "Plots", "StatsBase"]
# Pkg.add(packages)

using CSV
using DataFrames
using JLD2
using MLJ
using MLJDecisionTreeInterface
using Random
using Plots
using StatsBase

m = Pkg.Operations.Context().env.manifest
println("       CSV $(m[findfirst(v -> v.name == "CSV", m)].version)")
println("DataFrames $(m[findfirst(v -> v.name == "DataFrames", m)].version)")
println("      JLD2 $(m[findfirst(v -> v.name == "JLD2", m)].version)")
println("       MLJ $(m[findfirst(v -> v.name == "MLJ", m)].version)")
println("     Plots $(m[findfirst(v -> v.name == "Plots", m)].version)")
println(" StatsBase $(m[findfirst(v -> v.name == "StatsBase", m)].version)")
println()

@info "Loading data.."

# load models
if isfile("models/clozapine_classifier_model.jlso")
    println("Loading model: clozapine_classifier_model.jlso")
    model_rfc = machine("models/clozapine_classifier_model.jlso")
else
    error("File models/clozapine_classifier_model.jlso cannot be opened!")
    exit(-1)
end
if isfile("models/clozapine_regressor_model.jlso")
    println("Loading model: clozapine_regressor_model.jlso")
    clo_model_rfr = machine("models/clozapine_regressor_model.jlso")
else
    error("File models/clozapine_regressor_model.jlso cannot be opened!")
    exit(-1)
end
if isfile("models/norclozapine_regressor_model.jlso")
    println("Loading model: norclozapine_regressor_model.jlso")
    nclo_model_rfr = machine("models/norclozapine_regressor_model.jlso")
else
    error("File models/norclozapine_regressor_model.jlso cannot be opened!")
    exit(-1)
end
if isfile("models/scaler.jld")
    println("Loading: scaler.jld")
    scaler = JLD2.load_object("models/scaler.jld")
else
    error("File models/scaler.jld cannot be opened!")
    exit(-1)
end

vsearch(y::Real, x::AbstractVector) = findmin(abs.(x .- y))[2]

function predict_single_patient(patient_data::Vector{<:Real}, scaler)

    clo_group = 0
    clo_group_p = 0
    clo_group_adj = 0
    clo_group_p = 0
    clo_group_adj_p = 0
    clo_level = 0
    nclo_level = 0

    # m = scaler.mean
    # s = scaler.scale
    data = patient_data[2:end]
    data = StatsBase.transform(scaler, reshape(data, 1, length(data)))
    data[isnan.(data)] .= 0
    patient_data[2:end] = data
    # or
    # patient_data[2:end] = (patient_data[2:end] .- m) ./ s
    x_gender = Bool(patient_data[1])
    x_cont = patient_data[2:5]
    x_rest = patient_data[6:end]
    x1 = DataFrame(:male=>x_gender)
    x2 = DataFrame(reshape(x_cont, 1, length(x_cont)), ["age", "dose", "bmi", "crp"])
    x3 = DataFrame(reshape(x_rest, 1, length(x_rest)), ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
    x = hcat(x1, x2, x3)
    x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Continuous, :inhibitors_3a4=>Continuous, :substrates_3a4=>Continuous, :inducers_1a2=>Continuous, :inhibitors_1a2=>Continuous, :substrates_1a2=>Continuous)

    yhat1 = MLJ.predict(clo_model_rfr, x)[1]
    yhat1 = round(yhat1, digits=1)
    yhat3 = MLJ.predict(nclo_model_rfr, x)[1]
    yhat3 = round(yhat3, digits=1)
    println("Predicted CLO level: $(yhat1)")
    clo_level = yhat1
    println("Predicted NCLO level: $(yhat3)")
    nclo_level = yhat3
    yhat2 = MLJ.predict(model_rfc, x)[1]
    print("Predicted group: ")
    p_high = broadcast(pdf, yhat2, "high")
    p_norm = broadcast(pdf, yhat2, "norm")
    if p_norm > p_high
        println("NORM, prob = $(round(p_norm, digits=2))")
        clo_group = 0
        clo_group_p = p_norm
    else
        println("HIGH, prob = $(round(p_high, digits=2))")
        clo_group = 1
        clo_group_p = p_high
    end
    if yhat1 > 550
        p_high += 0.2
        p_norm -= 0.2
    elseif yhat1 <= 550
        p_norm += 0.2
        p_high -= 0.2
    end
    if yhat3 > 400
        p_high += 0.1
        p_norm -= 0.1
    elseif yhat3 <= 400
        p_norm += 0.1
        p_high -= 0.1
    end
    p_high > 1.0 && (p_high = 1.0)
    p_high < 0.0 && (p_high = 0.0)
    p_norm > 1.0 && (p_norm = 1.0)
    p_norm < 0.0 && (p_norm = 0.0)
    if p_norm > p_high
        println("Adjusted prediction: NORM, prob = $(round(p_norm, digits=2))")
        clo_group_adj = 0
        clo_group_adj_p = p_norm
    else
        println("Adjusted prediction: HIGH, prob = $(round(p_high, digits=2))")
        clo_group_adj = 0
        clo_group_adj_p = p_high
    end
    return clo_group, clo_group_p, clo_group_adj, clo_group_adj_p, clo_level, nclo_level
end

function recommended_dose(patient_data::Vector{<:Real}, scaler)

    doses = 0:12.5:1000
    clo_concentration = zeros(length(doses))
    nclo_concentration = zeros(length(doses))
    clo_group = zeros(Int64, length(doses))
    clo_group_p = zeros(length(doses))
    clo_group_adjusted= zeros(Int64, length(doses))
    clo_group_adjusted_p = zeros(length(doses))

    for idx in eachindex(doses)
        data = patient_data
        data = vcat(data[1:2], doses[idx], data[3:end])
        m = scaler.mean
        s = scaler.scale
        data[2:end] = (data[2:end] .- m) ./ s
        data[isnan.(data)] .= 0
        x_gender = Bool(data[1])
        x_cont = data[2:5]
        x_rest = data[6:end]
        x1 = DataFrame(:male=>x_gender)
        x2 = DataFrame(reshape(x_cont, 1, length(x_cont)), ["age", "dose", "bmi", "crp"])
        x3 = DataFrame(reshape(x_rest, 1, length(x_rest)), ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
        x = hcat(x1, x2, x3)
        x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Continuous, :inhibitors_3a4=>Continuous, :substrates_3a4=>Continuous, :inducers_1a2=>Continuous, :inhibitors_1a2=>Continuous, :substrates_1a2=>Continuous)
        yhat1 = MLJ.predict(clo_model_rfr, x)[1]
        yhat1 = round(yhat1, digits=1)
        clo_concentration[idx] = yhat1
        yhat3 = MLJ.predict(nclo_model_rfr, x)[1]
        yhat3 = round(yhat3, digits=1)
        nclo_concentration[idx] = yhat3
        yhat2 = MLJ.predict(model_rfc, x)[1]
        p_high = broadcast(pdf, yhat2, "high")
        p_norm = broadcast(pdf, yhat2, "norm")
        if p_norm > p_high
            clo_group[idx] = 0
            clo_group_p[idx] = round(p_norm, digits=2)
        else
            clo_group[idx] = 1
            clo_group_p[idx] = round(p_high, digits=2)
        end
        if yhat1 > 550
            p_high += 0.2
            p_norm -= 0.2
        elseif yhat1 <= 550
            p_norm += 0.2
            p_high -= 0.2
        end
        if yhat3 > 400
            p_high += 0.1
            p_norm -= 0.1
        elseif yhat3 <= 400
            p_norm += 0.1
            p_high -= 0.1
        end
        p_high > 1.0 && (p_high = 1.0)
        p_high < 0.0 && (p_high = 0.0)
        p_norm > 1.0 && (p_norm = 1.0)
        p_norm < 0.0 && (p_norm = 0.0)

        if p_norm > p_high
            clo_group_adjusted[idx] = 0
            clo_group_adjusted_p[idx] = round(p_norm, digits=2)
        else
            clo_group_adjusted[idx] = 1
            clo_group_adjusted_p[idx] = round(p_high, digits=2)
        end
    end

    return collect(doses), clo_concentration, nclo_concentration, clo_group, clo_group_p, clo_group_adjusted, clo_group_adjusted_p
end

function dose_range(doses, clo_concentration, nclo_concentration, clo_group, clo_group_p, clo_group_adjusted, clo_group_adjusted_p)

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

    plot(doses, clo_concentration, ylims=(0, 1000), xlims=(0, 1000), legend=false, xlabel="dose [mg/day]", ylabel="clozapine concentration [ng/mL]", margins=20Plots.px)
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

pt = [0, 60, 412.5, 27, 0.5, 0, 0, 0, 0, 0, 0]
predict_single_patient(pt, scaler)

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

pt = [0, 60, 26, 0, 0, 0, 0, 0, 0, 1]
doses, clo_concentration, nclo_concentration, clo_group, clo_group_p, clo_group_adjusted, clo_group_adjusted_p = recommended_dose(pt, scaler)
dose_range(doses, clo_concentration, nclo_concentration, clo_group, clo_group_p, clo_group_adjusted, clo_group_adjusted_p)
