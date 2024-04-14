# data columns:
# - male: 0/1
# - age: Float
# - dose: Float
# - bmi: Float
# - weight: Float
# - duration of clozapine treatment [days]: Float
# - crp: Float
# - inducers_3a4: Int
# - inhibitors_3a4: Int
# - substrates_3a4: Int
# - inducers_1a2: Int
# - inhibitors_1a2: Int
# - substrates_1a2: Int

println("Importing packages")
using CSV
using DataFrames
using MLJ
using MLJDecisionTreeInterface
using MLJLinearModels
using Random
using StatsBase
using JLD2
using Plots

# load training data
if isfile("clozapine_test.csv")
    println("Loading raw data: clozapine_test.csv")
    predict_raw_data = CSV.read("clozapine_test.csv", header=true, DataFrame)
else
    error("File clozapine_test.csv cannot be opened!")
    exit(-1)
end

# load models
if isfile("data/clozapine_classifier_model.jlso")
    println("Loading model: clozapine_classifier_model.jlso")
    model_rfc = machine("data/clozapine_classifier_model.jlso")
else
    error("File data/clozapine_classifier_model.jlso cannot be opened!")
    exit(-1)
end
if isfile("data/clozapine_regressor_model.jlso")
    println("Loading model: clozapine_regressor_model.jlso")
    clo_model_rfr = machine("data/clozapine_regressor_model.jlso")
else
    error("File data/clozapine_regressor_model.jlso cannot be opened!")
    exit(-1)
end
if isfile("data/norclozapine_regressor_model.jlso")
    println("Loading model: norclozapine_regressor_model.jlso")
    nclo_model_rfr = machine("data/norclozapine_regressor_model.jlso")
else
    error("File data/norclozapine_regressor_model.jlso cannot be opened!")
    exit(-1)
end
if isfile("data/scaler.jld")
    println("Loading: scaler.jld")
    scaler = JLD2.load_object("data/scaler.jld")
else
    error("File data/scaler.jld cannot be opened!")
    exit(-1)
end

vsearch(y::Real, x::AbstractVector) = findmin(abs.(x .- y))[2]

function predict_single_patient(patient_data::Vector{Float64}, scaler)

    m = scaler.mean
    s = scaler.scale
    patient_data[2:7] = (patient_data[2:7] .- m) ./ s
    patient_data[isnan.(patient_data)] .= 0
    x_gender = Bool(patient_data[1])
    x_cont = patient_data[2:7]
    x_rest = patient_data[8:end]
    x1 = DataFrame(:male=>x_gender)
    # x2 = DataFrame(x_cont, ["age", "dose", "bmi", "weight", "duration", "crp"])
    x_cont = x_cont[[1, 2, 3, 6]]
    x2 = DataFrame(reshape(x_cont, 1, length(x_cont)), ["age", "dose", "bmi", "crp"])
    x3 = DataFrame(reshape(x_rest, 1, length(x_rest)), ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
    x = hcat(x1, x2, x3)
    # x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :weight=>Continuous, :duration=>Continuous, :crp=>Continuous, :inducers_3a4=>Count,  :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)
    x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Count, :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)

    yhat1 = MLJ.predict(clo_model_rfr, x)[1]
    yhat1 = round(yhat1, digits=1)
    yhat3 = MLJ.predict(nclo_model_rfr, x)[1]
    yhat3 = round(yhat3, digits=1)
    println("Predicted CLO level: $(yhat1)")
    println("Predicted NCLO level: $(yhat3)")
    yhat2 = MLJ.predict(model_rfc, x)[1]
    print("Predicted group: ")
    p_high = broadcast(pdf, yhat2, "high")
    p_norm = broadcast(pdf, yhat2, "norm")
    if p_norm > p_high
        println("NORM, prob = $(round(p_norm, digits=2))")
    else
        println("HIGH, prob = $(round(p_high, digits=2))")
    end
    if yhat1 > 550
        p_high += 0.5
        p_norm -= 0.5
    elseif yhat1 <= 550
        p_norm += 0.5
        p_high -= 0.5
    end
    if yhat3 > 400
        p_high += 0.5
        p_norm -= 0.5
    elseif yhat3 <= 400
        p_norm += 0.5
        p_high -= 0.5
    end
    p_high > 1.0 && (p_high = 1.0)
    p_high < 0.0 && (p_high = 0.0)
    p_norm > 1.0 && (p_norm = 1.0)
    p_norm < 0.0 && (p_norm = 0.0)
    if p_norm > p_high
        println("Adjusted prediction: NORM, prob = $(round(p_norm, digits=2))")
    else
        println("Adjusted prediction: HIGH, prob = $(round(p_high, digits=2))")
    end
end

function recommended_dose(patient_data::Vector{Float64}, scaler)
    # - male: 0/1
    # - age: Float
    # + dose: Float
    # - bmi: Float
    # + weight: Float
    # + duration of clozapine treatment [days]: Float
    # - crp: Float
    # - inducers_3a4: Int
    # - inhibitors_3a4: Int
    # - substrates_3a4: Int
    # - inducers_1a2: Int
    # - inhibitors_1a2: Int
    # - substrates_1a2: Int

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
        data[2:7] = (data[2:7] .- m) ./ s
        data[isnan.(data)] .= 0
        x_gender = Bool(data[1])
        x_cont = data[2:7]
        x_rest = data[8:end]
        x1 = DataFrame(:male=>x_gender)
        # x2 = DataFrame(x_cont, ["age", "dose", "bmi", "weight", "duration", "crp"])
        x_cont = x_cont[[1, 2, 3, 6]]
        x2 = DataFrame(reshape(x_cont, 1, length(x_cont)), ["age", "dose", "bmi", "crp"])
        x3 = DataFrame(reshape(x_rest, 1, length(x_rest)), ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
        x = hcat(x1, x2, x3)
        # x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :weight=>Continuous, :duration=>Continuous, :crp=>Continuous, :inducers_3a4=>Count,  :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)
        x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Count, :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)
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
            p_high += 0.5
            p_norm -= 0.5
        elseif yhat1 <= 550
            p_norm += 0.5
            p_high -= 0.5
        end
        if yhat3 > 400
            p_high += 0.5
            p_norm -= 0.5
        elseif yhat3 <= 400
            p_norm += 0.5
            p_high -= 0.5
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

    # 220-550
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

    plot(doses, clo_concentration, ylims=(0, maximum(clo_concentration) > 550 ? maximum(clo_concentration) + 200 : 600), legend=false, xlabel="dose [mg/day]", ylabel="clozapine concentration [ng/mL]")
    hline!([220], lc=:green, ls=:dot)
    hline!([550], lc=:red, ls=:dot)
    vline!([doses[min_dose_idx]], lc=:green, ls=:dot)
    vline!([doses[max_dose_idx]], lc=:red, ls=:dot)

end

# - male: 0/1
# - age: Float
# - dose: Float
# - bmi: Float
# - weight: Float
# - duration of clozapine treatment [days]: Float
# - crp: Float
# - inducers_3a4: Int
# - inhibitors_3a4: Int
# - substrates_3a4: Int
# - inducers_1a2: Int
# - inhibitors_1a2: Int
# - substrates_1a2: Int

predict_single_patient([0,58,150,23.18,67,93,3,0,0,0,0,0,1], scaler)

doses, clo_concentration, nclo_concentration, clo_group, clo_group_p, clo_group_adjusted, clo_group_adjusted_p = recommended_dose([0,58,23.18,1,1,3,0,0,0,0,0,1], scaler)

dose_range(doses, clo_concentration, nclo_concentration, clo_group, clo_group_p, clo_group_adjusted, clo_group_adjusted_p)
