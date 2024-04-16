println("Importing packages")
using HTTP, JSON3
using CSV
using DataFrames
using MLJ
using MLJDecisionTreeInterface
using MLJLinearModels
using Random
using StatsBase
using JLD2
using Plots

# load models
if isfile("models/clozapine_classifier_model.jlso")
    println("Loading: clozapine_classifier_model.jlso")
    model_rfc = machine("models/clozapine_classifier_model.jlso")
else
    error("File models/clozapine_classifier_model.jlso cannot be opened!")
    exit(-1)
end
if isfile("models/clozapine_regressor_model.jlso")
    println("Loading: clozapine_regressor_model.jlso")
    clo_model_rfr = machine("models/clozapine_regressor_model.jlso")
else
    error("File models/clozapine_regressor_model.jlso cannot be opened!")
    exit(-1)
end
if isfile("models/norclozapine_regressor_model.jlso")
    println("Loading: norclozapine_regressor_model.jlso")
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

function ctp(patient_data::Vector{<:Real}, scaler)

    clo_group = 0
    clo_group_p = 0
    clo_group_adj = 0
    clo_group_p = 0
    clo_group_adj_p = 0
    clo_level = 0
    nclo_level = 0

    # m = scaler.mean
    # s = scaler.scale
    data = patient_data[2:5]
    data = StatsBase.transform(scaler, reshape(data, 1, length(data)))
    patient_data[2:5] = data
    # patient_data[2:5] = (patient_data[2:5] .- m) ./ s
    # patient_data[isnan.(patient_data)] .= 0
    x_gender = Bool(patient_data[1])
    x_cont = patient_data[2:5]
    x_rest = patient_data[6:end]
    x1 = DataFrame(:male=>x_gender)
    x2 = DataFrame(reshape(x_cont, 1, length(x_cont)), ["age", "dose", "bmi", "crp"])
    x3 = DataFrame(reshape(x_rest, 1, length(x_rest)), ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
    x = hcat(x1, x2, x3)
    x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Count, :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)

    yhat1 = MLJ.predict(clo_model_rfr, x)[1]
    yhat1 = round(yhat1, digits=1)
    yhat3 = MLJ.predict(nclo_model_rfr, x)[1]
    yhat3 = round(yhat3, digits=1)
    clo_level = yhat1
    nclo_level = yhat3
    
    yhat2 = MLJ.predict(model_rfc, x)[1]
    p_high = broadcast(pdf, yhat2, "high")
    p_norm = broadcast(pdf, yhat2, "norm")
    if p_norm > p_high
        clo_group = 0
        clo_group_p = round(p_norm, digits=2)
    else
        clo_group = 0
        clo_group_p = round(p_high, digits=2)
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
        clo_group_adj = 0
        clo_group_adj_p = round(p_norm, digits=2)
    else
        clo_group_adj = 0
        clo_group_adj_p = round(p_high, digits=2)
    end
    return clo_group, clo_group_p, clo_group_adj, clo_group_adj_p, clo_level, nclo_level
end

function handle(req)
    if req.method == "POST"
        form = JSON3.read(String(req.body))
        sex = form.sex
        age = form.age
        clo_dose = Float64(form.clo_dose)
        bmi = Float64(form.bmi)
        crp = Float64(form.crp)
        a4_ind = Float64(form.a4_ind)
        a4_inh = Float64(form.a4_inh)
        a4_s = Float64(form.a4_s)
        a2_ind = Float64(form.a2_ind)
        a2_inh = Float64(form.a2_inh)
        a2_s = Float64(form.a2_s)
        clo_group, clo_group_p, clo_group_adj, clo_group_adj_p, clo_level, nclo_level = ctp([sex, age, clo_dose, bmi, crp, a4_ind, a4_inh, a4_s, a2_ind, a2_inh, a2_s], scaler)
        return HTTP.Response(200, "$(clo_group) $(clo_group_p) $(clo_group_adj) $(clo_group_adj_p) $(clo_level) $(nclo_level)")
    end
    return HTTP.Response(200, read("./index.html"))
end

HTTP.serve(handle, 8080)
