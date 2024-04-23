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
using Base64

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

vsearch(y::Real, x::AbstractVector) = findmin(abs.(x .- y))[2]

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
    data = patient_data[2:end]
    data = StatsBase.transform(scaler, reshape(data, 1, length(data)))
    data[isnan.(data)] .= 0
    patient_data[2:end] = data
    # patient_data[2:5] = (patient_data[2:5] .- m) ./ s
    x_gender = Bool(patient_data[1])
    x_cont = patient_data[2:5]
    x_rest = patient_data[6:end]
    x1 = DataFrame(:male=>x_gender)
    x2 = DataFrame(reshape(x_cont, 1, length(x_cont)), ["age", "dose", "bmi", "crp"])
    x3 = DataFrame(reshape(x_rest, 1, length(x_rest)), ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
    x = hcat(x1, x2, x3)
    x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Continuous, :inhibitors_3a4=>Continuous, :substrates_3a4=>Continuous, :inducers_1a2=>Continuous, :inhibitors_1a2=>Continuous, :substrates_1a2=>Continuous)

    Random.seed!(123)
    yhat1 = MLJ.predict(clo_model_rfr, x)[1]
    yhat1 = round(yhat1, digits=1)
    Random.seed!(123)
    yhat3 = MLJ.predict(nclo_model_rfr, x)[1]
    yhat3 = round(yhat3, digits=1)
    clo_level = yhat1
    nclo_level = yhat3
    
    Random.seed!(123)
    yhat2 = MLJ.predict(model_rfc, x)[1]
    p_norm = broadcast(pdf, yhat2, "norm")
    p_high = broadcast(pdf, yhat2, "high")
    if p_norm > p_high
        clo_group = 0
        clo_group_p = round(p_norm, digits=2)
    else
        clo_group = 1
        clo_group_p = round(p_high, digits=2)
    end
    if yhat1 > 550
        p_norm -= 0.2
        p_high += 0.2
    elseif yhat1 <= 550
        p_norm += 0.2
        p_high -= 0.2
    end
    if yhat3 > 400
        p_norm -= 0.1
        p_high += 0.1
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
        clo_group_adj = 1
        clo_group_adj_p = round(p_high, digits=2)
    end
    return clo_group, clo_group_p, clo_group_adj, clo_group_adj_p, clo_level, nclo_level
end

function recommended_dose(patient_data::Vector{<:Real}, scaler)

    # 220-550 ng/mL
    # source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10201335/

    doses = 0:12.5:1000
    clo_concentration = zeros(length(doses))
    nclo_concentration = zeros(length(doses))
    clo_group = zeros(Int64, length(doses))
    clo_group_p = zeros(length(doses))
    clo_group_adj= zeros(Int64, length(doses))
    clo_group_adj_p = zeros(length(doses))

    for idx in eachindex(doses)
        data = patient_data
        data = vcat(data[1:2], doses[idx], data[3:end])
        clo_group[idx], clo_group_p[idx], clo_group_adj[idx], clo_group_adj_p[idx], clo_concentration[idx], nclo_concentration[idx] = ctp(data, scaler)
    end

    if minimum(clo_concentration) < 220
        min_dose_idx = findfirst(x -> x > 220, clo_concentration) - 1
        min_dose_idx < 1 && (min_dose_idx = 1)
    else
        min_dose_idx = 1
    end

    if maximum(clo_concentration) > 550
        max_dose_idx = findfirst(x -> x > 550, clo_concentration) - 1
        max_dose_idx < 1 && (max_dose_idx = 1)
    else
        max_dose_idx = length(doses)
    end

    dose_range = (doses[min_dose_idx], doses[max_dose_idx])

    return dose_range, collect(doses), clo_concentration, nclo_concentration, clo_group, clo_group_p, clo_group_adj, clo_group_adj_p
end

function plot_recommended_dose(doses, clo_concentration, nclo_concentration, clo_group, clo_group_p, clo_group_adj, clo_group_adj_p)

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

    p = plot(doses, clo_concentration, ylims=(0, 1000), xlims=(0, 1000), legend=false, xlabel="dose [mg/day]", ylabel="clozapine concentration [ng/mL]", margins=20Plots.px)
    p = hline!([220], lc=:green, ls=:dot)
    p = hline!([550], lc=:red, ls=:dot)
    p = vline!([doses[min_dose_idx]], lc=:green, ls=:dot)
    p = vline!([doses[max_dose_idx]], lc=:red, ls=:dot)
    return p
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
        dose_range, doses, clo_concentration, nclo_concentration, clo_group, clo_group_p, clo_group_adj, clo_group_adj_p = recommended_dose([sex, age, bmi, crp, a4_ind, a4_inh, a4_s, a2_ind, a2_inh, a2_s], scaler)
        p = plot_recommended_dose(doses, clo_concentration, nclo_concentration, clo_group, clo_group_p, clo_group_adj, clo_group_adj_p)
        io = IOBuffer()
        iob64_encode = Base64EncodePipe(io)
        show(iob64_encode, MIME("image/png"), p)
        close(iob64_encode)
        p = String(take!(io))
        clo_group, clo_group_p, clo_group_adj, clo_group_adj_p, clo_level, nclo_level = ctp([sex, age, clo_dose, bmi, crp, a4_ind, a4_inh, a4_s, a2_ind, a2_inh, a2_s], scaler)
        return HTTP.Response(200, "$(clo_group) $(clo_group_p) $(clo_group_adj) $(clo_group_adj_p) $(clo_level) $(nclo_level) $(dose_range[1]) $(dose_range[2]) $(p)")
    end
    return HTTP.Response(200, read("./index.html"))
end

HTTP.serve(handle, 8080)
