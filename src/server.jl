@info "Importing packages"

using Pkg
# packages = ["HTTP", "JSON3", "CSV", "DataFrames", "JLD2", "Flux", "MLJ", "MLJFlux", "NNlib", "Optimisers", "Plots", "ProgressMeter", "Sockets", "StatsBase"]
# Pkg.add(packages)

using HTTP
using JSON3
using Base64
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
println("      HTTP $(m[findfirst(v -> v.name == "HTTP", m)].version)")
println("     JSON3 $(m[findfirst(v -> v.name == "JSON3", m)].version)")
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

# load models
if isfile("models/clozapine_regressor_model.jlso")
    println("Loading: clozapine_regressor_model.jlso")
    clo_model_regressor = machine("models/clozapine_regressor_model.jlso")
else
    error("File models/clozapine_regressor_model.jlso cannot be opened!")
    exit(-1)
end
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
println()

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

    # 250-550 ng/mL
    # source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10201335/

    doses = 0:12.5:800
    clo_concentration = zeros(length(doses))
    nclo_concentration = zeros(length(doses))
    clo_group = zeros(Int64, length(doses))
    clo_group_adjusted= zeros(Int64, length(doses))

    for idx in eachindex(doses)
        data = deepcopy(patient_data)
        data = vcat(data[1:2], doses[idx], data[3:end])
        clo_group[idx], clo_group_adjusted[idx], clo_concentration[idx], nclo_concentration[idx] = ctp(data, scaler_clo, scaler_nclo)
    end

    if minimum(clo_concentration) < 250
        min_dose_idx = findfirst(x -> x > 250, clo_concentration)
    else
        min_dose_idx = 1
    end

    if maximum(clo_concentration) > 550
        max_dose_idx = findfirst(x -> x > 550, clo_concentration)
    else
        max_dose_idx = length(doses)
    end

    dose_range = (doses[min_dose_idx], doses[max_dose_idx])

    return dose_range, collect(doses), clo_concentration, nclo_concentration, clo_group, clo_group_adjusted
end

function plot_recommended_dose(doses, clo_level)

    # 250-550
    # source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10201335/

    if minimum(clo_level) < 250
        min_dose_idx = findfirst(x -> x >= 250, clo_level)
    else
        min_dose_idx = 1
    end

    if maximum(clo_level) > 550
        max_dose_idx = findfirst(x -> x >= 550, clo_level)
    else
        max_dose_idx = length(doses)
    end

    p = plot(doses, clo_level, ylims=(0, 1000), xlims=(0, 800), legend=false, xlabel="dose [mg/day]", ylabel="clozapine concentration [ng/mL]", margins=20Plots.px)
    p = hline!([250], lc=:green, ls=:dot)
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
        dose_range, doses, clo_level, nclo_level, clo_group, clo_group_adj = recommended_dose([sex, age, bmi, crp, a4_ind, a4_inh, a4_s, a2_ind, a2_inh, a2_s], scaler_clo, scaler_nclo)
        p = plot_recommended_dose(doses, clo_level)
        io = IOBuffer()
        iob64_encode = Base64EncodePipe(io)
        show(iob64_encode, MIME("image/png"), p)
        close(iob64_encode)
        p = String(take!(io))
        @info "Query: sex: $sex, age: $age, clo_dose: $clo_dose, bmi: $bmi, crp: $crp, a4_ind: $a4_ind, a4_inh: $a4_inh, a4_s: $a4_s, a2_ind: $a2_ind, a2_inh: $a2_inh, a2_s: $a2_s"
        @info "Calculating predictions"
        clo_group, clo_group_adj, clo_level, nclo_level = ctp([sex, age, clo_dose, bmi, crp, a4_ind, a4_inh, a4_s, a2_ind, a2_inh, a2_s], scaler_clo, scaler_nclo)
        return HTTP.Response(200, ["Access-Control-Allow-Origin"=>"0.0.0.0", "Access-Control-Request-Method"=>"POST"], "$(clo_group) $(clo_group_adj) $(clo_level) $(nclo_level) $(dose_range[1]) $(dose_range[2]) $(p)")
    end
    return HTTP.Response(200, ["Access-Control-Allow-Origin"=>"0.0.0.0", "Access-Control-Request-Method"=>"POST"], read("./index.html"))
end

@info "Precompiling"
dose_range, doses, clo_level, nclo_level, clo_group, clo_group_adj = recommended_dose([0, 18, 25, 0.0, 0, 0, 0, 0, 0, 0], scaler_clo, scaler_nclo)
p = plot_recommended_dose(doses, clo_level)
clo_group, clo_group_adj, clo_level, nclo_level = ctp([0, 18, 100, 25, 0.0, 0, 0, 0, 0, 0, 0], scaler_clo, scaler_nclo)
println()

@info "Starting server"
HTTP.serve(handle, "0.0.0.0", 8080, verbose=true)
