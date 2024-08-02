export ctp
export recommended_dose
export dose_range

vsearch(y::Real, x::AbstractVector) = findmin(abs.(x .- y))[2]

"""
    ctp(patient_data::Vector{<:Real}, scaler_clo, scaler_nclo)

male: 0/1
age: Float
dose: Float
bmi: Float
crp: Float
inducers_3a4: Int
inhibitors_3a4: Int
substrates_3a4: Int
inducers_1a2: Int
inhibitors_1a2: Int
substrates_1a2: Int

pt = [0, 20, 512.5, 27, 10.5, 0, 0, 0, 0, 0, 0]
ctp(pt, CloToP.scaler_clo, CloToP.scaler_nclo)

"""
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

"""
    recommended_dose(patient_data::Vector{<:Real}, scaler_clo, scaler_nclo)

male: 0/1
age: Float
bmi: Float
crp: Float
inducers_3a4: Int
inhibitors_3a4: Int
substrates_3a4: Int
inducers_1a2: Int
inhibitors_1a2: Int
substrates_1a2: Int

pt = [0, 37, 23.9, 1.3, 1, 0, 1, 1, 0, 0]
recommended_dose(pt, CloToP.scaler_clo, CloToP.scaler_nclo)

# Returns

doses, clo_concentration, nclo_concentration, clo_group, clo_group_adjusted
"""
function recommended_dose(patient_data::Vector{<:Real}, scaler_clo, scaler_nclo)

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

    return collect(doses), clo_concentration, nclo_concentration, clo_group, clo_group_adjusted
end

"""
    dose_range(doses, clo_concentration, nclo_concentration, clo_group, clo_group_adjusted)

doses, clo_concentration, nclo_concentration, clo_group, clo_group_adjusted

# Returns

p

"""
function dose_range(doses, clo_concentration, nclo_concentration, clo_group, clo_group_adjusted)

    # therapeutic concentration range: 250-550 ng/mL
    # source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10201335/

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

    println("Minimum recommended dose: $(doses[min_dose_idx]) mg/day")
    println("Maximum recommended dose: $(doses[max_dose_idx]) mg/day")

    dose_range = (doses[min_dose_idx], doses[max_dose_idx])

    plot(doses, clo_concentration, ylims=(0, 1000), xlims=(0, 800), legend=false, xlabel="dose [mg/day]", ylabel="clozapine concentration [ng/mL]", margins=20Plots.px)
    hline!([250], lc=:green, ls=:dot)
    hline!([550], lc=:red, ls=:dot)
    vline!([doses[min_dose_idx]], lc=:green, ls=:dot)
    vline!([doses[max_dose_idx]], lc=:red, ls=:dot)
end
