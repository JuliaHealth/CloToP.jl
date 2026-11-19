println("Loading: packages")

using HTTP
using JSON3
using Base64
using DataFrames
using JLD2
using MLJ
using MLJFlux
using Optimisers
using Flux
using Plots
using StatsBase

# do not display plots
ENV["GKSwstype"] = "100"

println("        HTTP: $(string(pkgversion(HTTP)))")
println("       JSON3: $(string(pkgversion(JSON3)))")
println("  DataFrames: $(string(pkgversion(DataFrames)))")
println("        JLD2: $(string(pkgversion(JLD2)))")
println("        Flux: $(string(pkgversion(Flux)))")
println("         MLJ: $(string(pkgversion(MLJ)))")
println("     MLJFlux: $(string(pkgversion(MLJFlux)))")
println("  Optimisers: $(string(pkgversion(Optimisers)))")
println("       Plots: $(string(pkgversion(Plots)))")
println("   StatsBase: $(string(pkgversion(StatsBase)))")

# ---------------------------------------------------------------------------
# models and scalers (written by src/train.jl)
# ---------------------------------------------------------------------------

function load_file(f, loader)
    isfile(f) || error("File $f cannot be opened!")
    println("  Loading: $f")
    return loader(f)
end
load_machine(f) = load_file(f, MLJ.machine)
load_jld(f) = load_file(f, JLD2.load_object)

println("Loading models and scalers")

# total dose + morning/midday doses model (variant totaldose_morn_midday of src/train.jl),
# used both for the patient's dosing schedule and for the recommended dose range
const MODEL_SUFFIX = "_totaldose_morn_midday"
clozapine_regressor_machine = load_machine("models/clozapine_regressor_machine$(MODEL_SUFFIX).jlso")
norclozapine_regressor_machine = load_machine("models/norclozapine_regressor_machine$(MODEL_SUFFIX).jlso")
scaler_features = load_jld("models/scaler_features$(MODEL_SUFFIX).jld")
scaler_clozapine_level = load_jld("models/scaler_clozapine_level$(MODEL_SUFFIX).jld")
scaler_norclozapine_level = load_jld("models/scaler_nclozapine_level$(MODEL_SUFFIX).jld")

# CYP levels present in the training data
cyp_levels_seen = load_jld("models/cyp_levels.jld")

# polynomial dose-concentration models through the origin (coefficients of td, td², …),
# see reports/polynomial_models.csv
polynomial_models = load_jld("models/polynomial_models.jld")
const CLO_POLY = polynomial_models["clozapine"]
const NCLO_POLY = polynomial_models["norclozapine"]
println("  Clozapine polynomial coefficients:    $CLO_POLY")
println("  Norclozapine polynomial coefficients: $NCLO_POLY")

# ---------------------------------------------------------------------------
# features; must match src/train.jl
# ---------------------------------------------------------------------------

const CYP_FEATURES = ["inducers_3a4", "inhibitors_3a4", "substrates_3a4",
                      "inducers_1a2", "inhibitors_1a2", "substrates_1a2"]
const CYP_LEVELS = 0:5
const MALE_LEVELS = 0:1
const FEATURES = vcat(["male", "age", "total_dose", "morning_dose", "midday_dose", "bmi", "crp"], CYP_FEATURES)

# model input row: male, age, total dose, morning dose, midday dose, bmi, crp, CYP counts
feature_row(male, age, morning, midday, evening, bmi, crp, cyp) =
    Matrix{Float64}(reshape([male, age, morning + midday + evening, morning, midday, bmi, crp, cyp...], 1, :))

# polynomial models are used for total doses ≤ LOW_DOSE_MAX mg/day (must match src/train.jl),
# neural network models for total doses ≥ BLEND_MAX mg/day; in between, the predictions are
# blended linearly (weight of the neural network rising from 0 to 1)
const LOW_DOSE_MAX = 100
const BLEND_MAX = 200
polynomial(c, dose) = round(sum(c[i] * dose^i for i in eachindex(c)), digits = 1)
nn_weight(dose) = clamp((dose - LOW_DOSE_MAX) / (BLEND_MAX - LOW_DOSE_MAX), 0, 1)
blend(poly, nn, dose) = (1 - nn_weight(dose)) * poly + nn_weight(dose) * nn

# clozapine: therapeutic 250-550 ng/mL, toxic > 550 ng/mL
# source: Northwood et al. 2023, https://doi.org/10.1192/bjp.2023.27
const CLO_MIN = 250
const CLO_MAX = 550
# norclozapine: no established toxicity threshold exists (AGNP 2026 guideline:
# https://doi.org/10.1055/a-2860-7861); the threshold is the norclozapine level
# corresponding to CLO_MAX at the mean clozapine/norclozapine ratio of 1.73
# (meta-analysis, n = 2317; Schoretsanitis et al. 2019, https://doi.org/10.1080/17512433.2019.1617695,
# cited in Meyer 2024, https://doi.org/10.1016/j.schres.2023.10.041): 550 / 1.73 ≈ 320 ng/mL
const NCLO_MAX = 320

# Map CYP counts to levels present in training data (embeddings of absent levels are
# untrained): clamp to 0:5, then use the nearest lower level seen in training.
function map_cyp_levels!(X::Matrix{Float64}, colnames::Vector{String})
    for c in CYP_FEATURES
        j = findfirst(==(c), colnames)
        seen = cyp_levels_seen[c]
        mapped = Set{Pair{Float64,Int}}()
        for i in axes(X, 1)
            v = clamp(round(Int, X[i, j]), first(CYP_LEVELS), last(CYP_LEVELS))
            lower = filter(<=(v), seen)
            v_mapped = isempty(lower) ? minimum(seen) : maximum(lower)
            v_mapped != X[i, j] && push!(mapped, X[i, j] => v_mapped)
            X[i, j] = v_mapped
        end
        for (v, v_mapped) in mapped
            @warn "$c = $v not present in training data, using $v_mapped"
        end
    end
    return X
end

# Build the model input exactly as src/train.jl does: male → Multiclass, CYP counts →
# OrderedFactor with fixed levels, continuous columns Z-score standardized.
function make_df(X::Matrix{Float64}, colnames::Vector{String}, scaler)
    X = map_cyp_levels!(copy(X), colnames)
    cont = findall(c -> c != "male" && !(c in CYP_FEATURES), colnames)
    df = DataFrame()
    for (j, c) in enumerate(colnames)
        if c == "male"
            df[!, c] = categorical(round.(Int, X[:, j]), levels = MALE_LEVELS)
        elseif c in CYP_FEATURES
            df[!, c] = categorical(round.(Int, X[:, j]), levels = CYP_LEVELS, ordered = true)
        else
            k = findfirst(==(j), cont)
            col = (X[:, j] .- scaler.mean[k]) ./ scaler.scale[k]
            col[isnan.(col)] .= 0 # constant column (std = 0)
            df[!, c] = col
        end
    end
    return df
end

# Predict clozapine and norclozapine levels for each row of X.
function predict_levels(X::Matrix{Float64}, colnames, clo_mach, nclo_mach, scaler_features, scaler_clo, scaler_nclo)
    df = make_df(X, colnames, scaler_features)
    clo = StatsBase.reconstruct(scaler_clo, MLJ.predict(clo_mach, df))
    nclo = StatsBase.reconstruct(scaler_nclo, MLJ.predict(nclo_mach, df))
    clo = max.(round.(Float64.(clo), digits = 1), 0.0)
    nclo = max.(round.(Float64.(nclo), digits = 1), 0.0)
    return clo, nclo
end

clo_toxic(clo) = clo > CLO_MAX ? 1 : 0
clo_toxic_adj(clo, nclo) = (clo > CLO_MAX || nclo > NCLO_MAX) ? 1 : 0

# Patient's data:
# - male: 0/1
# - age: Float
# - morning dose: Float
# - midday dose: Float
# - evening dose: Float
# - bmi: Float
# - crp: Float
# - inducers_3a4: Int
# - inhibitors_3a4: Int
# - substrates_3a4: Int
# - inducers_1a2: Int
# - inhibitors_1a2: Int
# - substrates_1a2: Int
function ctp(patient_data::Vector{<:Real}, clo_regressor_machine, norclo_regressor_machine, scaler_features, scaler_clo_level, scaler_norclo_level)

    length(patient_data) == 13 || throw(ArgumentError("Patient data vector must have 13 elements."))

    total_dose = sum(patient_data[3:5])
    if total_dose <= LOW_DOSE_MAX
        return 0, 0, polynomial(CLO_POLY, total_dose), polynomial(NCLO_POLY, total_dose)
    end

    p = Float64.(patient_data)
    X = feature_row(p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8:13])
    clo, nclo = predict_levels(X, FEATURES, clo_regressor_machine, norclo_regressor_machine,
                               scaler_features, scaler_clo_level, scaler_norclo_level)
    clo_level = round(blend(polynomial(CLO_POLY, total_dose), clo[1], total_dose), digits = 1)
    nclo_level = round(blend(polynomial(NCLO_POLY, total_dose), nclo[1], total_dose), digits = 1)

    return clo_toxic(clo_level), clo_toxic_adj(clo_level, nclo_level), clo_level, nclo_level

end

# Moving average filter
function filter_mavg(
    s::AbstractVector;
    k::Int64 = 8,
)::Vector{Float64}
    # unfiltered samples stay at their original value
    s_filtered = copy(s) .* 1.0 # convert to floats

    # left edge: truncated window [1 … idx]
    @inbounds for idx in 1:k
        s_tmp = @view s[1:idx]
        s_filtered[idx] = mean(s_tmp)
    end

    # interior: full centered window [idx-k … idx+k]
    @inbounds for idx in (k + 1):(length(s) - k)
        s_tmp = @view s[(idx - k):(idx + k)]
        s_filtered[idx] = mean(s_tmp)
    end

    # right edge: truncated window [idx … end]
    @inbounds for idx in (length(s) - k + 1):length(s)
        s_tmp = @view s[idx:end]
        s_filtered[idx] = mean(s_tmp)
    end

    return s_filtered
end

# Index of the first dose with clozapine level ≥ CLO_MIN (nothing if never reached)
# and of the first dose with level > CLO_MAX (last dose if never exceeded).
function dose_range_idx(clo_concentration)
    min_idx = findfirst(>=(CLO_MIN), clo_concentration)
    max_idx = something(findfirst(>(CLO_MAX), clo_concentration), length(clo_concentration))
    return min_idx, max_idx
end

# Patient's data: as for ctp() (13 elements, including morning, midday and evening doses).
#
# The dose range is predicted for total doses 0-800 mg/day, keeping the patient's
# distribution of the total dose into morning, midday and evening doses; if all doses
# are 0, the whole dose is assumed to be taken in the evening.
#
# Returns the recommended dose range (minimum is nothing if CLO_MIN is not reached at
# the maximum dose), doses, predicted concentrations and groups.
function recommended_dose(patient_data::Vector{<:Real}, clo_regressor_machine, norclo_regressor_machine, scaler_features, scaler_clo_level, scaler_norclo_level)

    length(patient_data) == 13 || throw(ArgumentError("Patient data vector must have 13 elements."))

    doses = collect(0:6.25:800)
    clo_concentration = polynomial.(Ref(CLO_POLY), doses)
    nclo_concentration = polynomial.(Ref(NCLO_POLY), doses)

    # neural network predictions for doses > LOW_DOSE_MAX in a single batch
    nn_idx = findall(>(LOW_DOSE_MAX), doses)
    p = Float64.(patient_data)
    patient_total = sum(p[3:5])
    fractions = patient_total > 0 ? p[3:5] ./ patient_total : [0.0, 0.0, 1.0]
    X = reduce(vcat, [feature_row(p[1], p[2], (dose .* fractions)..., p[6], p[7], p[8:13]) for dose in doses[nn_idx]])
    clo_nn, nclo_nn = predict_levels(X, FEATURES, clo_regressor_machine, norclo_regressor_machine,
                                     scaler_features, scaler_clo_level, scaler_norclo_level)

    # polynomial models up to LOW_DOSE_MAX, linear blend up to BLEND_MAX, neural networks above
    clo_concentration[nn_idx] = blend.(clo_concentration[nn_idx], clo_nn, doses[nn_idx])
    nclo_concentration[nn_idx] = blend.(nclo_concentration[nn_idx], nclo_nn, doses[nn_idx])

    # gentle smoothing of the whole curve
    clo_concentration = filter_mavg(clo_concentration, k = 2)
    nclo_concentration = filter_mavg(nclo_concentration, k = 2)

    clo_groups = clo_toxic.(clo_concentration)
    clo_groups_adjusted = clo_toxic_adj.(clo_concentration, nclo_concentration)

    min_idx, max_idx = dose_range_idx(clo_concentration)
    dose_range = (isnothing(min_idx) ? nothing : doses[min_idx], doses[max_idx])

    return dose_range,
           doses,
           clo_concentration,
           nclo_concentration,
           clo_groups,
           clo_groups_adjusted

end

function plot_recommended_dose(doses, clo_concentration)

    min_idx, max_idx = dose_range_idx(clo_concentration)

    p = plot(ylims = (0, 1000),
             xlims = (0, 800),
             legend = false,
             xlabel = "Total clozapine dose [mg/day]",
             ylabel = "Clozapine concentration [ng/mL]",
             margins = 40Plots.px,
             guidefontsize = 14,
             tickfontsize = 12,
             size = (1200, 600))
    p = hline!([CLO_MIN],
               lc = :green,
               ls = :dot,
               lw = 1)
    p = hline!([CLO_MAX],
               lc = :red,
               ls = :dot,
               lw = 1)
    if !isnothing(min_idx)
        p = vline!([doses[min_idx]],
                   lc = :green,
                   ls = :dot,
                   lw = 1)
    end
    p = vline!([doses[max_idx]],
               lc = :red,
               ls = :dot,
               lw = 1)
    p = plot!(doses,
              clo_concentration,
              lc = :black,
              lw = 2)

    return p

end

const HEADERS = ["Access-Control-Allow-Origin"  => "*",
                 "Access-Control-Allow-Methods" => "GET, POST"]

function handle(req)
    if req.method == "POST"

        local sex, age, morning_dose, midday_dose, evening_dose, bmi, crp, a4_ind, a4_inh, a4_s, a2_ind, a2_inh, a2_s
        try
            form = JSON3.read(String(req.body))
            sex          = Int64(form.sex)
            age          = Float64(form.age)
            morning_dose = Float64(form.clo_dose_morning)
            midday_dose  = Float64(form.clo_dose_midday)
            evening_dose = Float64(form.clo_dose_evening)
            bmi          = Float64(form.bmi)
            crp          = Float64(form.crp)
            a4_ind       = Int64(form.a4_ind)
            a4_inh       = Int64(form.a4_inh)
            a4_s         = Int64(form.a4_s)
            a2_ind       = Int64(form.a2_ind)
            a2_inh       = Int64(form.a2_inh)
            a2_s         = Int64(form.a2_s)
            sex in MALE_LEVELS || throw(ArgumentError("sex must be 0 or 1"))
            all(>=(0), (morning_dose, midday_dose, evening_dose)) || throw(ArgumentError("doses must be ≥ 0"))
        catch e
            @warn "Invalid query" exception = e
            return HTTP.Response(400, HEADERS, "Invalid query: $(sprint(showerror, e))")
        end

        @info "Query: sex=$sex age=$age morning=$morning_dose midday=$midday_dose evening=$evening_dose bmi=$bmi crp=$crp"

        patient = [sex, age, morning_dose, midday_dose, evening_dose, bmi, crp, a4_ind, a4_inh, a4_s, a2_ind, a2_inh, a2_s]

        dose_range, doses, clo_conc, _, _, _ = recommended_dose(
            patient,
            clozapine_regressor_machine,
            norclozapine_regressor_machine,
            scaler_features,
            scaler_clozapine_level,
            scaler_norclozapine_level
        )

        p = plot_recommended_dose(doses, clo_conc)
        io = IOBuffer()
        iob64_encode = Base64EncodePipe(io)
        show(iob64_encode, MIME("image/png"), p)
        close(iob64_encode)
        plot_b64 = String(take!(io))

        @info "Calculating per-dose predictions"
        clo_group, clo_group_adj, clo_level, nclo_level = ctp(
            patient,
            clozapine_regressor_machine,
            norclozapine_regressor_machine,
            scaler_features,
            scaler_clozapine_level,
            scaler_norclozapine_level)

        # minimum dose not reached within the dose range: no spaces, the client splits on " "
        dose_min = isnothing(dose_range[1]) ? ">$(Int(doses[end]))" : string(dose_range[1])

        return HTTP.Response(200,
            HEADERS,
            "$(clo_group) $(clo_group_adj) $(clo_level) $(nclo_level) $(dose_min) $(dose_range[2]) $(plot_b64)")
    end
    return HTTP.Response(200,
        HEADERS,
        read("./index.html"))
end

println("Precompiling")
dose_range, doses, clo_level, nclo_level, clo_group, clo_group_adj = recommended_dose([0, 18, 50, 50, 50, 25, 0.0, 0, 0, 0, 0, 0, 0], clozapine_regressor_machine, norclozapine_regressor_machine, scaler_features, scaler_clozapine_level, scaler_norclozapine_level)
p = plot_recommended_dose(doses, clo_level)
clo_group, clo_group_adj, clo_level, nclo_level = ctp([0, 18, 50, 50, 50, 25, 0.0, 0, 0, 0, 0, 0, 0], clozapine_regressor_machine, norclozapine_regressor_machine, scaler_features, scaler_clozapine_level, scaler_norclozapine_level)

println("Starting server")
HTTP.serve(handle, "0.0.0.0", 8080, verbose = true)
