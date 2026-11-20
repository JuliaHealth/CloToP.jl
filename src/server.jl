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

# Three models are evaluated for each query (see reports/compare_summary.csv):
# - neural networks with the total dose + morning/midday doses (variant totaldose_morn_midday
#   of src/train.jl): lowest RMSE in repeated patient-level cross-validation (src/compare.jl)
#   for norclozapine; for clozapine, the representations did not differ significantly, and
#   this one gives far fewer non-monotonic dose-concentration curves than total dose + MM;
# - multiple linear regression with the total dose + morning/midday doses (src/train.jl);
# - polynomial models of the total dose through the origin (src/train.jl).

const COMMON_FEATURES_PRE = ["male", "age"]
const CYP_FEATURES = ["inducers_3a4", "inhibitors_3a4", "substrates_3a4",
                      "inducers_1a2", "inhibitors_1a2", "substrates_1a2"]
const COMMON_FEATURES_POST = vcat(["bmi", "crp"], CYP_FEATURES)
const CYP_LEVELS = 0:5
const MALE_LEVELS = 0:1
const MM_MIDDAY_WEIGHT = 0.5861325 # must match src/train.jl
features(dose_features) = vcat(COMMON_FEATURES_PRE, dose_features, COMMON_FEATURES_POST)

struct NNModel
    variant::String
    features::Vector{String}
    machine
    scaler_features
    scaler_level
end

# target: "clozapine" or "norclozapine"; file names as written by src/train.jl
function load_nn(target, variant, dose_features)
    suffix = "_$(variant)"
    level = target == "clozapine" ? "clozapine" : "nclozapine"
    return NNModel(variant, features(dose_features),
                   load_machine("models/$(target)_regressor_machine$(suffix).jlso"),
                   load_jld("models/scaler_features$(suffix).jld"),
                   load_jld("models/scaler_$(level)_level$(suffix).jld"))
end

const CLO_NN = load_nn("clozapine", "totaldose_morn_midday", ["total_dose", "morning_dose", "midday_dose"])
const NCLO_NN = load_nn("norclozapine", "totaldose_morn_midday", ["total_dose", "morning_dose", "midday_dose"])

# CYP levels present in the training data
cyp_levels_seen = load_jld("models/cyp_levels.jld")

# polynomial dose-concentration models through the origin (coefficients of td, td², …),
# see reports/polynomial_models.csv
polynomial_models = load_jld("models/polynomial_models.jld")
const CLO_POLY = polynomial_models["clozapine"]
const NCLO_POLY = polynomial_models["norclozapine"]
println("  Clozapine polynomial coefficients:    $CLO_POLY")
println("  Norclozapine polynomial coefficients: $NCLO_POLY")

# linear regression models: term => coefficient (intercept and features), see reports/linear_models.csv
linear_models = load_jld("models/linear_models.jld")
const CLO_LIN = linear_models["clozapine"]
const NCLO_LIN = linear_models["norclozapine"]

const MODEL_NAMES = ("Neural network", "Linear regression", "Polynomial")

# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

# All features of a patient for the given morning, midday and evening doses.
# p: patient data vector (see ctp())
function feature_dict(p, morning, midday, evening)
    total = morning + midday + evening
    d = Dict{String,Float64}("male" => p[1], "age" => p[2], "bmi" => p[6], "crp" => p[7],
                             "morning_dose" => morning, "midday_dose" => midday, "evening_dose" => evening,
                             "total_dose" => total,
                             "mm" => total > 0 ? (morning + MM_MIDDAY_WEIGHT * midday) / total : 0.0)
    for (i, c) in enumerate(CYP_FEATURES)
        d[c] = p[7 + i]
    end
    return d
end
feature_matrix(rows, colnames) = [r[c] for r in rows, c in colnames]

# polynomial models are used for total doses ≤ LOW_DOSE_MAX mg/day (must match src/train.jl);
# the neural network and linear regression models for total doses ≥ BLEND_MAX mg/day; in
# between, their predictions are blended linearly with the polynomial (weight rising from 0 to 1)
const LOW_DOSE_MAX = 100
const BLEND_MAX = 200
polynomial(c, dose) = round(sum(c[i] * dose^i for i in eachindex(c)), digits = 1)
model_weight(dose) = clamp((dose - LOW_DOSE_MAX) / (BLEND_MAX - LOW_DOSE_MAX), 0, 1)
blend(poly, model, dose) = (1 - model_weight(dose)) * poly + model_weight(dose) * model

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

# neural network prediction for each feature row
function predict_nn(m::NNModel, rows)
    df = make_df(feature_matrix(rows, m.features), m.features, m.scaler_features)
    ŷ = StatsBase.reconstruct(m.scaler_level, MLJ.predict(m.machine, df))
    return max.(round.(Float64.(ŷ), digits = 1), 0.0)
end

# linear regression prediction for each feature row
predict_linear(coef, rows) =
    [max(round(sum(b * (t == "intercept" ? 1.0 : r[t]) for (t, b) in coef), digits = 1), 0.0) for r in rows]

clo_toxic(clo) = clo > CLO_MAX ? 1 : 0
clo_toxic_adj(clo, nclo) = (clo > CLO_MAX || nclo > NCLO_MAX) ? 1 : 0

# CLO and NCLO concentrations predicted by the three models (MODEL_NAMES) for each dose in
# `doses` (total daily dose, distributed as `fractions` into morning/midday/evening doses);
# returns vectors of (clo, nclo) per model
function predict_models(p, doses, fractions)
    clo_poly = polynomial.(Ref(CLO_POLY), doses)
    nclo_poly = polynomial.(Ref(NCLO_POLY), doses)
    clo_nn, nclo_nn, clo_lin, nclo_lin = copy(clo_poly), copy(nclo_poly), copy(clo_poly), copy(nclo_poly)
    idx = findall(>(LOW_DOSE_MAX), doses)
    if !isempty(idx)
        rows = [feature_dict(p, (d .* fractions)...) for d in doses[idx]]
        clo_nn[idx] = blend.(clo_poly[idx], predict_nn(CLO_NN, rows), doses[idx])
        nclo_nn[idx] = blend.(nclo_poly[idx], predict_nn(NCLO_NN, rows), doses[idx])
        clo_lin[idx] = blend.(clo_poly[idx], predict_linear(CLO_LIN, rows), doses[idx])
        nclo_lin[idx] = blend.(nclo_poly[idx], predict_linear(NCLO_LIN, rows), doses[idx])
    end
    return [(clo_nn, nclo_nn), (clo_lin, nclo_lin), (clo_poly, nclo_poly)]
end

dose_fractions(p) = sum(p[3:5]) > 0 ? p[3:5] ./ sum(p[3:5]) : [0.0, 0.0, 1.0]

# Monotonicity constraint (safety net): the predicted concentration must not decrease with an
# increasing dose, so each curve is replaced by its running maximum. This does not change the
# recommended dose range, which uses the first dose crossing each threshold.
monotone(c) = accumulate(max, c)

# dose grid of the recommended dose range
const DOSES = collect(0:6.25:800)

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
#
# Returns, for each model (MODEL_NAMES): (clo group, adjusted clo group, clo level, nclo level).
function ctp(patient_data::Vector{<:Real})
    length(patient_data) == 13 || throw(ArgumentError("Patient data vector must have 13 elements."))
    p = Float64.(patient_data)
    total_dose = sum(p[3:5])
    # prediction at the patient's dose, constrained (running maximum) by the predictions at the
    # lower doses of the dose grid
    preds = predict_models(p, vcat(filter(<(total_dose), DOSES), total_dose), dose_fractions(p))
    out = map(preds) do (c, n)
        clo, nclo = round(monotone(c)[end], digits = 1), round(monotone(n)[end], digits = 1)
        (clo_toxic(clo), clo_toxic_adj(clo, nclo), clo, nclo)
    end
    return out
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

# recommended doses are rounded to multiples of the smallest dose that can be given
# (half of a 25 mg tablet) [mg]
const DOSE_ROUNDING = 12.5

# Recommended dose range for the CLO curve `clo` over `doses`:
# - minimum: lowest dose with CLO ≥ CLO_MIN, rounded up to a multiple of DOSE_ROUNDING
#   (nothing if CLO_MIN is not reached within the dose range);
# - maximum: highest dose with CLO ≤ CLO_MAX, rounded down to a multiple of DOSE_ROUNDING
#   (the last dose if CLO_MAX is not exceeded; nothing if it is exceeded at every dose).
# The curves are non-decreasing (running maximum), so the highest dose with CLO ≤ CLO_MAX is
# the dose just before the first one exceeding it. Returns (minimum, maximum, empty), where
# empty is true if no rounded dose lies within the range (minimum > maximum).
function dose_range(doses, clo)
    min_idx = findfirst(>=(CLO_MIN), clo)
    over_idx = findfirst(>(CLO_MAX), clo)
    max_dose = isnothing(over_idx) ? doses[end] : over_idx == 1 ? nothing : doses[over_idx - 1]
    dmin = isnothing(min_idx) ? nothing : ceil(doses[min_idx] / DOSE_ROUNDING) * DOSE_ROUNDING
    dmax = isnothing(max_dose) ? nothing : floor(max_dose / DOSE_ROUNDING) * DOSE_ROUNDING
    empty = isnothing(dmax) || (!isnothing(dmin) && dmin > dmax)
    return dmin, dmax, empty
end

# Patient's data: as for ctp() (13 elements, including morning, midday and evening doses).
#
# The dose range is predicted for total doses 0-800 mg/day, keeping the patient's
# distribution of the total dose into morning, midday and evening doses; if all doses
# are 0, the whole dose is assumed to be taken in the evening.
#
# Returns the doses and, for each model (MODEL_NAMES), the recommended dose range (see
# dose_range()) and the predicted CLO and NCLO curves.
function recommended_dose(patient_data::Vector{<:Real})
    length(patient_data) == 13 || throw(ArgumentError("Patient data vector must have 13 elements."))
    p = Float64.(patient_data)
    doses = DOSES
    results = map(predict_models(p, doses, dose_fractions(p))) do (clo, nclo)
        # gentle smoothing of the whole curve, then the monotonicity constraint
        clo = monotone(filter_mavg(clo, k = 2))
        nclo = monotone(filter_mavg(nclo, k = 2))
        (dose_range = dose_range(doses, clo), clo = clo, nclo = nclo)
    end
    return doses, results
end

function plot_recommended_dose(doses, results)
    p = plot(ylims = (0, 1000),
             xlims = (0, 800),
             legend = :topleft,
             legendfontsize = 12,
             xlabel = "Total clozapine dose [mg/day]",
             ylabel = "Clozapine concentration [ng/mL]",
             margins = 40Plots.px,
             guidefontsize = 14,
             tickfontsize = 12,
             size = (1200, 600))
    styles = [(:black, :solid), (:royalblue, :dash), (:darkorange, :dashdot)]
    for (name, r, (lc, ls)) in zip(MODEL_NAMES, results, styles)
        plot!(p, doses, r.clo, label = name, lc = lc, ls = ls, lw = 2)
    end
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

        doses, results = recommended_dose(patient)

        p = plot_recommended_dose(doses, results)
        io = IOBuffer()
        iob64_encode = Base64EncodePipe(io)
        show(iob64_encode, MIME("image/png"), p)
        close(iob64_encode)
        plot_b64 = String(take!(io))

        @info "Calculating per-dose predictions"
        levels = ctp(patient)

        # dose range as text: ">800" if the minimum is not reached within the dose range;
        # "none" for both if no rounded dose lies within the range
        fmt(d) = isinteger(d) ? string(Int(d)) : string(d)
        dose_min(r) = r.dose_range[3] ? "none" : isnothing(r.dose_range[1]) ? ">$(fmt(doses[end]))" : fmt(r.dose_range[1])
        dose_max(r) = r.dose_range[3] ? "none" : fmt(r.dose_range[2])
        models = [Dict("name" => name, "clo_group" => g, "clo_group_adj" => ga, "clo" => c, "nclo" => n,
                       "dose_min" => dose_min(r), "dose_max" => dose_max(r))
                  for (name, (g, ga, c, n), r) in zip(MODEL_NAMES, levels, results)]

        return HTTP.Response(200,
            vcat(HEADERS, ["Content-Type" => "application/json"]),
            JSON3.write(Dict("models" => models, "plot" => plot_b64)))
    end
    return HTTP.Response(200,
        HEADERS,
        read("./index.html"))
end

println("Precompiling")
doses, results = recommended_dose([0, 18, 50, 50, 50, 25, 0.0, 0, 0, 0, 0, 0, 0])
p = plot_recommended_dose(doses, results)
levels = ctp([0, 18, 50, 50, 50, 25, 0.0, 0, 0, 0, 0, 0, 0])

println("Starting server")
HTTP.serve(handle, "0.0.0.0", 8080, verbose = true)
