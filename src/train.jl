# Train clozapine / norclozapine NeuralNetworkRegressor models.
#
# Usage:
#   julia -t 4 src/train.jl 2>&1 | tee reports/report.txt    # train all variants
#   julia -t 4 src/train.jl totaldose_mm                     # train selected variant(s) only
#
# Model variants (dose-related features, each a separate input of the model):
#   threedoses             3 features: morning dose, midday dose, evening dose
#   totaldose              1 feature:  total dose
#   totaldose_mm           2 features: total dose, MM
#                          where MM = (morning dose + 0.5861325 × midday dose) / total dose
#   totaldose_morn_midday  3 features: total dose, morning dose, midday dose

const OPTIMIZE = true # tune hyperparameters
const VERBOSE = 0     # show progress
const NFOLDS = 6      # cross-validation folds used for tuning
const SEED = 42
const MM_MIDDAY_WEIGHT = 0.5861325
const POLY_MAX_ORDER = 4 # polynomial dose-concentration models: orders 1:POLY_MAX_ORDER
const LOW_DOSE_MAX = 100 # src/server.jl uses the polynomial models for total doses ≤ LOW_DOSE_MAX

!isdir("reports") && mkdir("reports")
!isdir("models") && mkdir("models")

println("Loading: packages")

using CSV
using DataFrames
using JLD2
using MLJ
using MLJFlux
using Optimisers
using Flux
using LinearAlgebra
using Random
using Plots
using StatsBase
import Distributions

ENV["GKSwstype"] = "100" # do not display plots

println("         CSV: $(string(pkgversion(CSV)))")
println("  DataFrames: $(string(pkgversion(DataFrames)))")
println("        JLD2: $(string(pkgversion(JLD2)))")
println("        Flux: $(string(pkgversion(Flux)))")
println("         MLJ: $(string(pkgversion(MLJ)))")
println("     MLJFlux: $(string(pkgversion(MLJFlux)))")
println("  Optimisers: $(string(pkgversion(Optimisers)))")
println("       Plots: $(string(pkgversion(Plots)))")
println("   StatsBase: $(string(pkgversion(StatsBase)))")
println("     Threads: $(Threads.nthreads())")

println("Loading: NeuralNetworkRegressor")
NeuralNetworkRegressor = @MLJ.load NeuralNetworkRegressor pkg=MLJFlux verbosity=0

# ---------------------------------------------------------------------------
# model variants
# ---------------------------------------------------------------------------

const COMMON_FEATURES_PRE = ["male", "age"]
const COMMON_FEATURES_POST = ["bmi", "crp",
                              "inducers_3a4", "inhibitors_3a4", "substrates_3a4",
                              "inducers_1a2", "inhibitors_1a2", "substrates_1a2"]

# name => (dose features, file suffix); suffixes of the first two variants are kept
# unchanged, so that src/server.jl loads the retrained models without modification
const VARIANTS = [
    "threedoses"            => (["morning_dose", "midday_dose", "evening_dose"], ""),
    "totaldose"             => (["total_dose"], "_totaldose"),
    "totaldose_mm"          => (["total_dose", "mm"], "_totaldose_mm"),
    "totaldose_morn_midday" => (["total_dose", "morning_dose", "midday_dose"], "_totaldose_morn_midday"),
]

features(dose_features) = vcat(COMMON_FEATURES_PRE, dose_features, COMMON_FEATURES_POST)

# variants requested on the command line (default: all)
const requested = isempty(ARGS) ? first.(VARIANTS) : ARGS
for v in requested
    v in first.(VARIANTS) || error("Unknown variant: $v. Available: $(join(first.(VARIANTS), ", "))")
end

# categorical features; levels must match src/server.jl
const CYP_FEATURES = COMMON_FEATURES_POST[3:end]
const CYP_LEVELS = 0:5 # number of co-administered drugs, as allowed by the web form
const MALE_LEVELS = 0:1

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

rmse(ŷ, y) = RootMeanSquaredError()(ŷ, y)
rsq(ŷ, y) = RSquared()(ŷ, y)

function print_accuracy(ŷ, y; indent = "    ")
    println(indent, "R²:   ", round(rsq(ŷ, y), digits = 2))
    println(indent, "RMSE: ", round(rmse(ŷ, y), digits = 4))
end

# performance scores, exported to reports/performance.csv
const PERFORMANCE_FILE = "reports/performance.csv"
const PERFORMANCE = DataFrame(variant = String[], target = String[], model = String[], dataset = String[],
                              n = Int[], r2 = Float64[], rmse = Float64[], mae = Float64[],
                              eta = Float64[], lambda = Float64[], alpha = Float64[],
                              epochs = Int[], batch_size = Int[])

function record_performance!(variant, target, model_label, dataset, model, ŷ, y)
    push!(PERFORMANCE, (variant, target, model_label, dataset, length(y),
                        rsq(ŷ, y), rmse(ŷ, y), mean_absolute_error(ŷ, y),
                        model.optimiser.eta, model.lambda, model.alpha, model.epochs, model.batch_size))
end

# keep rows of variants not retrained in this run
function save_performance()
    perf = PERFORMANCE
    if isfile(PERFORMANCE_FILE)
        old = CSV.read(PERFORMANCE_FILE, DataFrame, stringtype = String)
        perf = vcat(old[.!in(Set(PERFORMANCE.variant)).(old.variant), :], PERFORMANCE)
    end
    rank = Dict(name => i for (i, name) in enumerate(first.(VARIANTS)))
    sort!(perf, [order(:variant, by = v -> get(rank, v, typemax(Int))), :target], alg = MergeSort)
    CSV.write(PERFORMANCE_FILE, perf)
end

function error_plot(y, ŷ, title; kwargs...)
    Plots.plot(y .- ŷ,
               seriestype = :stem,
               marker = :circle,
               markersize = 2,
               xlabel = "Observations",
               ylabel = "Error",
               title = title,
               legend = false;
               kwargs...)
end

# symmetric y-limits covering all errors
error_ylims(errs...) = (m = maximum(e -> maximum(abs, e), errs); (-1.05m, 1.05m))

# Z-score standardize the continuous columns using a scaler fitted on rows `fit_idx`;
# male (0/1) → Multiclass, CYP counts (raw, 0:5) → OrderedFactor with fixed levels, so
# that src/server.jl (single-row inputs) builds identical categorical columns.
# Returns the DataFrame and the scaler (continuous columns only, in column order).
function make_df(X::Matrix{Float64}, colnames::Vector{String}, fit_idx)
    cont = findall(c -> c != "male" && !(c in CYP_FEATURES), colnames)
    scaler = StatsBase.fit(ZScoreTransform, X[fit_idx, cont], dims = 1)
    df = DataFrame()
    for (j, c) in enumerate(colnames)
        if c == "male"
            df[!, c] = categorical(Int.(X[:, j]), levels = MALE_LEVELS)
        elseif c in CYP_FEATURES
            df[!, c] = categorical(Int.(X[:, j]), levels = CYP_LEVELS, ordered = true)
        else
            k = findfirst(==(j), cont)
            col = (X[:, j] .- scaler.mean[k]) ./ scaler.scale[k]
            col[isnan.(col)] .= 0 # constant column (std = 0)
            df[!, c] = col
        end
    end
    return df, scaler
end

function predict_level(mach, X, y_scaler)
    ŷ = StatsBase.reconstruct(y_scaler, MLJ.predict(mach, X))
    return max.(ŷ, 0)
end

function build_model() # n_in (after categorical embeddings) is provided by @builder
    builder = MLJFlux.@builder begin
        init = Flux.glorot_uniform(rng)
        Chain(
            Dense(n_in => 128, swish, init = init),
            Dropout(0.2; rng = rng),
            Dense(128 => 64, swish, init = init),
            Dropout(0.2; rng = rng),
            Dense(64 => 32, swish, init = init),
            Dropout(0.1; rng = rng),
            Dense(32 => 1, init = init),
        )
    end
    return NeuralNetworkRegressor(builder = builder,
                                  optimiser = Optimisers.AdamW(eta = 0.003), # no weight decay; regularization via lambda/alpha
                                  loss = Flux.Losses.mae,
                                  epochs = 2000,      # passes through the complete training dataset
                                  batch_size = 8,     # samples per weight update
                                  lambda = 0.001,     # regularization strength, [0, ∞)
                                  alpha = 0.75,       # L2/L1 mix: 0 = L2, 1 = L1
                                  rng = SEED)
end

# Run a grid search over `ranges` using cross-validation folds `folds`; return the best
# model and its CV score (mean absolute error on normalized levels).
function tune(model, ranges, X, y, folds)
    tuned_model = TunedModel(model = model,
                             tuning = Grid(),
                             resampling = folds,
                             range = ranges,
                             measure = mean_absolute_error,
                             check_measure = true,
                             train_best = false,
                             acceleration = CPUThreads(),
                             acceleration_resampling = CPUThreads())
    tuned_mach = machine(tuned_model, X, y, scitype_check_level = 1)
    MLJ.fit!(tuned_mach, force = true, verbosity = VERBOSE)
    r = report(tuned_mach)
    return r.best_model, r.best_history_entry.measurement[1]
end

function cv_score(model, X, y, folds)
    e = evaluate(model, X, y,
                 resampling = folds,
                 measure = mean_absolute_error,
                 acceleration = CPUThreads(),
                 verbosity = 0)
    return e.measurement[1]
end

# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

# Tune on the training rows, report accuracy on the (untouched) test rows.
# Tuning decisions use CV scores only; the test set is used for reporting only.
function train(X::DataFrame, y::Vector{Float64}, y_scaler::ZScoreTransform, train_idx, train_real_idx, test_idx, folds, var::String, variant::String, suffix::String)

    y_norm = StatsBase.transform(y_scaler, y)
    X_train = X[train_idx, :]
    y_train = y_norm[train_idx]
    model = build_model()

    mach = machine(model, X_train, y_train, scitype_check_level = 1)
    MLJ.fit!(mach, force = true, verbosity = VERBOSE)
    ŷ_initial = predict_level(mach, X[test_idx, :], y_scaler)
    println("  Initial model (testing)")
    print_accuracy(ŷ_initial, y[test_idx])
    record_performance!(variant, var, "initial", "testing", model, ŷ_initial, y[test_idx])

    OPTIMIZE || return model, ŷ_initial

    println("Running MLJ grid search")
    best_score = cv_score(model, X_train, y_train, folds)
    println("  Initial model CV MAE: ", round(best_score, digits = 4))

    stages = [
        ("eta", [range(model, :optimiser; values = [Optimisers.AdamW(eta = η) for η in (1e-4, 3e-4, 1e-3, 3e-3)])]),
        ("lambda, alpha", [range(model, :lambda; values = [0.001, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0]),
                           range(model, :alpha; values = [0.0, 0.25, 0.5, 0.75, 1.0])]),
        ("epochs, batch_size", [range(model, :epochs; values = [100, 200, 400, 800, 1600, 3200, 6400]),
                                range(model, :batch_size; values = [2, 4, 8, 16, 32, 64, 128])]),
    ]
    for (name, ranges) in stages
        println("  Optimizing: $name")
        best, score = tune(model, ranges, X_train, y_train, folds)
        println("    Best CV MAE: ", round(score, digits = 4))
        if score < best_score
            println("    Improvement, modifying the model parameters")
            model, best_score = best, score
        else
            println("    No improvement, keeping the model parameters")
        end
    end

    println("  Tuned model parameters:")
    println("    eta:        ", model.optimiser.eta)
    println("    alpha:      ", model.alpha)
    println("    lambda:     ", model.lambda)
    println("    epochs:     ", model.epochs)
    println("    batch_size: ", model.batch_size)

    mach = machine(model, X_train, y_train, scitype_check_level = 1)
    MLJ.fit!(mach, force = true, verbosity = VERBOSE)
    ŷ_train = predict_level(mach, X[train_real_idx, :], y_scaler)
    ŷ_test = predict_level(mach, X[test_idx, :], y_scaler)
    println("  Tuned model (training)")
    print_accuracy(ŷ_train, y[train_real_idx])
    println("  Tuned model (testing)")
    print_accuracy(ŷ_test, y[test_idx])
    record_performance!(variant, var, "tuned", "training", model, ŷ_train, y[train_real_idx])
    record_performance!(variant, var, "tuned", "testing", model, ŷ_test, y[test_idx])

    yl = error_ylims(y[test_idx] .- ŷ_initial, y[test_idx] .- ŷ_test)
    p = plot(error_plot(y[test_idx], ŷ_initial, "Prediction error: initial model (testing)"),
             error_plot(y[test_idx], ŷ_test, "Prediction error: tuned model (testing)"),
             layout = (2, 1), ylims = yl, dpi = 300)
    savefig(p, "reports/fig1_$(var)_tuning$(suffix).png")

    yl = error_ylims(y[train_real_idx] .- ŷ_train, y[test_idx] .- ŷ_test)
    p = plot(error_plot(y[train_real_idx], ŷ_train, "Prediction error: tuned model (training)"),
             error_plot(y[test_idx], ŷ_test, "Prediction error: tuned model (testing)"),
             layout = (2, 1), ylims = yl, dpi = 300)
    savefig(p, "reports/fig2_$(var)_tuned$(suffix).png")

    return model, ŷ_test

end

# Train the tuned model on all observations.
function train_final(model, X::DataFrame, y::Vector{Float64}, y_scaler::ZScoreTransform, real_idx, var::String, variant::String, suffix::String)

    mach = machine(model, X, StatsBase.transform(y_scaler, y), scitype_check_level = 1)
    MLJ.fit!(mach, force = true, verbosity = VERBOSE)
    ŷ = round.(predict_level(mach, X, y_scaler), digits = 1)

    # report real observations only (synthetic zero-dose rows excluded)
    y_r, ŷ_r = y[real_idx], ŷ[real_idx]
    sorting_idx = sortperm(y_r)
    p = Plots.plot(y_r[sorting_idx] .- ŷ_r[sorting_idx],
                   xlabel = "Observations (sorted by increasing level)",
                   ylabel = "Error",
                   title = "Prediction error: final model (all observations)",
                   legend = false,
                   ylims = error_ylims(y_r .- ŷ_r),
                   dpi = 300)
    savefig(p, "reports/fig3_$(var)_final$(suffix).png")

    println("  Final model (all observations, in-sample)")
    print_accuracy(ŷ_r, y_r)
    record_performance!(variant, var, "final", "all", model, ŷ_r, y_r)

    return mach, ŷ_r

end

function print_groups(y, ŷ; threshold = 550)
    lv = ["norm", "high"]
    group = categorical(ifelse.(y .> threshold, "high", "norm"), levels = lv, ordered = true)
    group_pred = categorical(ifelse.(ŷ .> threshold, "high", "norm"), levels = lv, ordered = true)
    cm = MLJ.confusion_matrix(group_pred, group)
    acc = sum(diag(cm.mat)) / sum(cm.mat)
    println("    Accuracy: ", round(acc, digits = 2))
    println("    Misclassification rate: ", round(1 - acc, digits = 2))
    println("    Confusion matrix:")
    display(cm)
    println()
end

# ---------------------------------------------------------------------------
# polynomial dose-concentration models
# ---------------------------------------------------------------------------

# Least-squares polynomial of order k through the origin (zero dose → zero level):
# y = b1·x + b2·x² + … + bk·xᵏ. Statistics follow R's summary(lm(y ~ 0 + …)):
# without an intercept, R² and F are computed about zero (uncentered).
function polyfit(x::Vector{Float64}, y::Vector{Float64}, k::Int)
    n = length(y)
    X = reduce(hcat, [x .^ i for i in 1:k])
    F = qr(X)
    coef = F \ y
    resid = y .- X * coef
    df = n - k
    rss = sum(abs2, resid)
    σ² = rss / df
    XtX_inv = inv(cholesky(Symmetric(X' * X)))
    se = sqrt.(σ² .* diag(XtX_inv))
    t = coef ./ se
    p = 2 .* Distributions.ccdf.(Distributions.TDist(df), abs.(t))
    tss = sum(abs2, y) # uncentered
    r2 = 1 - rss / tss
    adj_r2 = 1 - (1 - r2) * n / df
    f = ((tss - rss) / k) / σ²
    f_p = Distributions.ccdf(Distributions.FDist(k, df), f)
    # Gaussian log-likelihood, σ counted as a parameter (as R's logLik)
    loglik = -n / 2 * (log(2π) + log(rss / n) + 1)
    aic = -2loglik + 2(k + 1)
    bic = -2loglik + log(n) * (k + 1)
    # leave-one-out cross-validation from the hat matrix
    h = vec(sum((X * XtX_inv) .* X, dims = 2))
    loocv_rmse = sqrt(mean(abs2, resid ./ (1 .- h)))
    # must be usable for low doses: non-negative and non-decreasing on [0, LOW_DOSE_MAX]
    d = 0:0.5:LOW_DOSE_MAX
    c = [sum(coef[i] * dose^i for i in 1:k) for dose in d]
    valid_low_dose = all(>=(0), c) && all(>=(0), diff(c))
    return (order = k, n = n, df = df, coef = coef, se = se, t = t, p = p,
            rse = sqrt(σ²), r2 = r2, adj_r2 = adj_r2, f = f, f_p = f_p,
            loglik = loglik, aic = aic, bic = bic, loocv_rmse = loocv_rmse,
            valid_low_dose = valid_low_dose, selected = false)
end

poly_formula(r) = "conc = " * join(["$(r.coef[i]) × td" * (i > 1 ? "^$i" : "") for i in 1:r.order], " + ")

# one CSV row; coefficient columns b1…b$(POLY_MAX_ORDER) are missing above the model order
function poly_row(r, var)
    row = Dict{Symbol,Any}(:target => var, :order => r.order, :selected => r.selected, :n => r.n, :df => r.df,
                           :rse => r.rse, :r2 => r.r2, :adj_r2 => r.adj_r2,
                           :f => r.f, :f_df1 => r.order, :f_df2 => r.df, :f_p => r.f_p,
                           :loglik => r.loglik, :aic => r.aic, :bic => r.bic, :loocv_rmse => r.loocv_rmse,
                           :valid_low_dose => r.valid_low_dose)
    for i in 1:POLY_MAX_ORDER
        for (name, v) in (("b", r.coef), ("se_b", r.se), ("t_b", r.t), ("p_b", r.p))
            row[Symbol(name, i)] = i <= r.order ? v[i] : missing
        end
    end
    cols = vcat([:target, :order, :selected, :n, :df],
                [Symbol(name, i) for name in ("b", "se_b", "t_b", "p_b") for i in 1:POLY_MAX_ORDER],
                [:rse, :r2, :adj_r2, :f, :f_df1, :f_df2, :f_p, :loglik, :aic, :bic, :loocv_rmse, :valid_low_dose])
    return NamedTuple(c => row[c] for c in cols)
end

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

isfile("data/clozapine.csv") || error("File data/clozapine.csv cannot be opened!")
println("Loading: clozapine.csv")
raw_data = CSV.read("data/clozapine.csv", header = true, DataFrame)

# ---------------------------------------------------------------------------
# polynomial dose-concentration models (all observations, no outlier removal)
# ---------------------------------------------------------------------------

println()
println("Polynomial models: concentration ~ total dose")
println("=" ^ 60)

poly_td = Float64.(raw_data.morning_dose .+ raw_data.midday_dose .+ raw_data.evening_dose)
poly_rows = [polyfit(poly_td, Float64.(raw_data[!, var]), k) for var in ("clozapine", "norclozapine") for k in 1:POLY_MAX_ORDER]
poly_vars = [var for var in ("clozapine", "norclozapine") for _ in 1:POLY_MAX_ORDER]
poly_selected = Dict{String,Vector{Float64}}()
for var in ("clozapine", "norclozapine")
    idx = findall(==(var), poly_vars)
    valid = filter(i -> poly_rows[i].valid_low_dose, idx)
    isempty(valid) && error("No polynomial model of $var concentration is non-negative and non-decreasing for total doses 0-$LOW_DOSE_MAX mg!")
    best = valid[argmin([poly_rows[i].bic for i in valid])]
    for i in idx
        poly_rows[i] = merge(poly_rows[i], (selected = i == best,))
    end
    poly_selected[var] = poly_rows[best].coef
    for i in idx
        r = poly_rows[i]
        println("  $var, order $(r.order): BIC = $(round(r.bic, digits = 2)), adj. R² = $(round(r.adj_r2, digits = 4)), RSE = $(round(r.rse, digits = 1))",
                r.valid_low_dose ? "" : " (invalid for low doses)", r.selected ? "  ← selected" : "")
    end
    println("  $var: ", poly_formula(poly_rows[best]))
end
poly_df = DataFrame([poly_row(r, var) for (r, var) in zip(poly_rows, poly_vars)])
println("Saving: reports/polynomial_models.csv")
CSV.write("reports/polynomial_models.csv", poly_df)
println("Saving: models/polynomial_models.jld")
JLD2.save_object("models/polynomial_models.jld", poly_selected)

# remove extreme (> mean + 3×SD) clozapine outliers
clo_outlier = mean(raw_data.clozapine) + 3 * std(raw_data.clozapine)
clo_outliers_idx = raw_data.clozapine .> clo_outlier
println("Number of extreme clozapine outliers removed: $(count(clo_outliers_idx))")
raw_data = raw_data[.!clo_outliers_idx, :]

# derived dose features (computed from the three doses; data/clozapine.csv columns
# total_dose / mm_ratio are not used)
raw_data.total_dose = raw_data.morning_dose .+ raw_data.midday_dose .+ raw_data.evening_dose
all(>(0), raw_data.total_dose) || error("Total dose must be > 0 for all observations!")
raw_data.mm = (raw_data.morning_dose .+ MM_MIDDAY_WEIGHT .* raw_data.midday_dose) ./ raw_data.total_dose

for c in CYP_FEATURES
    all(in(CYP_LEVELS), raw_data[!, c]) || error("Column $c must contain integers in $CYP_LEVELS!")
end
all(in(MALE_LEVELS), raw_data.male) || error("Column male must be 0 or 1!")

# CYP levels present in the data: embeddings of absent levels are never trained, so
# src/server.jl maps such inputs to the nearest lower level seen in training
cyp_levels_seen = Dict(c => sort(unique(Int.(raw_data[!, c]))) for c in CYP_FEATURES)
for c in CYP_FEATURES
    println("  Levels of $c in training data: $(cyp_levels_seen[c])")
end

# synthetic zero-dose observations (hard constraint: zero dose → zero level),
# one per patient, same covariates; MM keeps the patient's value, as the dose
# distribution is undefined for a zero dose
zero_data = copy(raw_data)
zero_data[!, [:morning_dose, :midday_dose, :evening_dose, :total_dose]] .= 0
zero_data[!, [:clozapine, :norclozapine]] .= 0

# Augmented data set: real observations twice (2:1 real:synthetic weighting), then
# synthetic. Row i, i + n and i + 2n belong to patient i.
n = nrow(raw_data)
data = vcat(raw_data, raw_data, zero_data)
patient = repeat(1:n, 3)
real_idx = collect(1:n)
println("Number of real observations: $n")
println("Number of synthetic observations added: $n")

# split 80:20 by patient, so that no patient (or its copy / zero-dose row) appears
# in both training and testing; test only on real observations
println("Splitting 80:20 (by patient)")
train_p, test_p = MLJ.partition(1:n, 0.80, shuffle = true, rng = MersenneTwister(SEED))
train_idx = findall(in(Set(train_p)), patient)
test_idx = collect(test_p) # real rows only
train_real_idx = collect(train_p) # real training rows, for reporting

# CV folds over training patients (indices relative to train_idx); each validation
# fold contains real observations only, its patients are excluded from training
function patient_folds(train_idx, patient, n_real, nfolds, rng)
    tp = shuffle(rng, unique(patient[train_idx]))
    chunks = [tp[k:nfolds:end] for k in 1:nfolds]
    folds = Tuple{Vector{Int},Vector{Int}}[]
    for c in chunks
        s = Set(c)
        tr = [i for (i, r) in enumerate(train_idx) if !(patient[r] in s)]
        te = [i for (i, r) in enumerate(train_idx) if patient[r] in s && r <= n_real]
        push!(folds, (tr, te))
    end
    return folds
end
folds = patient_folds(train_idx, patient, n, NFOLDS, MersenneTwister(SEED))

clo_level = Float64.(data.clozapine)
nclo_level = Float64.(data.norclozapine)

# ---------------------------------------------------------------------------
# train all requested variants
# ---------------------------------------------------------------------------

for (variant, (dose_features, suffix)) in VARIANTS

    variant in requested || continue

    colnames = features(dose_features)
    X = Matrix{Float64}(data[!, colnames])

    println()
    println("Model variant: $variant")
    println("Features: $(join(colnames, ", "))")
    println("=" ^ 60)

    # feature scaler fitted on training rows (tuning) and on all rows (final model)
    X_df, _ = make_df(X, colnames, train_idx)
    X_df_all, scaler_features = make_df(X, colnames, eachindex(clo_level))
    println("  Number of observations: $(nrow(X_df))")
    println("  Number of features: $(ncol(X_df))")

    results = Dict{String,Any}()
    for (var, label, y) in (("clozapine", "clo", clo_level), ("norclozapine", "nclo", nclo_level))
        println()
        println("Creating regressor model: $var ($variant)")
        println("-" ^ 60)

        scaler_y = StatsBase.fit(ZScoreTransform, y[train_idx])
        model, ŷ_test = train(X_df, y, scaler_y, train_idx, train_real_idx, test_idx, folds, var, variant, suffix)

        if var == "clozapine"
            println("  Clozapine level groups (testing)")
            print_groups(y[test_idx], ŷ_test)
        end

        println("Training the final model")
        scaler_y = StatsBase.fit(ZScoreTransform, y)
        mach, ŷ = train_final(model, X_df_all, y, scaler_y, real_idx, var, variant, suffix)

        if var == "clozapine"
            println("  Clozapine level groups (all observations, in-sample)")
            print_groups(y[real_idx], ŷ)
        end

        DataFrame(Symbol(label) => y[real_idx], Symbol(label * "_pred") => ŷ) |>
            CSV.write("reports/$(label)_errors$(suffix).csv")

        results[var] = (mach, scaler_y)
    end

    println()
    for (var, fname) in (("clozapine", "clozapine"), ("norclozapine", "nclozapine"))
        mach, scaler_y = results[var]
        println("Saving: $(var)_regressor_machine$(suffix).jlso")
        MLJ.save("models/$(var)_regressor_machine$(suffix).jlso", mach)
        println("Saving: scaler_$(fname)_level$(suffix).jld")
        JLD2.save_object("models/scaler_$(fname)_level$(suffix).jld", scaler_y)
    end
    println("Saving: scaler_features$(suffix).jld")
    JLD2.save_object("models/scaler_features$(suffix).jld", scaler_features)

    save_performance() # after each variant, so that results survive an interrupted run
    println("Saving: $(PERFORMANCE_FILE)")

end

println("Saving: cyp_levels.jld")
JLD2.save_object("models/cyp_levels.jld", cyp_levels_seen)

println()
println("Training completed")
println()
