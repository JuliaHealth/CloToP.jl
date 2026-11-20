# Robustness of the comparison of the dose representations (model variants).
#
# Usage:
#   julia -t auto src/compare.jl verify      # reproduce the single-split testing results of src/train.jl
#   julia -t auto src/compare.jl [R [K]]     # R × K-fold patient-level CV (default: 5 × 5)
#   julia -t auto src/compare.jl [R [K]] reuse
#   julia -t auto src/compare.jl [R [K]] sensitivity [reuse]
#                                            # sensitivity analysis: also without extreme norclozapine
#                                            # levels (> mean + 3 SD); outputs reports/compare_sensitivity_*
#                                            # as above, but the neural network predictions are read
#                                            # from reports/compare_predictions.csv (references refitted)
#
# Each variant is evaluated with the hyperparameters selected by src/train.jl
# (reports/performance.csv) in repeated K-fold cross-validation at the patient level:
# in every repeat, each patient is predicted once by a model trained without that
# patient (and without its copy and zero-dose observation). Preprocessing is identical
# to src/train.jl. Reference models (training mean; quadratic model of the total dose
# without intercept; multiple linear regression with each dose representation and the
# other features of the networks) are fitted within the same folds. Models are compared on paired
# per-patient errors (averaged over the repeats): bootstrap 95% CIs (patients resampled)
# and sign-flip permutation tests.
#
# Note: the hyperparameters were tuned on the 98 training patients of src/train.jl, which
# also appear in the CV folds; this favours all neural network variants to a similar
# extent, but not the (untuned) reference models.
#
# Outputs: reports/compare_predictions.csv (out-of-fold predictions),
#          reports/compare_summary.csv, reports/compare_pairs.csv

const SEED = 42
const MM_MIDDAY_WEIGHT = 0.5861325
const NBOOT = 10_000
const NPERM = 10_000
const VERBOSE = 0

println("Loading: packages")
using CSV
using DataFrames
using MLJ
using MLJFlux
using Optimisers
using Flux
using LinearAlgebra
using Random
using StatsBase
using Printf

BLAS.set_num_threads(1)
NeuralNetworkRegressor = @MLJ.load NeuralNetworkRegressor pkg=MLJFlux verbosity=0
println("     Threads: $(Threads.nthreads())")

# ---------------------------------------------------------------------------
# definitions identical to src/train.jl
# ---------------------------------------------------------------------------

const COMMON_FEATURES_PRE = ["male", "age"]
const COMMON_FEATURES_POST = ["bmi", "crp",
                              "inducers_3a4", "inhibitors_3a4", "substrates_3a4",
                              "inducers_1a2", "inhibitors_1a2", "substrates_1a2"]
const VARIANTS = [
    "threedoses"            => ["morning_dose", "midday_dose", "evening_dose"],
    "totaldose"             => ["total_dose"],
    "totaldose_mm"          => ["total_dose", "mm"],
    "totaldose_morn_midday" => ["total_dose", "morning_dose", "midday_dose"],
]
features(dose_features) = vcat(COMMON_FEATURES_PRE, dose_features, COMMON_FEATURES_POST)
const CYP_FEATURES = COMMON_FEATURES_POST[3:end]
const CYP_LEVELS = 0:5
const MALE_LEVELS = 0:1

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
            col[isnan.(col)] .= 0
            df[!, c] = col
        end
    end
    return df, scaler
end

function predict_level(mach, X, y_scaler)
    ŷ = StatsBase.reconstruct(y_scaler, MLJ.predict(mach, X))
    return max.(ŷ, 0)
end

function build_model(; eta, lambda, alpha, epochs, batch_size)
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
                                  optimiser = Optimisers.AdamW(eta = eta),
                                  loss = Flux.Losses.mae,
                                  epochs = epochs,
                                  batch_size = batch_size,
                                  lambda = lambda,
                                  alpha = alpha,
                                  rng = SEED)
end

# ---------------------------------------------------------------------------
# data (as in src/train.jl)
# ---------------------------------------------------------------------------

raw_data = CSV.read("data/clozapine.csv", DataFrame)
clo_outlier = mean(raw_data.clozapine) + 3 * std(raw_data.clozapine)
raw_data = raw_data[raw_data.clozapine .<= clo_outlier, :]
const SENSITIVITY = "sensitivity" in ARGS
const OUT = SENSITIVITY ? "reports/compare_sensitivity_" : "reports/compare_"
if SENSITIVITY
    nclo_outlier = mean(raw_data.norclozapine) + 3 * std(raw_data.norclozapine)
    println("Sensitivity analysis: removing norclozapine > $(round(nclo_outlier, digits = 1)) ng/mL: ",
            raw_data.norclozapine[raw_data.norclozapine .> nclo_outlier])
    raw_data = raw_data[raw_data.norclozapine .<= nclo_outlier, :]
end
raw_data.total_dose = raw_data.morning_dose .+ raw_data.midday_dose .+ raw_data.evening_dose
raw_data.mm = (raw_data.morning_dose .+ MM_MIDDAY_WEIGHT .* raw_data.midday_dose) ./ raw_data.total_dose
zero_data = copy(raw_data)
zero_data[!, [:morning_dose, :midday_dose, :evening_dose, :total_dose]] .= 0
zero_data[!, [:clozapine, :norclozapine]] .= 0
n = nrow(raw_data)
data = vcat(raw_data, raw_data, zero_data)
patient = repeat(1:n, 3)
td_real = Float64.(raw_data.total_dose)
const TARGETS = ("clozapine", "norclozapine")
Y = Dict(t => Float64.(data[!, t]) for t in TARGETS)
XM = Dict(v => Matrix{Float64}(data[!, features(df)]) for (v, df) in VARIANTS)
println("Patients: $n")

# tuned hyperparameters
perf = CSV.read("reports/performance.csv", DataFrame, stringtype = String)
function hyper(variant, target)
    r = perf[(perf.variant .== variant) .& (perf.target .== target) .& (perf.model .== "final"), :]
    nrow(r) == 1 || error("No tuned parameters for $variant / $target in reports/performance.csv")
    r = r[1, :]
    return (eta = r.eta, lambda = r.lambda, alpha = r.alpha, epochs = r.epochs, batch_size = r.batch_size)
end

# Train a variant on the rows of patients `train_p` and predict the real rows of `test_p`.
function fit_predict(variant, target, train_p, test_p)
    s = Set(train_p)
    train_idx = findall(p -> p in s, patient)
    colnames = features(Dict(VARIANTS)[variant])
    X_df, _ = make_df(XM[variant], colnames, train_idx)
    y = Y[target]
    scaler_y = StatsBase.fit(ZScoreTransform, y[train_idx])
    mach = machine(build_model(; hyper(variant, target)...), X_df[train_idx, :],
                   StatsBase.transform(scaler_y, y)[train_idx], scitype_check_level = 1)
    MLJ.fit!(mach, force = true, verbosity = VERBOSE)
    return predict_level(mach, X_df[collect(test_p), :], scaler_y)
end

function references(target, train_p, test_p)
    y = Y[target][1:n]
    mean_pred = fill(mean(y[train_p]), length(test_p))
    Xq = hcat(td_real, td_real .^ 2)
    b = Xq[train_p, :] \ y[train_p]
    return mean_pred, max.(Xq[collect(test_p), :] * b, 0)
end

# Multiple linear regression (OLS with intercept) on the real observations, with the same
# features as the neural networks (CYP counts and sex as numeric covariates). The three doses
# variant spans the same linear space as total dose + morning/midday doses and is therefore
# not fitted separately; "shares" uses the morning and midday fractions of the total dose.
const LINEAR_VARIANTS = [
    "linear_totaldose"             => ["total_dose"],
    "linear_totaldose_mm"          => ["total_dose", "mm"],
    "linear_totaldose_morn_midday" => ["total_dose", "morning_dose", "midday_dose"],
    "linear_totaldose_shares"      => ["total_dose", "morning_share", "midday_share"],
]
raw_data.morning_share = raw_data.morning_dose ./ raw_data.total_dose
raw_data.midday_share = raw_data.midday_dose ./ raw_data.total_dose
XL = Dict(v => Matrix{Float64}(raw_data[!, features(df)]) for (v, df) in LINEAR_VARIANTS)

function linear_reference(variant, target, train_p, test_p)
    X = XL[variant]
    keep = [j for j in axes(X, 2) if length(unique(X[train_p, j])) > 1]   # drop constant columns
    A = hcat(ones(n), X[:, keep])
    b = A[train_p, :] \ Y[target][train_p]
    return max.(A[collect(test_p), :] * b, 0)
end

rmse(e) = sqrt(mean(abs2, e))
r2(y, ŷ) = 1 - sum(abs2, y .- ŷ) / sum(abs2, y .- mean(y))

# ---------------------------------------------------------------------------
# verify: reproduce the single split of src/train.jl
# ---------------------------------------------------------------------------

if get(ARGS, 1, "") == "verify"
    train_p, test_p = MLJ.partition(1:n, 0.80, shuffle = true, rng = MersenneTwister(SEED))
    jobs = [(v, t) for (v, _) in VARIANTS for t in TARGETS]
    res = Vector{Any}(undef, length(jobs))
    Threads.@threads for i in eachindex(jobs)
        res[i] = fit_predict(jobs[i]..., train_p, test_p)
    end
    for ((v, t), ŷ) in zip(jobs, res)
        y = Y[t][test_p]
        r = perf[(perf.variant .== v) .& (perf.target .== t) .& (perf.model .== "tuned") .& (perf.dataset .== "testing"), :][1, :]
        @printf("%-22s %-13s RMSE %7.2f (train.jl %7.2f)  R² %6.3f (%6.3f)\n", v, t, rmse(y .- ŷ), r.rmse, r2(y, ŷ), r.r2)
    end
    exit()
end

# ---------------------------------------------------------------------------
# repeated K-fold patient-level cross-validation
# ---------------------------------------------------------------------------

const REUSE = "reuse" in ARGS
const NUM_ARGS = filter(a -> !(a in ("reuse", "sensitivity")), ARGS)
const R = parse(Int, get(NUM_ARGS, 1, "5"))
const K = parse(Int, get(NUM_ARGS, 2, "5"))
println("Repeated cross-validation: $R × $K-fold, by patient")

folds = Tuple{Int,Vector{Int},Vector{Int}}[]
for r in 1:R
    p = shuffle(MersenneTwister(SEED + r), 1:n)
    for k in 1:K
        test_p = sort(p[k:K:end])
        push!(folds, (r, setdiff(1:n, test_p), test_p))
    end
end

models = vcat(first.(VARIANTS), ["reference_mean", "reference_quadratic"], first.(LINEAR_VARIANTS))
# out-of-fold predictions: pred[(target, model)][patient, repeat]
pred = Dict((t, m) => fill(NaN, n, R) for t in TARGETS for m in models)

if REUSE
    println("Reading neural network predictions: $(OUT)predictions.csv")
    old = CSV.read("$(OUT)predictions.csv", DataFrame, stringtype = String)
    for row in eachrow(old)
        if row.model in first.(VARIANTS)
            row.repeat <= R || error("$(OUT)predictions.csv has more repeats than R = $R")
            row.observed == Y[row.target][row.patient] || error("$(OUT)predictions.csv does not match the data")
            pred[(row.target, row.model)][row.patient, row.repeat] = row.predicted
        end
    end
else
    jobs = [(v, t, f) for (v, _) in VARIANTS for t in TARGETS for f in eachindex(folds)]
    println("Fitting $(length(jobs)) neural networks")
    done = Threads.Atomic{Int}(0)
    t0 = time()
    Threads.@threads :dynamic for (v, t, f) in jobs
        r, train_p, test_p = folds[f]
        pred[(t, v)][test_p, r] = fit_predict(v, t, train_p, test_p)
        d = Threads.atomic_add!(done, 1) + 1
        d % 20 == 0 && @printf("  %d/%d fitted (%.0f min)\n", d, length(jobs), (time() - t0) / 60)
    end
end
for t in TARGETS, (r, train_p, test_p) in folds
    m, q = references(t, train_p, test_p)
    pred[(t, "reference_mean")][test_p, r] = m
    pred[(t, "reference_quadratic")][test_p, r] = q
    for (v, _) in LINEAR_VARIANTS
        pred[(t, v)][test_p, r] = linear_reference(v, t, train_p, test_p)
    end
end
@assert !any(any(isnan, p) for p in values(pred))

out = DataFrame(target = String[], model = String[], repeat = Int[], patient = Int[], observed = Float64[], predicted = Float64[])
for t in TARGETS, m in models, r in 1:R, i in 1:n
    push!(out, (t, m, r, i, Y[t][i], pred[(t, m)][i, r]))
end
CSV.write("$(OUT)predictions.csv", out)
println("Saving: $(OUT)predictions.csv")

# ---------------------------------------------------------------------------
# summary: metrics of the pooled out-of-fold predictions, mean (SD) over repeats
# ---------------------------------------------------------------------------

summary = DataFrame(target = String[], model = String[], rmse = Float64[], rmse_sd = Float64[],
                    mae = Float64[], mae_sd = Float64[], r2 = Float64[], r2_sd = Float64[],
                    accuracy = Float64[], sensitivity = Float64[], specificity = Float64[])
for t in TARGETS, m in models
    y = Y[t][1:n]
    rm = [rmse(y .- pred[(t, m)][:, r]) for r in 1:R]
    ma = [mean(abs.(y .- pred[(t, m)][:, r])) for r in 1:R]
    rs = [r2(y, pred[(t, m)][:, r]) for r in 1:R]
    acc = sens = spec = NaN
    if t == "clozapine"   # toxic: CLO > 550 ng/mL, pooled over repeats
        obs = repeat(y .> 550, R)
        prd = vec(pred[(t, m)]) .> 550
        acc = mean(obs .== prd); sens = mean(prd[obs]); spec = mean(.!prd[.!obs])
    end
    push!(summary, (t, m, mean(rm), std(rm), mean(ma), std(ma), mean(rs), std(rs), acc, sens, spec))
end
CSV.write("$(OUT)summary.csv", summary)
println("Saving: $(OUT)summary.csv")

# ---------------------------------------------------------------------------
# paired comparisons on per-patient errors (averaged over repeats)
# ---------------------------------------------------------------------------

abs_err(t, m) = vec(mean(abs.(Y[t][1:n] .- pred[(t, m)]), dims = 2))
sq_err(t, m) = vec(mean(abs2.(Y[t][1:n] .- pred[(t, m)]), dims = 2))

function boot_ci(f, a, b; rng)
    s = [f(a[idx], b[idx]) for idx in (rand(rng, 1:n, n) for _ in 1:NBOOT)]
    return quantile(s, 0.025), quantile(s, 0.975)
end
# two-sided sign-flip permutation test of mean(d) = 0
function perm_p(d; rng)
    obs = abs(mean(d))
    c = count(_ -> abs(mean(d .* rand(rng, (-1, 1), length(d)))) >= obs, 1:NPERM)
    return (c + 1) / (NPERM + 1)
end

pairs = DataFrame(target = String[], model_1 = String[], model_2 = String[],
                  delta_mae = Float64[], delta_mae_lo = Float64[], delta_mae_hi = Float64[], p_mae = Float64[],
                  delta_rmse = Float64[], delta_rmse_lo = Float64[], delta_rmse_hi = Float64[], p_mse = Float64[],
                  patients_better = Int[])
nn = first.(VARIANTS)
lin = first.(LINEAR_VARIANTS)
# network vs the linear model with the same dose information (three doses: same linear space
# as total dose + morning/midday doses)
lin_match = Dict("threedoses" => "linear_totaldose_morn_midday", "totaldose" => "linear_totaldose",
                 "totaldose_mm" => "linear_totaldose_mm", "totaldose_morn_midday" => "linear_totaldose_morn_midday")
comparisons = vcat([(a, b) for (i, a) in enumerate(nn) for b in nn[i+1:end]],
                   [(a, ref) for a in nn for ref in ("reference_quadratic", "reference_mean")],
                   [(a, b) for (i, a) in enumerate(lin) for b in lin[i+1:end]],
                   [(a, "reference_quadratic") for a in lin],
                   [(a, lin_match[a]) for a in nn])
for t in TARGETS, (m1, m2) in comparisons
    rng = MersenneTwister(SEED)
    a1, a2 = abs_err(t, m1), abs_err(t, m2)
    s1, s2 = sq_err(t, m1), sq_err(t, m2)
    dmae = mean(a1) - mean(a2)
    lo, hi = boot_ci((x, y) -> mean(x) - mean(y), a1, a2; rng)
    drmse = sqrt(mean(s1)) - sqrt(mean(s2))
    lo2, hi2 = boot_ci((x, y) -> sqrt(mean(x)) - sqrt(mean(y)), s1, s2; rng)
    push!(pairs, (t, m1, m2, dmae, lo, hi, perm_p(a1 .- a2; rng), drmse, lo2, hi2, perm_p(s1 .- s2; rng), count(a1 .< a2)))
end
CSV.write("$(OUT)pairs.csv", pairs)
println("Saving: $(OUT)pairs.csv")

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

for t in TARGETS
    println()
    println("$t: out-of-fold performance, mean (SD) over $R repeats of $K-fold CV")
    for r in eachrow(summary[summary.target .== t, :])
        @printf("  %-28s RMSE %6.1f (%4.1f)  MAE %6.1f (%4.1f)  R² %5.2f (%4.2f)", r.model, r.rmse, r.rmse_sd, r.mae, r.mae_sd, r.r2, r.r2_sd)
        t == "clozapine" ? @printf("  acc %.2f sens %.2f spec %.2f\n", r.accuracy, r.sensitivity, r.specificity) : println()
    end
    println("  paired differences (model 1 − model 2), 95% bootstrap CI, permutation p")
    for r in eachrow(pairs[pairs.target .== t, :])
        @printf("  %-28s vs %-28s ΔMAE %6.1f [%6.1f, %6.1f] p=%.3f  ΔRMSE %6.1f [%6.1f, %6.1f] p=%.3f  better in %d/%d\n",
                r.model_1, r.model_2, r.delta_mae, r.delta_mae_lo, r.delta_mae_hi, r.p_mae,
                r.delta_rmse, r.delta_rmse_lo, r.delta_rmse_hi, r.p_mse, r.patients_better, n)
    end
end
