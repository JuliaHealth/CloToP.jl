# Contribution of the dosing schedule to the predictions of the deployed models (neural
# network and linear regression with the total dose + morning/midday doses; item 8 of the
# manuscript task list).
#
# Usage (models written by src/train.jl):
#   julia src/schedule_effect.jl 2>&1 | tee reports/schedule_effect.txt
#
# The models are loaded by evaluating src/server.jl without starting the HTTP server.
# Patients: data/clozapine.csv without the CLO outlier (n = 122), as in src/train.jl.
#
# 1. Schedule effect at a fixed total dose: for each patient, the CLO and NCLO concentrations
#    are predicted for the patient's own total daily dose, given in the evening only (MM = 0)
#    and shifted to the morning in steps of 5% up to one half (morning fraction f ≤ 0.5, the
#    observed range; no midday dose); reported
#    as the mean ratio to the evening-only prediction, compared with the ratio expected under
#    the one-compartment PK model (src/pk_schedule.jl; AGNP 2026 and Li et al. 2012 typical
#    parameters).
# 2. Permutation importance (in-sample): increase in MAE when the schedule (morning, midday,
#    and evening fractions of the dose) is permuted between patients while each patient keeps
#    their total dose; for comparison, when the total dose is permuted while each patient keeps
#    their schedule (fractions). 200 permutations.
#
# Output: reports/schedule_effect.csv (ratios), reports/schedule_importance.csv

using CSV
using Printf
using Random

# load the models and prediction functions of the server without starting it
let src = read("src/server.jl", String)
    i = findlast("HTTP.serve(", src)
    include_string(Main, src[1:first(i)-1], "src/server.jl")
end

const NPERM = 200
const SEED = 2026

data = CSV.read("data/clozapine.csv", DataFrame)
data = data[data.clozapine .<= mean(data.clozapine) + 3 * std(data.clozapine), :]
n = nrow(data)
# patient vector as used by ctp() in src/server.jl
pvec(r) = Float64[r.male, r.age, 0, 0, 0, r.bmi, r.crp, r.inducers_3a4, r.inhibitors_3a4, r.substrates_3a4,
                  r.inducers_1a2, r.inhibitors_1a2, r.substrates_1a2]
P = [pvec(r) for r in eachrow(data)]
M, D, E = Float64.(data.morning_dose), Float64.(data.midday_dose), Float64.(data.evening_dose)
TD = M .+ D .+ E

predict_all(morning, midday, evening) = begin
    rows = [feature_dict(P[i], morning[i], midday[i], evening[i]) for i in 1:n]
    (clo_nn = predict_nn(CLO_NN, rows), nclo_nn = predict_nn(NCLO_NN, rows),
     clo_lin = predict_linear(CLO_LIN, rows), nclo_lin = predict_linear(NCLO_LIN, rows))
end

# ---------------------------------------------------------------------------
# 1. schedule effect at a fixed total dose
# ---------------------------------------------------------------------------

# PK ratio morning fraction f vs evening only (trough before the morning dose; linear in f)
# (one-compartment model of src/pk_schedule.jl: dose 24 h vs 12 h before sampling)
pk_term(ka, ke, Δ) = exp(-ke * Δ) / (1 - exp(-ke * 24)) - exp(-ka * Δ) / (1 - exp(-ka * 24))
pk_ratio(ka, ke) = pk_term(ka, ke, 24) / pk_term(ka, ke, 12)
const PK_RATIO_AGNP = pk_ratio(1.66, 0.058)
const PK_RATIO_LI = pk_ratio(1.3, 21.9 / 526)

fr = 0:0.05:0.5 # observed morning fractions: ≤ 0.5 in 121 of 122 patients
base = predict_all(zeros(n), zeros(n), TD)
effect = DataFrame(morning_fraction = Float64[], target = String[], model = String[], mean_ratio = Float64[],
                   median_ratio = Float64[], q25 = Float64[], q75 = Float64[])
for f in fr
    pr = predict_all(f .* TD, zeros(n), (1 - f) .* TD)
    for (target, key, bkey) in (("clozapine", :clo_nn, :clo_nn), ("clozapine", :clo_lin, :clo_lin),
                                ("norclozapine", :nclo_nn, :nclo_nn), ("norclozapine", :nclo_lin, :nclo_lin))
        ok = base[bkey] .> 0
        r = pr[key][ok] ./ base[bkey][ok]
        push!(effect, (f, target, occursin("nn", String(key)) ? "neural network" : "linear regression",
                       mean(r), median(r), quantile(r, 0.25), quantile(r, 0.75)))
    end
    push!(effect, (f, "clozapine", "PK AGNP 2026", 1 - f * (1 - PK_RATIO_AGNP), NaN, NaN, NaN))
    push!(effect, (f, "clozapine", "PK Li 2012", 1 - f * (1 - PK_RATIO_LI), NaN, NaN, NaN))
end
println("1. Predicted concentration, morning fraction f of the patient's own total dose (rest in the evening),")
println("   relative to evening-only dosing: mean ratio (median; IQR) over $n patients")
for target in ("clozapine", "norclozapine"), model in unique(effect.model)
    e = effect[(effect.target .== target) .& (effect.model .== model), :]
    isempty(e) && continue
    @printf("   %-12s %-18s", target, model)
    for f in (0.25, 0.5)
        r = e[e.morning_fraction .== f, :][1, :]
        isnan(r.median_ratio) ? @printf("  f = %.2f: %.3f", f, r.mean_ratio) :
            @printf("  f = %.2f: %.3f (%.3f; %.3f–%.3f)", f, r.mean_ratio, r.median_ratio, r.q25, r.q75)
    end
    println()
end
println("   linear regression coefficients (CLO): morning ", round(CLO_LIN["morning_dose"], digits = 3),
        ", midday ", round(CLO_LIN["midday_dose"], digits = 3), ", total ", round(CLO_LIN["total_dose"], digits = 3), " ng/mL per mg")
println("   linear regression coefficients (NCLO): morning ", round(NCLO_LIN["morning_dose"], digits = 3),
        ", midday ", round(NCLO_LIN["midday_dose"], digits = 3), ", total ", round(NCLO_LIN["total_dose"], digits = 3), " ng/mL per mg")

# ---------------------------------------------------------------------------
# 2. permutation importance
# ---------------------------------------------------------------------------

obs = (clo = Float64.(data.clozapine), nclo = Float64.(data.norclozapine))
own = predict_all(M, D, E)
mae(a, b) = mean(abs.(a .- b))
mae0 = Dict(:clo_nn => mae(own.clo_nn, obs.clo), :clo_lin => mae(own.clo_lin, obs.clo),
            :nclo_nn => mae(own.nclo_nn, obs.nclo), :nclo_lin => mae(own.nclo_lin, obs.nclo))
rng = MersenneTwister(SEED)
inc = Dict((k, w) => Float64[] for k in keys(mae0), w in ("schedule", "total dose"))
for _ in 1:NPERM
    π = randperm(rng, n)
    # schedule permuted, own total dose
    ps = predict_all(M[π] ./ TD[π] .* TD, D[π] ./ TD[π] .* TD, E[π] ./ TD[π] .* TD)
    # total dose permuted, own schedule
    pt = predict_all(M ./ TD .* TD[π], D ./ TD .* TD[π], E ./ TD .* TD[π])
    for k in keys(mae0)
        push!(inc[(k, "schedule")], mae(ps[k], k in (:clo_nn, :clo_lin) ? obs.clo : obs.nclo) - mae0[k])
        push!(inc[(k, "total dose")], mae(pt[k], k in (:clo_nn, :clo_lin) ? obs.clo : obs.nclo) - mae0[k])
    end
end
imp = DataFrame(target = String[], model = String[], permuted = String[], mae = Float64[],
                mae_increase = Float64[], q025 = Float64[], q975 = Float64[])
println("\n2. Permutation importance (in-sample, final models): increase in MAE [ng/mL], mean [2.5–97.5%] over $NPERM permutations")
for k in (:clo_nn, :clo_lin, :nclo_nn, :nclo_lin), w in ("schedule", "total dose")
    v = inc[(k, w)]
    target = startswith(String(k), "clo") ? "clozapine" : "norclozapine"
    model = endswith(String(k), "nn") ? "neural network" : "linear regression"
    push!(imp, (target, model, w, mae0[k], mean(v), quantile(v, 0.025), quantile(v, 0.975)))
    @printf("   %-12s %-18s MAE %6.1f; %-10s permuted: %+6.1f [%+6.1f, %+6.1f]\n",
            target, model, mae0[k], w, mean(v), quantile(v, 0.025), quantile(v, 0.975))
end

CSV.write("reports/schedule_effect.csv", effect)
CSV.write("reports/schedule_importance.csv", imp)
println("\nSaving: reports/schedule_effect.csv, reports/schedule_importance.csv")
