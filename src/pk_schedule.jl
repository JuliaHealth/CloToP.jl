# Pharmacokinetic (PK) modelling of the effect of the dosing schedule on the morning trough
# clozapine (CLO) concentration (secondary aim of the CloToP 2.0 manuscript).
#
# Usage:
#   julia src/pk_schedule.jl 2>&1 | tee reports/pk_schedule.txt
#
# One-compartment model with first-order absorption at steady state (superposition of the
# daily doses, dosing interval 24 h); doses at 8:00 (morning), 14:00 (midday), and 20:00
# (evening); sampling immediately before the morning dose, i.e., 24, 18, and 12 h after the
# morning, midday, and evening doses (as in the derivation of the MM weight in the Methods).
#
# 1. Typical patient (deterministic): AGNP 2026 reference parameters (ka 1.66 1/h,
#    ke 0.058 1/h, CL/F 20.16 L/h) and Li et al. 2012 typical parameters (female non-smoker:
#    CL/F 21.9 L/h, V/F 526 L, ka 1.3 1/h): concentration-time curves and trough vs. MM.
# 2. Simulation (real schedules, synthetic PK parameters): the morning, midday, and evening
#    doses, sex, and smoking status (≥ 1 CYP1A2 inducer) of the 122 patients (data/clozapine.csv
#    without the CLO outlier, as in src/train.jl) are combined with PK parameters drawn from the
#    Li et al. 2012 population model: CL/F = 21.9 × 1.45 (smoker) × 1.208 (male) × exp(η1),
#    V/F = 526 × exp(η2), η1 ~ N(0, 0.411²), η2 ~ N(0, 0.599²), independent; ka = 1.3 1/h;
#    no residual error. In each replicate, the simulated troughs are regressed on the total dose,
#    sex, and smoking status, without and with MM (as in src/linear_stats.jl): coefficient of MM,
#    increase in R², and power (p < 0.05). Larger samples are drawn by resampling patients with
#    replacement.
#
# Output: reports/pk_schedule.csv (simulation summary), reports/pk_curves.csv (typical
# patient), reports/pk_schedule_sims.csv (replicates, n = 122)

using CSV
using DataFrames
using Distributions
using LinearAlgebra
using Printf
using Random
using Statistics

const MM_MIDDAY_WEIGHT = 0.5861325
const DOSE_TIMES = (8.0, 14.0, 20.0) # morning, midday, evening [h]
const T_SAMPLE = 8.0                 # [h], immediately before the morning dose (as in the manuscript Methods)
const TAU = 24.0
const NSIM = 2000
const SEED = 2026
const SAMPLE_SIZES = [122, 250, 500, 1000]

# steady-state concentration [ng/mL] at clock time t for doses [mg] at clock times `times`
function css(t, doses, times; ka, ke, v)
    c = 0.0
    for (d, td) in zip(doses, times)
        d == 0 && continue
        Δ = mod(t - td, TAU)
        c += d * ka / (v * (ka - ke)) * (exp(-ke * Δ) / (1 - exp(-ke * TAU)) - exp(-ka * Δ) / (1 - exp(-ka * TAU)))
    end
    return 1000 * c # mg/L → ng/mL
end
trough(doses; kw...) = css(T_SAMPLE, doses, DOSE_TIMES; kw...)

const AGNP = (ka = 1.66, ke = 0.058, v = 20.16 / 0.058)
const LI = (ka = 1.3, ke = 21.9 / 526, v = 526.0)

# ---------------------------------------------------------------------------
# 1. typical patient
# ---------------------------------------------------------------------------

println("1. Typical patient, 300 mg/day, trough before the morning dose [ng/mL]")
for (name, p) in (("AGNP 2026", AGNP), ("Li 2012 (female non-smoker)", LI))
    e, d, m = trough([0, 0, 300]; p...), trough([0, 300, 0]; p...), trough([300, 0, 0]; p...)
    @printf("   %-28s evening %.0f, midday %.0f, morning %.0f; morning/evening %.3f; midday weight %.4f; t½ %.1f h\n",
            name, e, d, m, m / e, (e - d) / (e - m), log(2) / p.ke)
end

schedules = [("evening 300 mg", [0, 0, 300]), ("morning 300 mg", [300, 0, 0]),
             ("morning 150 + evening 150 mg", [150, 0, 150]), ("100 mg three times daily", [100, 100, 100])]
curves = DataFrame(model = String[], schedule = String[], time = Float64[], concentration = Float64[])
for (name, p) in (("AGNP 2026", AGNP), ("Li 2012", LI)), (s, doses) in schedules, t in 0:0.25:24
    push!(curves, (name, s, t, css(t, doses, DOSE_TIMES; p...)))
end
CSV.write("reports/pk_curves.csv", curves)

# ---------------------------------------------------------------------------
# 2. simulation with the real schedules
# ---------------------------------------------------------------------------

data = CSV.read("data/clozapine.csv", DataFrame)
data = data[data.clozapine .<= mean(data.clozapine) + 3 * std(data.clozapine), :]
n0 = nrow(data)
doses = [Float64[r.morning_dose, r.midday_dose, r.evening_dose] for r in eachrow(data)]
td = sum.(doses)
mm = [(d[1] + MM_MIDDAY_WEIGHT * d[2]) / sum(d) for d in doses]
male = Float64.(data.male)
smoker = Float64.(data.inducers_1a2 .> 0)
println("\n2. Simulation: real schedules of $n0 patients ($(Int(sum(male))) men, $(Int(sum(smoker))) smokers), Li 2012 PK parameters")

function ols_mm(y, td, mm, covs)
    X0 = hcat(ones(length(y)), td, covs)
    X1 = hcat(X0, mm)
    r2(X) = (b = X \ y; 1 - sum(abs2, y .- X * b) / sum(abs2, y .- mean(y)))
    b = X1 \ y
    res = y .- X1 * b
    df = length(y) - size(X1, 2)
    se = sqrt(sum(abs2, res) / df * inv(X1' * X1)[end, end])
    return (b = b[end], p = 2 * ccdf(TDist(df), abs(b[end] / se)), dr2 = r2(X1) - r2(X0))
end

function simulate(rng, idx)
    η1 = rand(rng, Normal(0, 0.411), length(idx))
    η2 = rand(rng, Normal(0, 0.599), length(idx))
    y = similar(η1)
    for (k, i) in enumerate(idx)
        cl = 21.9 * (smoker[i] == 1 ? 1.45 : 1.0) * (male[i] == 1 ? 1.208 : 1.0) * exp(η1[k])
        v = 526 * exp(η2[k])
        y[k] = trough(doses[i]; ka = 1.3, ke = cl / v, v = v)
    end
    return y
end

rng = MersenneTwister(SEED)
summary = DataFrame(n = Int[], beta_median = Float64[], beta_q025 = Float64[], beta_q975 = Float64[],
                    dr2_median = Float64[], power = Float64[], p_beta_ge_observed = Float64[],
                    mean_conc = Float64[], sd_ln_cd = Float64[])
const BETA_OBSERVED = -84.38 # CLO, n = 122 (reports/linear_stats.txt)
sims122 = DataFrame()
for n in SAMPLE_SIZES
    res = map(1:NSIM) do _
        idx = n == n0 ? collect(1:n0) : rand(rng, 1:n0, n)
        y = simulate(rng, idx)
        r = ols_mm(y, td[idx], mm[idx], hcat(male[idx], smoker[idx]))
        (r..., mean_conc = mean(y), sd_ln_cd = std(log.(y ./ td[idx])))
    end
    b = [r.b for r in res]
    push!(summary, (n, median(b), quantile(b, 0.025), quantile(b, 0.975), median([r.dr2 for r in res]),
                    mean(r.p < 0.05 for r in res), mean(b .>= BETA_OBSERVED),
                    mean(r.mean_conc for r in res), mean(r.sd_ln_cd for r in res)))
    n == n0 && (global sims122 = DataFrame(res))
end

real_sd = std(log.(data.clozapine ./ td))
println("   real data: mean CLO $(round(mean(data.clozapine), digits = 1)) ng/mL, SD of ln(C/D) $(round(real_sd, digits = 3))")
println("   n      β(MM) median [2.5–97.5%]      ΔR² median   power   P(β ≥ observed β = $BETA_OBSERVED)   mean CLO   SD ln(C/D)")
for r in eachrow(summary)
    @printf("   %-5d %8.1f [%7.1f, %7.1f]   %8.3f    %5.3f   %6.3f                        %6.1f     %.3f\n",
            r.n, r.beta_median, r.beta_q025, r.beta_q975, r.dr2_median, r.power, r.p_beta_ge_observed, r.mean_conc, r.sd_ln_cd)
end
println("   (β(MM): change in the simulated trough [ng/mL] for MM increasing from 0 to 1 at a fixed total")
println("   dose, adjusted for sex and smoking; power: proportion of replicates with p < 0.05; $NSIM replicates)")

CSV.write("reports/pk_schedule.csv", summary)
CSV.write("reports/pk_schedule_sims.csv", sims122)
println("\nSaving: reports/pk_schedule.csv, reports/pk_schedule_sims.csv, reports/pk_curves.csv")
