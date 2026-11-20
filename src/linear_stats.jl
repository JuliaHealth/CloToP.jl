# Linear statistics of the clinical data on the dosing schedule (secondary aim of the
# CloToP 2.0 manuscript).
#
# Usage:
#   julia src/linear_stats.jl 2>&1 | tee reports/linear_stats.txt
#
# Data: data/clozapine.csv after removal of the clozapine outlier (> mean + 3 SD), as in
# src/train.jl (n = 122); sensitivity analysis: additionally without norclozapine levels
# > mean + 3 SD (n = 121).
#
# Analyses:
#   1. concentration-to-dose ratio (C/D = CLO / total dose) vs MM: Pearson and Spearman
#      correlation; evening-only dosing (MM = 0) vs other schedules (Mann-Whitney U test)
#   2. multiple linear regression of CLO, NCLO, C/D, and the NCLO:CLO ratio (MPR) on the total
#      dose and the covariates, without and with MM: coefficient of MM, increase in R²
#   3. total dose vs MM: Spearman correlation
#   4. ln(CLO) vs ln(total dose), adjusted for the covariates: slope with 95% CI and test of
#      slope = 1 (dose-proportional concentration)
# Covariates (as in the neural networks): sex, age, BMI, CRP, and the numbers of CYP
# inducers/inhibitors/substrates (numeric; columns constant in the data are dropped).
#
# Output: reports/linear_stats.csv

using CSV
using DataFrames
using Distributions
using LinearAlgebra
using Printf
using Statistics
using StatsBase

const MM_MIDDAY_WEIGHT = 0.5861325
const COVARIATES = ["male", "age", "bmi", "crp",
                    "inducers_3a4", "inhibitors_3a4", "substrates_3a4",
                    "inducers_1a2", "inhibitors_1a2", "substrates_1a2"]

# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

# two-sided p value of a correlation coefficient (t distribution, n - 2 degrees of freedom)
cor_p(r, n) = 2 * ccdf(TDist(n - 2), abs(r) * sqrt((n - 2) / (1 - r^2)))

# Mann-Whitney U test, normal approximation with tie and continuity corrections
function mann_whitney(x, y)
    nx, ny = length(x), length(y)
    rk = tiedrank(vcat(x, y))
    u = sum(rk[1:nx]) - nx * (nx + 1) / 2
    n = nx + ny
    t = sum(c^3 - c for c in values(countmap(vcat(x, y))))
    σ = sqrt(nx * ny / 12 * ((n + 1) - t / (n * (n - 1))))
    z = (abs(u - nx * ny / 2) - 0.5) / σ
    return u, 2 * ccdf(Normal(), z)
end

# ordinary least squares with an intercept
function ols(y, X, names)
    A = hcat(ones(length(y)), X)
    b = A \ y
    res = y .- A * b
    n, k = size(A)
    df = n - k
    s2 = sum(abs2, res) / df
    se = sqrt.(diag(s2 * inv(A' * A)))
    t = b ./ se
    tq = quantile(TDist(df), 0.975)
    r2 = 1 - sum(abs2, res) / sum(abs2, y .- mean(y))
    return (names = vcat("intercept", names), b = b, se = se, t = t, p = 2 .* ccdf.(TDist(df), abs.(t)),
            lo = b .- tq .* se, hi = b .+ tq .* se, r2 = r2, adj_r2 = 1 - (1 - r2) * (n - 1) / df,
            rss = sum(abs2, res), df = df, n = n)
end
coef_of(m, name) = (i = findfirst(==(name), m.names); (b = m.b[i], se = m.se[i], lo = m.lo[i], hi = m.hi[i], p = m.p[i]))

# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------

results = DataFrame(dataset = String[], analysis = String[], outcome = String[], term = String[],
                    estimate = Float64[], lo = Float64[], hi = Float64[], p = Float64[],
                    r2_without = Float64[], r2_with = Float64[], n = Int[])

function analyse(d::DataFrame, label::String)
    n = nrow(d)
    td = Float64.(d.morning_dose .+ d.midday_dose .+ d.evening_dose)
    mm = (d.morning_dose .+ MM_MIDDAY_WEIGHT .* d.midday_dose) ./ td
    clo = Float64.(d.clozapine)
    nclo = Float64.(d.norclozapine)
    cd = clo ./ td
    mpr = nclo ./ clo
    covs = [c for c in COVARIATES if length(unique(d[!, c])) > 1]
    C = Matrix{Float64}(d[!, covs])

    println()
    println("=" ^ 78)
    println("$label (n = $n); covariates: ", join(covs, ", "))
    println("=" ^ 78)

    # 1. C/D vs MM
    r = cor(cd, mm); ρ = corspearman(cd, mm)
    println("\n1. Concentration-to-dose ratio (C/D, [ng/mL]/[mg/day]) vs MM")
    @printf("   Pearson r = %.3f (p = %.3f); Spearman ρ = %.3f (p = %.3f)\n", r, cor_p(r, n), ρ, cor_p(ρ, n))
    g0 = mm .== 0
    u, pu = mann_whitney(cd[g0], cd[.!g0])
    @printf("   evening dose only (MM = 0, n = %d): median C/D %.2f (IQR %.2f–%.2f)\n", count(g0), median(cd[g0]), quantile(cd[g0], 0.25), quantile(cd[g0], 0.75))
    @printf("   other schedules (MM > 0, n = %d):    median C/D %.2f (IQR %.2f–%.2f); Mann-Whitney U = %.1f, p = %.3f\n",
            count(.!g0), median(cd[.!g0]), quantile(cd[.!g0], 0.25), quantile(cd[.!g0], 0.75), u, pu)
    push!(results, (label, "C/D vs MM", "C/D", "Pearson r", r, NaN, NaN, cor_p(r, n), NaN, NaN, n))
    push!(results, (label, "C/D vs MM", "C/D", "Spearman rho", ρ, NaN, NaN, cor_p(ρ, n), NaN, NaN, n))
    push!(results, (label, "C/D vs MM", "C/D", "median MM = 0 minus MM > 0", median(cd[g0]) - median(cd[.!g0]), NaN, NaN, pu, NaN, NaN, n))

    # 2. multiple regression without and with MM
    println("\n2. Multiple linear regression on total dose + covariates, without and with MM")
    println("   outcome      β(MM)  [95% CI]                p      R² without → with  (ΔR²)   adj. R² without → with")
    for (name, y) in (("CLO", clo), ("NCLO", nclo), ("C/D", cd), ("MPR", mpr))
        m0 = ols(y, hcat(td, C), vcat("total_dose", covs))
        m1 = ols(y, hcat(td, mm, C), vcat("total_dose", "mm", covs))
        c = coef_of(m1, "mm")
        @printf("   %-6s %10.2f  [%9.2f, %9.2f]  %6.3f   %.3f → %.3f (%+.3f)   %.3f → %.3f\n",
                name, c.b, c.lo, c.hi, c.p, m0.r2, m1.r2, m1.r2 - m0.r2, m0.adj_r2, m1.adj_r2)
        push!(results, (label, "regression +MM", name, "mm", c.b, c.lo, c.hi, c.p, m0.r2, m1.r2, n))
    end
    println("   (β(MM): change in the outcome for MM increasing from 0 to 1, i.e. from evening-only to")
    println("   morning-only dosing; MPR = NCLO / CLO)")

    # 3. total dose vs MM
    ρ = corspearman(td, mm)
    @printf("\n3. Total dose vs MM: Spearman ρ = %.3f (p = %.4f)\n", ρ, cor_p(ρ, n))
    push!(results, (label, "total dose vs MM", "total dose", "Spearman rho", ρ, NaN, NaN, cor_p(ρ, n), NaN, NaN, n))

    # 4. ln(CLO) vs ln(dose), adjusted
    m = ols(log.(clo), hcat(log.(td), C), vcat("ln_total_dose", covs))
    c = coef_of(m, "ln_total_dose")
    p1 = 2 * ccdf(TDist(m.df), abs((c.b - 1) / c.se))
    mu = ols(log.(clo), reshape(log.(td), :, 1), ["ln_total_dose"])
    cu = coef_of(mu, "ln_total_dose")
    p1u = 2 * ccdf(TDist(mu.df), abs((cu.b - 1) / cu.se))
    println("\n4. ln(CLO) vs ln(total dose)")
    @printf("   adjusted:   β = %.3f (95%% CI %.3f–%.3f); test of β = 1: p = %.4f; doubling the dose → CLO × %.2f\n",
            c.b, c.lo, c.hi, p1, 2^c.b)
    @printf("   unadjusted: β = %.3f (95%% CI %.3f–%.3f); test of β = 1: p = %.4f\n", cu.b, cu.lo, cu.hi, p1u)
    push!(results, (label, "ln(CLO) vs ln(dose), adjusted", "ln CLO", "ln_total_dose (p: beta = 1)", c.b, c.lo, c.hi, p1, NaN, m.r2, n))
    push!(results, (label, "ln(CLO) vs ln(dose), unadjusted", "ln CLO", "ln_total_dose (p: beta = 1)", cu.b, cu.lo, cu.hi, p1u, NaN, mu.r2, n))
end

data = CSV.read("data/clozapine.csv", DataFrame)
data = data[data.clozapine .<= mean(data.clozapine) + 3 * std(data.clozapine), :]
analyse(data, "main analysis")
sens = data[data.norclozapine .<= mean(data.norclozapine) + 3 * std(data.norclozapine), :]
analyse(sens, "sensitivity analysis: without extreme norclozapine")

CSV.write("reports/linear_stats.csv", results)
println("\nSaving: reports/linear_stats.csv")
