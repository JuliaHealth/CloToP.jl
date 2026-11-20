# Out-of-fold prediction errors of the three deployed model types (manuscript Figure 1).
#
# Usage (after src/compare.jl):
#   julia src/plot_cv_errors.jl [sensitivity]
#
# Input:  reports/compare_predictions.csv (or reports/compare_sensitivity_predictions.csv)
# Output: reports/fig_cv_errors.png (or reports/fig_cv_errors_sensitivity.png)
#
# For each patient, the error (observed − predicted) is averaged over the repeats of the
# cross-validation; patients are sorted by increasing observed concentration. Rows: CLO, NCLO;
# columns: neural network and linear regression (total dose + morning/midday doses) and the
# quadratic model of the total dose (reference_quadratic in src/compare.jl).

using CSV
using DataFrames
using Plots
using Statistics

const SENS = "sensitivity" in ARGS
const IN = SENS ? "reports/compare_sensitivity_predictions.csv" : "reports/compare_predictions.csv"
const OUT = SENS ? "reports/fig_cv_errors_sensitivity.png" : "reports/fig_cv_errors.png"

const MODELS = [("totaldose_morn_midday", "Neural network", :black),
                ("linear_totaldose_morn_midday", "Linear regression", :royalblue),
                ("reference_quadratic", "Quadratic model", :darkorange)]
const TARGETS = [("clozapine", "CLO"), ("norclozapine", "NCLO")]

pred = CSV.read(IN, DataFrame)

panels = Any[]
letters = "ABCDEF"
k = 0
for (target, tlabel) in TARGETS
    errs = Dict{String,Vector{Float64}}()
    local obs
    for (model, _, _) in MODELS
        d = combine(groupby(pred[(pred.target .== target) .& (pred.model .== model), :], :patient),
                    :observed => first => :observed, :predicted => mean => :predicted)
        sort!(d, :observed)
        obs = d.observed
        errs[model] = d.observed .- d.predicted
    end
    lim = maximum(maximum(abs, e) for e in values(errs)) * 1.05
    for (model, mlabel, color) in MODELS
        global k += 1
        e = errs[model]
        p = plot(e, seriestype = :sticks, color = color, linewidth = 1,
                 xlabel = "Patients (sorted by increasing $tlabel)",
                 ylabel = k in (1, 4) ? "Error [ng/mL]" : "",
                 title = "$(letters[k])  $tlabel: $mlabel",
                 titlelocation = :left, titlefontsize = 10, guidefontsize = 9, tickfontsize = 8,
                 legend = false, ylims = (-lim, lim), xlims = (0, length(e) + 1))
        hline!(p, [0], color = :gray, linewidth = 0.5)
        push!(panels, p)
    end
end

fig = plot(panels..., layout = (2, 3), size = (1500, 800), dpi = 300,
           left_margin = 6Plots.mm, bottom_margin = 5Plots.mm)
savefig(fig, OUT)
println("Saving: $OUT")
