# Dosing schedule figure of the manuscript (Figure 3).
#
# Usage (after src/pk_schedule.jl and src/schedule_effect.jl):
#   julia src/plot_schedule.jl
#
# A: steady-state CLO concentration-time curves of a typical patient (AGNP 2026 parameters),
#    300 mg/day given in four schedules (reports/pk_curves.csv).
# B: morning trough CLO concentration vs. MM for 300 mg/day: one-compartment model with the
#    AGNP 2026 and Li et al. 2012 typical parameters, and the observed concentrations of the 122
#    patients normalized to 300 mg/day (CLO × 300 / total daily dose).
# C: predicted CLO concentration relative to evening-only dosing when a fraction of the
#    patient's own total dose is moved to the morning: one-compartment model vs. the deployed
#    neural network (mean and interquartile range over patients) and linear regression
#    (reports/schedule_effect.csv).
#
# Output: reports/fig_schedule.png

using CSV
using DataFrames
using Plots
using Statistics

const MM_MIDDAY_WEIGHT = 0.5861325

curves = CSV.read("reports/pk_curves.csv", DataFrame)
effect = CSV.read("reports/schedule_effect.csv", DataFrame)
data = CSV.read("data/clozapine.csv", DataFrame)
data = data[data.clozapine .<= mean(data.clozapine) + 3 * std(data.clozapine), :]

fs = (titlefontsize = 10, guidefontsize = 9, tickfontsize = 8, legendfontsize = 7, titlelocation = :left)

# A
pa = plot(; xlabel = "Clock time [h]", ylabel = "CLO concentration [ng/mL]", title = "A  Typical patient, 300 mg/day",
          xticks = 0:4:24, xlims = (0, 24), legend = :topleft, fs...)
styles = Dict("evening 300 mg" => (:black, :solid), "morning 300 mg" => (:darkorange, :dash),
              "morning 150 + evening 150 mg" => (:royalblue, :dashdot), "100 mg three times daily" => (:seagreen, :dot))
for s in ("evening 300 mg", "morning 150 + evening 150 mg", "100 mg three times daily", "morning 300 mg")
    c = curves[(curves.model .== "AGNP 2026") .& (curves.schedule .== s), :]
    plot!(pa, c.time, c.concentration, label = s, color = styles[s][1], linestyle = styles[s][2], linewidth = 1.5)
end
vline!(pa, [8], color = :gray, linestyle = :dash, linewidth = 0.8, label = "sampling (before the morning dose)")
ylims!(pa, (0, ylims(pa)[2] * 1.45))

# B
td = data.morning_dose .+ data.midday_dose .+ data.evening_dose
mm = (data.morning_dose .+ MM_MIDDAY_WEIGHT .* data.midday_dose) ./ td
pb = scatter(mm, data.clozapine .* 300 ./ td, markersize = 3, color = :gray70, markerstrokewidth = 0,
             label = "patients (normalized to 300 mg/day)", xlabel = "MM", ylabel = "Trough CLO concentration [ng/mL]",
             title = "B  Trough vs. MM, 300 mg/day", xlims = (-0.02, 1.02), legend = :topright; fs...)
for (name, e, m, col, ls) in (("PK model, AGNP 2026", 593.2, 296.4, :black, :solid), ("PK model, Li 2012", 566.2, 342.6, :firebrick, :dash))
    plot!(pb, [0, 1], [e, m], label = name, color = col, linestyle = ls, linewidth = 1.5)
end

# C
pc = plot(; xlabel = "Fraction of the total dose moved to the morning", ylabel = "Relative to evening-only dosing",
          title = "C  Schedule effect at a fixed total dose", legend = :bottomleft, fs...)
for (model, col, ls) in (("neural network", :black, :solid), ("linear regression", :royalblue, :dash),
                         ("PK AGNP 2026", :gray40, :dot), ("PK Li 2012", :firebrick, :dashdot))
    e = sort(effect[(effect.target .== "clozapine") .& (effect.model .== model), :], :morning_fraction)
    if model == "neural network"
        plot!(pc, e.morning_fraction, e.q25, fillrange = e.q75, fillalpha = 0.15, color = col, linewidth = 0,
              label = "neural network, IQR")
    end
    label = startswith(model, "PK") ? replace(model, "PK " => "PK model, ") : model * ", mean"
    plot!(pc, e.morning_fraction, e.mean_ratio, color = col, linestyle = ls, linewidth = 1.5, label = label)
end
hline!(pc, [1], color = :gray, linewidth = 0.5, label = "")

fig = plot(pa, pb, pc, layout = (1, 3), size = (1500, 480), dpi = 300,
           left_margin = 6Plots.mm, bottom_margin = 6Plots.mm)
savefig(fig, "reports/fig_schedule.png")
println("Saving: reports/fig_schedule.png")
