println("Importing packages")
using CSV
using DataFrames
using MLJ
using MLJDecisionTreeInterface
using MLJLinearModels
using Random
using StatsBase
using JLD2
using Plots

# load training data
if isfile("data/clozapine_test.csv")
    println("Loading: clozapine_test.csv")
    test_raw_data = CSV.read("data/clozapine_test.csv", header=true, DataFrame)
else
    error("File data/clozapine_test.csv cannot be opened!")
    exit(-1)
end

# load models
if isfile("models/clozapine_classifier_model.jlso")
    println("Loading: clozapine_classifier_model.jlso")
    model_rfc = machine("models/clozapine_classifier_model.jlso")
else
    error("File models/clozapine_classifier_model.jlso cannot be opened!")
    exit(-1)
end
if isfile("models/clozapine_regressor_model.jlso")
    println("Loading: clozapine_regressor_model.jlso")
    clo_model_rfr = machine("models/clozapine_regressor_model.jlso")
else
    error("File models/clozapine_regressor_model.jlso cannot be opened!")
    exit(-1)
end
if isfile("models/norclozapine_regressor_model.jlso")
    println("Loading: norclozapine_regressor_model.jlso")
    nclo_model_rfr = machine("models/norclozapine_regressor_model.jlso")
else
    error("File models/norclozapine_regressor_model.jlso cannot be opened!")
    exit(-1)
end
if isfile("models/scaler.jld")
    println("Loading: scaler.jld")
    scaler = JLD2.load_object("models/scaler.jld")
else
    error("File models/scaler.jld cannot be opened!")
    exit(-1)
end

# preprocess
y1 = test_raw_data[:, 1]
y2 = repeat(["norm"], length(y1))
y2[y1 .> 550] .= "high"
y3 = test_raw_data[:, 2]
x = Matrix(test_raw_data[:, 3:end])

# standardize
println("Processing: standardize")
data = x[:, 2:5]
# scaler = StatsBase.fit(ZScoreTransform, data, dims=1)
# data = StatsBase.transform(scaler, data)
# or
m = scaler.mean
s = scaler.scale
for idx in 1:size(data, 1)
    data[idx, :] = (data[idx, :] .- m) ./ s
end
data[isnan.(data)] .= 0
x_gender = Bool.(x[:, 1])
x_cont = data
x_rest = x[:, 6:end]
x1 = DataFrame(:male=>x_gender)
x2 = DataFrame(x_cont, ["age", "dose", "bmi", "crp"])
x3 = DataFrame(x_rest, ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
x = hcat(x1, x2, x3)
x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Count, :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)
y2 = DataFrame(group=y2)
y2 = coerce(y2.group, OrderedFactor{2})

println("Number of entries: $(size(y1, 1))")
println()
println("Calculating predictions")
println("-----------------------")
println()
println("Regressor:")
Random.seed!(123)
yhat1 = MLJ.predict(clo_model_rfr, x)
yhat1 = round.(yhat1, digits=1)
Random.seed!(123)
yhat3 = MLJ.predict(nclo_model_rfr, x)
yhat3 = round.(yhat3, digits=1)
rmse_clo = zeros(length(yhat1))
rmse_nclo = zeros(length(yhat3))
for idx in eachindex(yhat1)
    rmse_clo[idx] = round.(sqrt((yhat1[idx] - y1[idx])^2), digits=2)
    rmse_nclo[idx] = round.(sqrt((yhat3[idx] - y3[idx])^2), digits=2)
    println("Subject ID: $idx \t CLO level: $(y1[idx]) \t prediction: $(yhat1[idx]) \t RMSE: $(rmse_clo[idx])")
    println("Subject ID: $idx \t NCLO level: $(y3[idx]) \t prediction: $(yhat3[idx]) \t RMSE: $(rmse_nclo[idx])")
    println()
end

Random.seed!(123)
yhat2 = MLJ.predict(model_rfc, x)
subj1 = 0
subj2 = 0
subj3 = 0
subj4 = 0
subj1_adj = 0
subj2_adj = 0
subj3_adj = 0
subj4_adj = 0
println("Classifier:")
yhat2_adj = repeat(["norm"], length(yhat2))
for idx in eachindex(yhat2)
    print("Subject ID: $idx \t group: $(uppercase(String(y2[idx]))) \t")
    p_high = broadcast(pdf, yhat2[idx], "high")
    p_norm = broadcast(pdf, yhat2[idx], "norm")
    if p_norm > p_high
        print("prediction: NORM, prob = $(round(p_norm, digits=2)) \t")
        if String(y2[idx]) == "norm"
            global subj1 += 1
        elseif String(y2[idx]) == "high"
            global subj2 += 1
        end
    else
        print("prediction: HIGH, prob = $(round(p_high, digits=2)) \t")
        if String(y2[idx]) == "high"
            global subj4 += 1
        elseif String(y2[idx]) == "norm"
            global subj3 += 1
        end
    end
    if yhat1[idx] > 550
        p_high += 0.2
        p_norm -= 0.2
    elseif yhat1[idx] <= 550
        p_norm += 0.2
        p_high -= 0.2
    end
    if yhat3[idx] > 400
        p_high += 0.1
        p_norm -= 0.1
    elseif yhat3[idx] <= 400
        p_norm += 0.1
        p_high -= 0.1
    end
    p_high > 1.0 && (p_high = 1.0)
    p_high < 0.0 && (p_high = 0.0)
    p_norm > 1.0 && (p_norm = 1.0)
    p_norm < 0.0 && (p_norm = 0.0)
    if p_norm > p_high
        println("adj. prediction: NORM, prob = $(round(p_norm, digits=2))")
        if String(y2[idx]) == "norm"
            global subj1_adj += 1
        elseif String(y2[idx]) == "high"
            global subj2_adj += 1
        end
    else
        println("adj. prediction: HIGH, prob = $(round(p_high, digits=2))")
        yhat2_adj[idx] = "high"
        if String(y2[idx]) == "high"
            global subj4_adj += 1
        elseif String(y2[idx]) == "norm"
            global subj3_adj += 1
        end
    end
end

yhat2_adj = coerce(yhat2_adj, OrderedFactor{2})

println()
println("Classifier model prediction accuracy:")
println("\tlog_loss: ", round(log_loss(yhat2, y2) |> mean, digits=4))
println("\tAUC: ", round(auc(yhat2, y2), digits=4))
println("\tmisclassification rate: ", round(misclassification_rate(mode.(yhat2), y2), digits=2))
println("\taccuracy: ", 1 - round(misclassification_rate(mode.(yhat2), y2), digits=2))
cm = confusion_matrix(mode.(yhat2), y2)
println("""
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │ $(lpad(cm[4], 4, " ")) │ $(lpad(cm[2], 4, " ")) │
prediction      ├──────┼──────┤
           high │ $(lpad(cm[3], 4, " ")) │ $(lpad(cm[1], 4, " ")) │
                └──────┴──────┘
         """)
println("\tsensitivity (TP): ", round(cm.mat[1, 1] / sum(cm.mat[:, 1]), digits=2))
println("\tspecificity (TP): ", round(cm.mat[2, 2] / sum(cm.mat[:, 2]), digits=2))
println()
println("Adjusted classifier:")
println("\tmisclassification rate: ", round(misclassification_rate(yhat2_adj, y2), digits=2))
println("\taccuracy: ", 1 - round(misclassification_rate(yhat2_adj, y2), digits=2))
cm = confusion_matrix(yhat2_adj, y2)
println("""
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │ $(lpad(cm[4], 4, " ")) │ $(lpad(cm[2], 4, " ")) │
prediction      ├──────┼──────┤
           high │ $(lpad(cm[3], 4, " ")) │ $(lpad(cm[1], 4, " ")) │
                └──────┴──────┘
         """)
println("\tsensitivity (TP): ", round(cm.mat[1, 1] / sum(cm.mat[:, 1]), digits=2))
println("\tspecificity (TP): ", round(cm.mat[2, 2] / sum(cm.mat[:, 2]), digits=2))
println()

println("Regressor model prediction accuracy:")
println("CLOZAPINE:")
m = RSquared()
println("\tR²:\t", round(m(yhat1, y1), digits=4))
m = RootMeanSquaredError()
println("\tRMSE:\t", round(m(yhat1, y1), digits=4))
println("NORCLOZAPINE:")
m = RSquared()
println("\tR²:\t", round(m(yhat3, y3), digits=4))
m = RootMeanSquaredError()
println("\tRMSE:\t", round(m(yhat3, y3), digits=4))
println()
print("Regressor execution time and memory use:\t")
@time yhat = MLJ.predict(clo_model_rfr, x)
print("Classifier execution time and memory use:\t")
@time yhat2 = MLJ.predict(model_rfc, x)
println()

p1 = Plots.plot(y1, label="data", ylims=(0, 2000), xlabel="patients", ylabel="clozapine [ng/mL]", )
p1 = Plots.plot!(yhat1, label="prediction", line=:dot, lw=2)
p2 = Plots.plot(y3, label="data", ylims=(0, 1000), xlabel="patients", ylabel="norclozapine [ng/mL]", )
p2 = Plots.plot!(yhat3, label="prediction", line=:dot, lw=2)
p = Plots.plot(p1, p2, layout=(2, 1))
savefig(p, "rr_test_accuracy.png")

println("Analysis completed.")
