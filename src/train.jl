@info "Loading packages.."

using Pkg
# packages = ["CSV", "DataFrames", "JLD2", "MLJ", "MLJFlux", "NNlib", "Flux" "Plots", "StatsBase"]
# Pkg.add(packages)

using CSV
using DataFrames
using JLD2
using MLJ
using MLJDecisionTreeInterface
using MLJFlux
using NNlib
using Flux
using Random
using Plots
using StatsBase

m = Pkg.Operations.Context().env.manifest
println("                     CSV $(m[findfirst(v -> v.name == "CSV", m)].version)")
println("              DataFrames $(m[findfirst(v -> v.name == "DataFrames", m)].version)")
println("                    JLD2 $(m[findfirst(v -> v.name == "JLD2", m)].version)")
println("                     MLJ $(m[findfirst(v -> v.name == "MLJ", m)].version)")
println("MLJDecisionTreeInterface $(m[findfirst(v -> v.name == "MLJDecisionTreeInterface", m)].version)")
println("                 MLJFlux $(m[findfirst(v -> v.name == "MLJFlux", m)].version)")
println("                    Flux $(m[findfirst(v -> v.name == "Flux", m)].version)")
println("                   NNlib $(m[findfirst(v -> v.name == "MLJFlux", m)].version)")
println("                   Plots $(m[findfirst(v -> v.name == "Plots", m)].version)")
println("               StatsBase $(m[findfirst(v -> v.name == "StatsBase", m)].version)")
println()

@info "Loading data.."

# load training data
if isfile("data/clozapine_train.csv")
    println("Loading: clozapine_train.csv")
    train_raw_data = CSV.read("data/clozapine_train.csv", header=true, DataFrame)
    println()
else
    error("File data/clozapine_train.csv cannot be opened!")
    exit(-1)
end

# preprocess
@info "Preprocessing.."
y1 = train_raw_data[:, 1]
y2 = repeat(["norm"], length(y1))
y2[y1 .> 550] .= "high"
y3 = train_raw_data[:, 2]
x = Matrix(train_raw_data[:, 3:end])

println("Number of entries: $(length(y1))")

# add zero-dose data for each patient
# we need artificial zero-dose data to force the lowest concentration to equal 0, as the notion of intercept does not exist in trees
y1_z = zeros(length(y1))
y2_z = repeat(["norm"], length(y1))
x_z1 = deepcopy(x)
x_z1[:, 3] .= 0
y1 = vcat(y1, y1_z)
y2 = vcat(y2, y2_z)
y3 = vcat(y3, y1_z)
x = vcat(x, x_z1)

# standardize
println("Standardizing")
data = x[:, 2:end]
scaler = StatsBase.fit(ZScoreTransform, data[:, 1:4], dims=1)
data[:, 1:4] = StatsBase.transform(scaler, data[:, 1:4])
data[isnan.(data)] .= 0
x_gender = Bool.(x[:, 1])
x_cont = data[:, 1:4]
x_rest = round.(Int64, data[:, 5:end])

# create DataFrame
x1 = DataFrame(:male=>x_gender)
x2 = DataFrame(x_cont, ["age", "dose", "bmi", "crp"])
x3 = DataFrame(x_rest, ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
x = Float32.(hcat(x1, x2, x3))
# x = coerce(x, :male=>OrderedFactor{2}, :age=>Continuous, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Count, :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)
x = coerce(x, :male=>OrderedFactor{2}, :age=>Continuous, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Continuous, :inhibitors_3a4=>Continuous, :substrates_3a4=>Continuous, :inducers_1a2=>Continuous, :inhibitors_1a2=>Continuous, :substrates_1a2=>Continuous)
y2 = DataFrame(group=y2)
y2 = coerce(y2.group, OrderedFactor{2})
# scitype(y)
# levels(y)

println("Splitting: 70:30")
train_idx, test_idx = partition(eachindex(y2), 0.7, shuffle=true)
println()

@info "Creating classifier model"

nnc = @MLJ.load NeuralNetworkClassifier pkg=MLJFlux verbosity=0
model = nnc(builder = MLJFlux.Short(n_hidden = 32, 
                                    dropout = 0.01, 
                                    σ = NNlib.tanh_fast),
            finaliser = NNlib.softmax, 
            optimiser = Adam(0.01, (0.9, 0.999), IdDict{Any,Any}()),
            loss = Flux.Losses.crossentropy, 
            epochs = 10, 
            batch_size = 1, 
            lambda = 0.0, 
            alpha = 0.001)
mach = machine(model, x[train_idx, :], y2[train_idx], scitype_check_level=0)
MLJ.fit!(mach, force=true, verbosity=1)

println("Initial cross entropy: $(round(cross_entropy(MLJ.predict(mach, x[train_idx, :]), y2[train_idx]) |> mean, digits=4))")

ep = 10:10:10_000
training_loss = zeros(length(ep))
@info "Optimizing: epochs"
@Threads.threads for idx in 1:length(ep)
    # model.optimiser.eta = model.optimiser.eta + 0.01
    model.epochs = ep[idx]
    fit!(mach, verbosity=0)
    training_loss[idx] = cross_entropy(MLJ.predict(mach, x[train_idx, :]), y2[train_idx]) |> mean
end
_, idx = findmin(abs.(training_loss .- minimum(training_loss)))
model.epochs = ep[idx]
println("Epochs: $(model.epochs)")
println("Cross entropy: $(round(cross_entropy(MLJ.predict(mach, x[train_idx, :]), y2[train_idx]) |> mean, digits=4))")

batch_size = 1:10
training_loss = zeros(length(batch_size))
@info "Optimizing: batch size"
@Threads.threads for idx in 1:length(batch_size)
    model.batch_size = batch_size[idx]
    fit!(mach, verbosity=0)
    training_loss[idx] = cross_entropy(MLJ.predict(mach, x[train_idx, :]), y2[train_idx]) |> mean
end
_, idx = findmin(abs.(training_loss .- minimum(training_loss)))
model.batch_size = batch_size[idx]
println("Batch size: $(model.batch_size)")
println("Cross entropy: $(round(cross_entropy(MLJ.predict(mach, x[train_idx, :]), y2[train_idx]) |> mean, digits=4))")

eta = 0.001:0.001:0.1
training_loss = zeros(length(eta))
@info "Optimizing: eta"
@Threads.threads for idx in 1:length(eta)
    model.optimiser.eta = eta[idx]
    fit!(mach, verbosity=0)
    training_loss[idx] = cross_entropy(MLJ.predict(mach, x[train_idx, :]), y2[train_idx]) |> mean
end
_, idx = findmin(abs.(training_loss .- minimum(training_loss)))
model.optimiser.eta = eta[idx]
println("Eta: $(model.optimiser.eta)")
println("Cross entropy: $(round(cross_entropy(MLJ.predict(mach, x[train_idx, :]), y2[train_idx]) |> mean, digits=4))")

alpha = 0.001:0.001:1
training_loss = zeros(length(alpha))
@info "Optimizing: alpha"
@Threads.threads for idx in 1:length(alpha)
    # model.optimiser.eta = model.optimiser.eta + 0.01
    model.alpha = alpha[idx]
    fit!(mach, verbosity=0)
    training_loss[idx] = cross_entropy(MLJ.predict(mach, x[train_idx, :]), y2[train_idx]) |> mean
end
_, idx = findmin(abs.(training_loss .- minimum(training_loss)))
model.alpha = alpha[idx]
println("Alpha: $(model.alpha)")
println("Cross entropy: $(round(cross_entropy(MLJ.predict(mach, x[train_idx, :]), y2[train_idx]) |> mean, digits=4))")

println("Final cross entropy: $(round(cross_entropy(MLJ.predict(mach, x[train_idx, :]), y2[train_idx]) |> mean, digits=4))")

yhat = MLJ.predict(mach, x[train_idx, :])
println("Classifier training accuracy:")
println("\tlog_loss: ", round(log_loss(yhat, y2[train_idx]) |> mean, digits=2))
println("\tAUC: ", round(auc(yhat, y2[train_idx]), digits=2))
println("\tmisclassification rate: ", round(misclassification_rate(mode.(yhat), y2[train_idx]), digits=2))
println("\taccuracy: ", round(1 - misclassification_rate(mode.(yhat), y2[train_idx]), digits=2))
println("confusion matrix:")
cm = confusion_matrix(mode.(yhat), y2[train_idx])
println("\tsensitivity (TPR): ", round(cm.mat[1, 1] / sum(cm.mat[:, 1]), digits=2))
println("\tspecificity (TNR): ", round(cm.mat[2, 2] / sum(cm.mat[:, 2]), digits=2))
println("""
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │ $(lpad(cm.mat[4], 4, " ")) │ $(lpad(cm.mat[2], 4, " ")) │
prediction      ├──────┼──────┤
           high │ $(lpad(cm.mat[3], 4, " ")) │ $(lpad(cm.mat[1], 4, " ")) │
                └──────┴──────┘
         """)

yhat = MLJ.predict(mach, x[test_idx, :])
println("Classifier testing accuracy:")
println("\tlog_loss: ", round(log_loss(yhat, y2[test_idx]) |> mean, digits=2))
println("\tAUC: ", round(auc(yhat, y2[test_idx]), digits=2))
println("\tmisclassification rate: ", round(misclassification_rate(mode.(yhat), y2[test_idx]), digits=2))
println("\taccuracy: ", round(1 - misclassification_rate(mode.(yhat), y2[test_idx]), digits=2))
println("confusion matrix:")
cm = confusion_matrix(mode.(yhat), y2[test_idx])
println("\tsensitivity (TPR): ", round(cm.mat[1, 1] / sum(cm.mat[:, 1]), digits=2))
println("\tspecificity (TNR): ", round(cm.mat[2, 2] / sum(cm.mat[:, 2]), digits=2))
println("""
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │ $(lpad(cm.mat[4], 4, " ")) │ $(lpad(cm.mat[2], 4, " ")) │
prediction      ├──────┼──────┤
           high │ $(lpad(cm.mat[3], 4, " ")) │ $(lpad(cm.mat[1], 4, " ")) │
                └──────┴──────┘
         """)

@info "Creating regressor model"
println("Predicting: CLOZAPINE")

rfr = @MLJ.load RandomForestRegressor pkg=DecisionTree verbosity=0
model_clo = rfr(max_depth = -1, 
                min_samples_leaf = 1, 
                min_samples_split = 2, 
                min_purity_increase = 0.0, 
                n_subfeatures = -1, 
                n_trees = 750, 
                sampling_fraction = 1.0, 
                feature_importance = :impurity)
nnr = @MLJ.load NeuralNetworkRegressor pkg=MLJFlux verbosity=0
model_clo = nnr(builder = MLJFlux.Short(n_hidden=64, dropout=0.1, σ=hardtanh),
                optimiser = Adam(0.01, (0.9, 0.999), 1.0e-8, IdDict{Any, Any}()), 
                loss = Flux.Losses.mse, 
                epochs = 1000, 
                batch_size = 2, 
                lambda = 0.1, 
                alpha = 0.0) 
mach_clo = machine(model_clo, x[train_idx, :], y1[train_idx], scitype_check_level=0)
MLJ.fit!(mach_clo, force=true, verbosity=1)
yhat = MLJ.predict(mach_clo, x[train_idx, :])
println("Regressor training accuracy")
m = RSquared()
println("\tR²: ", round(m(yhat, y1[train_idx]), digits=2))
m = RootMeanSquaredError()
println("\tRMSE: ", round(m(yhat, y1[train_idx]), digits=2))

yhat = MLJ.predict(mach_clo, x[test_idx, :])
println("Regressor testing accuracy")
m = RSquared()
println("\tR²: ", round(m(yhat, y1[test_idx]), digits=2))
m = RootMeanSquaredError()
println("\tRMSE: ", round(m(yhat, y1[test_idx]), digits=2))

println("Predicting: NORCLOZAPINE")

model_nclo = rfr(max_depth = -1, 
                 min_samples_leaf = 1, 
                 min_samples_split = 2, 
                 min_purity_increase = 0.0, 
                 n_subfeatures = -1, 
                 n_trees = 250, 
                 sampling_fraction = 1.0, 
                 feature_importance = :impurity)
model_nclo = nnr(builder = MLJFlux.Short(n_hidden=64, dropout=0.1, σ=hardtanh),
                 optimiser = Adam(0.01, (0.9, 0.999), 1.0e-8, IdDict{Any, Any}()), 
                 loss = Flux.Losses.mse, 
                 epochs = 1000, 
                 batch_size = 2, 
                 lambda = 0.1, 
                 alpha = 0.0) 
mach_nclo = machine(model_nclo, x[train_idx, :], y3[train_idx], scitype_check_level=0)
MLJ.fit!(mach_nclo, force=true, verbosity=1)
yhat = MLJ.predict(mach_nclo, x[train_idx, :])
println("Regressor training accuracy")
m = RSquared()
println("\tR²: ", round(m(yhat, y3[train_idx]), digits=2))
m = RootMeanSquaredError()
println("\tRMSE: ", round(m(yhat, y3[train_idx]), digits=2))

yhat = MLJ.predict(mach_nclo, x[test_idx, :])
println("Regressor testing accuracy")
m = RSquared()
println("\tR²: ", round(m(yhat, y3[test_idx]), digits=2))
m = RootMeanSquaredError()
println("\tRMSE: ", round(m(yhat, y3[test_idx]), digits=2))
println()

# final training on the whole dataset
@info "Training final model.."
mach_clas = machine(model, x, y2, scitype_check_level=0)
MLJ.fit!(mach_clas, force=true, verbosity=1)
yhat = MLJ.predict(mach_clas, x)
println("Classifier accuracy:")
println("\tlog_loss: ", round(log_loss(yhat, y2) |> mean, digits=2))
println("\tAUC: ", round(auc(yhat, y2), digits=2))
println("\tmisclassification rate: ", round(misclassification_rate(mode.(yhat), y2), digits=2))
println("\taccuracy: ", round(1 - misclassification_rate(mode.(yhat), y2), digits=2))
println("confusion matrix:")
cm = confusion_matrix(mode.(yhat), y2)
println("\tsensitivity (TPR): ", round(cm.mat[1, 1] / sum(cm.mat[:, 1]), digits=2))
println("\tspecificity (TNR): ", round(cm.mat[2, 2] / sum(cm.mat[:, 2]), digits=2))
println("""
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │ $(lpad(cm.mat[4], 4, " ")) │ $(lpad(cm.mat[2], 4, " ")) │
prediction      ├──────┼──────┤
           high │ $(lpad(cm.mat[3], 4, " ")) │ $(lpad(cm.mat[1], 4, " ")) │
                └──────┴──────┘
         """)
println("Predicting: CLOZAPINE")
mach_clo = machine(model_clo, x, y1, scitype_check_level=0)
MLJ.fit!(mach_clo, force=true, verbosity=1)
yhat = MLJ.predict(mach_clo, x)
sorting_idx = sortperm(y1)
p1 = Plots.plot(y1[sorting_idx], label="data", ylims=(0, 2000), xlabel="patients", ylabel="clozapine [ng/mL]")
p1 = Plots.plot!(yhat[sorting_idx], label="prediction", line=:dot, lw=2)
println("Regressor accuracy")
m = RSquared()
println("\tR²: ", round(m(yhat, y1), digits=2))
m = RootMeanSquaredError()
println("\tRMSE: ", round(m(yhat, y1), digits=2))
println("Predicting: NORCLOZAPINE")
mach_nclo = machine(model_nclo, x, y3, scitype_check_level=0)
MLJ.fit!(mach_nclo, force=true, verbosity=1)
yhat = MLJ.predict(mach_nclo, x)
sorting_idx = sortperm(y3)
p2 = Plots.plot(y3[sorting_idx], label="data", ylims=(0, 2000), xlabel="patients", ylabel="norclozapine [ng/mL]")
p2 = Plots.plot!(yhat[sorting_idx], label="prediction", line=:dot, lw=2)
println("Regressor accuracy")
m = RSquared()
println("\tR²: ", round(m(yhat, y3), digits=2))
m = RootMeanSquaredError()
println("\tRMSE: ", round(m(yhat, y3), digits=2))
println()

@info "Saving models.."

println("Saving: clozapine_classifier_model.jlso")
MLJ.save("models/clozapine_classifier_model.jlso", mach_clas)
println("Saving: clozapine_regressor_model.jlso")
MLJ.save("models/clozapine_regressor_model.jlso", mach_clo)
println("Saving: norclozapine_regressor_model.jlso")
MLJ.save("models/norclozapine_regressor_model.jlso", mach_nclo)
println("Saving: scaler.jld")
JLD2.save_object("models/scaler.jld", scaler)
println()

p = Plots.plot(p1, p2, layout=(2, 1))
savefig(p, "images/rr_training_accuracy.png")
