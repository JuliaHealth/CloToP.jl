println("Importing packages")
using CSV
using DataFrames
using MLJ
using MLJDecisionTreeInterface
using Random
using StatsBase
using JLD2
using Plots

# load training data
if isfile("data/clozapine_train.csv")
    println("Loading: clozapine_train.csv")
    train_raw_data = CSV.read("data/clozapine_train.csv", header=true, DataFrame)
else
    error("File data/clozapine_train.csv cannot be opened!")
    exit(-1)
end

# preprocess
y1 = train_raw_data[:, 1]
y2 = repeat(["norm"], length(y1))
y2[y1 .> 550] .= "high"
x = Matrix(train_raw_data[:, 3:end])

# add zero-dose data
y1_z = zeros(length(y1))
y2_z = repeat(["norm"], length(y1))
x_z = deepcopy(x)
x_z[:, 3] .= 0
y1 = vcat(y1, y1_z)
y2 = vcat(y2, y2_z)
x = vcat(x, x_z)

println("Number of entries: $(length(y1))")

# standardize
println("Processing: standardize")
data = x[:, 2:end]
scaler = StatsBase.fit(ZScoreTransform, data, dims=1)
# scaler = StatsBase.fit(UnitRangeTransform, data, dims=1)
data = StatsBase.transform(scaler, data)
data[isnan.(data)] .= 0
x_gender = Bool.(x[:, 1])
x_cont = data[:, 1:4]
x_rest = data[:, 5:end]
x1 = DataFrame(:male=>x_gender)
x2 = DataFrame(x_cont, ["age", "dose", "bmi", "crp"])
x3 = DataFrame(x_rest, ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
x = hcat(x1, x2, x3)
x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Continuous, :inhibitors_3a4=>Continuous, :substrates_3a4=>Continuous, :inducers_1a2=>Continuous, :inhibitors_1a2=>Continuous, :substrates_1a2=>Continuous)
y = DataFrame(group=y2)
y = coerce(y.group, OrderedFactor{2})
# scitype(y)
# levels(y)

println()

println("Creating classifier model: RandomForestClassifier")
rfc = @MLJ.load RandomForestClassifier pkg=DecisionTree verbosity=0
train_idx, test_idx = partition(eachindex(y), 0.7, rng=123) # 70:30 split

#=
info(RandomForestClassifier)

n_trees_range = range(Int, :n_trees, lower=1, upper=1000)
max_depth_range = range(Int, :max_depth, lower=1, upper=100)
min_samples_leaf_range = range(Int, :min_samples_leaf, lower=1, upper=100)
min_samples_split_range = range(Int, :min_samples_split, lower=1, upper=100)
min_purity_increase_range = range(Float64, :min_purity_increase, lower=0.1, upper=1.0)
sampling_fraction_range = range(Float64, :sampling_fraction, lower=0.1, upper=1.0)
p = [n_trees_range, max_depth_range, min_samples_leaf_range, min_samples_split_range, min_purity_increase_range, sampling_fraction_range]
m = [log_loss, auc, accuracy, f1score, misclassification_rate, cross_entropy]
m = [log_loss]

# split
# Evaluating over 78125 metamodels
Random.seed!(123)
self_tuning_rfc1 = TunedModel(model=rfc(feature_importance=:split),
                              resampling=CV(nfolds=5),
                              tuning=Grid(resolution=5),
                              range=p,
                              measure=m)
m_self_tuning_rfc1 = machine(self_tuning_rfc1, x, y)
MLJ.fit!(m_self_tuning_rfc1, rows=train_idx)
MLJ.fit!(m_self_tuning_rfc1, rows=test_idx)
MLJ.fit!(m_self_tuning_rfc1)
fitted_params(m_self_tuning_rfc1).best_model
report(m_self_tuning_rfc1).best_history_entry

# impurity
Random.seed!(123)
self_tuning_rfc2 = TunedModel(model=rfc(feature_importance=:impurity),
                              resampling=CV(nfolds=5),
                              tuning=Grid(resolution=5),
                              range=p,
                              measure=m)
m_self_tuning_rfc2 = machine(self_tuning_rfc2, x, y)
MLJ.fit!(m_self_tuning_rfc2)
fitted_params(m_self_tuning_rfc2).best_model
report(m_self_tuning_rfc2).best_history_entry

model = fitted_params(m_self_tuning_rfc1).best_model
model = fitted_params(m_self_tuning_rfc2).best_model

report(m_self_tuning_rfc).best_history_entry
=#
Random.seed!(123)
model = rfc(max_depth = 26, 
            min_samples_leaf = 1, 
            min_samples_split = 2, 
            min_purity_increase = 0.0, 
            n_subfeatures = -1, 
            n_trees = 750, 
            sampling_fraction = 1.0, 
            feature_importance = :split)
#=
evaluate(model,
         x, y,
         resampling=CV(nfolds=10),
         measures=[log_loss, accuracy, f1score, misclassification_rate, cross_entropy])
=#
mach = machine(model, x, y, scitype_check_level=0)
MLJ.fit!(mach, force=true, verbosity=0)
yhat = MLJ.predict(mach, x)

println("RandomForestClassifier accuracy report:")
println("\tlog_loss: ", round(log_loss(yhat, y) |> mean, digits=4))
println("\tAUC: ", round(auc(yhat, y), digits=4))
println("\tmisclassification rate: ", round(misclassification_rate(mode.(yhat), y), digits=2))
println("\taccuracy: ", 1 - round(misclassification_rate(mode.(yhat), y), digits=2))
println("confusion matrix:")
cm = confusion_matrix(mode.(yhat), y)
println("\tsensitivity (TP): ", round(cm.mat[1, 1] / sum(cm.mat[:, 1]), digits=2))
println("\tspecificity (TP): ", round(cm.mat[2, 2] / sum(cm.mat[:, 2]), digits=2))
println("""
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │ $(lpad(cm[4], 4, " ")) │ $(lpad(cm[2], 4, " ")) │
prediction      ├──────┼──────┤
           high │ $(lpad(cm[3], 4, " ")) │ $(lpad(cm[1], 4, " ")) │
                └──────┴──────┘
         """)

println("Creating regressor model: RandomForestRegressor")

println("Processing: CLOZAPINE")
y = train_raw_data[:, 1]
y_z = zeros(length(y))
y = vcat(y, y_z)

rfr = @MLJ.load RandomForestRegressor pkg=DecisionTree verbosity=0

#=
info(RandomForestRegressor)

n_trees_range = range(Int, :n_trees, lower=1, upper=1000)
max_depth_range = range(Int, :max_depth, lower=1, upper=100)
min_samples_leaf_range = range(Int, :min_samples_leaf, lower=1, upper=100)
min_samples_split_range = range(Int, :min_samples_split, lower=1, upper=100)
n_subfeatures_range = range(Int, :n_subfeatures, lower=1, upper=11)
min_purity_increase_range = range(Float64, :min_purity_increase, lower=0.1, upper=1.0, scale=:log)
sampling_fraction_range = range(Float64, :sampling_fraction, lower=0.1, upper=1.0, scale=:log)
feature_importance_range = range(rfr(), FeatureSelector(), :feature_importance, values = [[:split], [:impurity]])
iterator(feature_importance_range)
p = [n_trees_range, max_depth_range, min_samples_leaf_range, min_samples_split_range, n_subfeatures_range, min_purity_increase_range, sampling_fraction_range, feature_importance_range]
p = [n_trees_range]
p = [n_trees_range, max_depth_range, sampling_fraction_range]
m = [root_mean_squared_error, rms]
Random.seed!(123)
self_tuning_rfr1 = TunedModel(model=rfr(),
                              resampling=CV(nfolds=5),
                              tuning=Grid(resolution=5),
                              range=p,
                              measure=m)
m_self_tuning_rfr1 = machine(self_tuning_rfr1, x, y)
MLJ.fit!(m_self_tuning_rfr1)
fitted_params(m_self_tuning_rfr1).best_model

Random.seed!(123)
self_tuning_rfr2 = TunedModel(model=rfr(feature_importance=:impurity, sampling_fraction=1.0),
                              resampling=CV(nfolds=5),
                              tuning=Grid(resolution=5),
                              range=p,
                              measure=m)
m_self_tuning_rfr2 = machine(self_tuning_rfr2, x, y)
MLJ.fit!(m_self_tuning_rfr2)
fitted_params(m_self_tuning_rfr2).best_model

MLJ.fit!(m_self_tuning_rfr1, rows=train_idx)
MLJ.fit!(m_self_tuning_rfr1, rows=test_idx)
MLJ.fit!(m_self_tuning_rfr1)

report(m_self_tuning_rfr1).best_history_entry
report(m_self_tuning_rfr2).best_history_entry

model = fitted_params(m_self_tuning_rfr1).best_model
model = fitted_params(m_self_tuning_rfr2).best_model
=#

Random.seed!(123)
model_clo = rfr(max_depth = -1, 
                min_samples_leaf = 1, 
                min_samples_split = 2, 
                min_purity_increase = 0.0, 
                n_subfeatures = -1, 
                n_trees = 500, 
                sampling_fraction = 1.0, 
                feature_importance = :split)
#=
evaluate(model_clo,
         x, y,
         resampling=CV(nfolds=10),
         measure=[rsq, root_mean_squared_error])
=#
mach_clo = machine(model_clo, x, y, scitype_check_level=0)
MLJ.fit!(mach_clo, force=true, verbosity=0)
yhat = MLJ.predict(mach_clo, x)
#yhat_reconstructed = round.(((yhat .* scaler_y.scale[1]) .+ scaler_y.mean[1]), digits=1)
#y_reconstructed = round.(((y .* scaler_y.scale[1]) .+ scaler_y.mean[1]), digits=1)
sorting_idx = sortperm(y)
p1 = Plots.plot(y[sorting_idx], label="data", ylims=(0, 2000), xlabel="patients", ylabel="clozapine [ng/mL]", )
p1 = Plots.plot!(yhat[sorting_idx], label="prediction", line=:dot, lw=2)
# regression parameters
# params = fitted_params(mach_clo)
# params.coefs # coefficient of the regression with names
# params.intercept # intercept
println("RandomForestRegressor accuracy report:")
m = RSquared()
println("\tR²: ", round(m(yhat, y), digits=4))
m = RootMeanSquaredError()
println("\tRMSE: ", round(m(yhat, y), digits=4))
println()

println("Processing: NORCLOZAPINE")
y = train_raw_data[:, 2]
y_z = zeros(length(y))
y = vcat(y, y_z)

#=
info(RandomForestRegressor)
evaluate(model,
         x, y,
         resampling=CV(nfolds=10),
         measure=[rsq, root_mean_squared_error])
n_trees_range = range(Int, :n_trees, lower=1, upper=1000)
max_depth_range = range(Int, :max_depth, lower=1, upper=100)
min_samples_leaf_range = range(Int, :min_samples_leaf, lower=1, upper=100)
min_samples_split_range = range(Int, :min_samples_split, lower=1, upper=100)
n_subfeatures_range = range(Int, :n_subfeatures, lower=1, upper=11)
min_purity_increase_range = range(Float64, :min_purity_increase, lower=0.1, upper=1.0)
sampling_fraction_range = range(Float64, :sampling_fraction, lower=0.1, upper=1.0)
p = [n_trees_range, max_depth_range, min_samples_leaf_range, min_samples_split_range, n_subfeatures_range, min_purity_increase_range, sampling_fraction_range]
p = [n_trees_range]
m = [root_mean_squared_error]
self_tuning_rfr1 = TunedModel(model=rfr(max_depth = -1, 
                                        min_samples_leaf = 1, 
                                        min_samples_split = 2, 
                                        min_purity_increase = 0.0, 
                                        n_subfeatures = -1, 
                                        sampling_fraction = 0.999, 
                                        feature_importance=:split),
                              resampling=CV(nfolds=10),
                              tuning=Grid(resolution=100),
                              range=p,
                              measure=m)
self_tuning_rfr2 = TunedModel(model=rfr(max_depth = -1, 
                                        min_samples_leaf = 1, 
                                        min_samples_split = 2, 
                                        min_purity_increase = 0.0, 
                                        n_subfeatures = -1, 
                                        sampling_fraction = 0.999, 
                                        feature_importance=:impurity),
                              resampling=CV(nfolds=10),
                              tuning=Grid(resolution=100),
                              range=p,
                              measure=m)
m_self_tuning_rfr1 = machine(self_tuning_rfr1, x, y)
MLJ.fit!(m_self_tuning_rfr1)
fitted_params(m_self_tuning_rfr1).best_model

m_self_tuning_rfr2 = machine(self_tuning_rfr2, x, y)
MLJ.fit!(m_self_tuning_rfr2)
fitted_params(m_self_tuning_rfr2).best_model

MLJ.fit!(m_self_tuning_rfr, rows=train_idx)
MLJ.fit!(m_self_tuning_rfr, rows=test_idx)

report(m_self_tuning_rfr1).best_history_entry
report(m_self_tuning_rfr2).best_history_entry

model = fitted_params(m_self_tuning_rfr1).best_model
model = fitted_params(m_self_tuning_rfr2).best_model
=#

Random.seed!(123)
model_nclo = rfr(max_depth = -1, 
                 min_samples_leaf = 1, 
                 min_samples_split = 2, 
                 min_purity_increase = 0.0, 
                 n_subfeatures = -1, 
                 n_trees = 500, 
                 sampling_fraction = 1.0, 
                 feature_importance = :split)

mach_nclo = machine(model_nclo, x, y, scitype_check_level=0)
MLJ.fit!(mach_nclo, force=true, verbosity=0)
yhat = MLJ.predict(mach_nclo, x)
# yhat_reconstructed = round.(((yhat .* scaler.scale[1]) .+ scaler.mean[1]), digits=1)
# y_reconstructed = round.(((y .* scaler.scale[1]) .+ scaler.mean[1]), digits=1)
sorting_idx = sortperm(y)
p2 = Plots.plot(y[sorting_idx], label="data", ylims=(0, 1000), xlabel="patients", ylabel="norclozapine [ng/mL]", )
p2 = Plots.plot!(yhat[sorting_idx], label="prediction", line=:dot, lw=2)
# regression parametersmach_nclo)
# params.coefs # coefficient of the regression with names
# params.intercept # intercept
println("RandomForestRegressor accuracy report:")
m = RSquared()
println("\tR²: ", round(m(yhat, y), digits=4))
m = RootMeanSquaredError()
println("\tRMSE: ", round(m(yhat, y), digits=4))
println()
println("Saving: clozapine_classifier_model.jlso")
MLJ.save("models/clozapine_classifier_model.jlso", mach)
println("Saving: clozapine_regressor_model.jlso")
MLJ.save("models/clozapine_regressor_model.jlso", mach_clo)
println("Saving: norclozapine_regressor_model.jlso")
MLJ.save("models/norclozapine_regressor_model.jlso", mach_nclo)
println("Saving: scaler.jld")
JLD2.save_object("models/scaler.jld", scaler)
println()

p = Plots.plot(p1, p2, layout=(2, 1))
savefig(p, "rr_train_accuracy.png")

println("Training completed.")
