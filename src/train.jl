@info "Loading packages.."

using Pkg
# packages = ["CSV", "DataFrames", "JLD2", "MLJ", "MLJFlux", "NNlib", "Flux" "Plots", "StatsBase"]
# Pkg.add(packages)

using CSV
using DataFrames
using JLD2
using MLJ
# using MLJDecisionTreeInterface
# using MLJXGBoostInterface
using MLJFlux
using NNlib
using Flux
using Random
using Plots
using StatsBase

Random.seed!(rand(1:1000))

m = Pkg.Operations.Context().env.manifest
println("       CSV $(m[findfirst(v -> v.name == "CSV", m)].version)")
println("DataFrames $(m[findfirst(v -> v.name == "DataFrames", m)].version)")
println("      JLD2 $(m[findfirst(v -> v.name == "JLD2", m)].version)")
println("       MLJ $(m[findfirst(v -> v.name == "MLJ", m)].version)")
println("   MLJFlux $(m[findfirst(v -> v.name == "MLJFlux", m)].version)")
println("      Flux $(m[findfirst(v -> v.name == "Flux", m)].version)")
println("     NNlib $(m[findfirst(v -> v.name == "MLJFlux", m)].version)")
println("     Plots $(m[findfirst(v -> v.name == "Plots", m)].version)")
println(" StatsBase $(m[findfirst(v -> v.name == "StatsBase", m)].version)")
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
# y1_z = zeros(length(y1))
# y2_z = repeat(["norm"], length(y1))
# x_z1 = deepcopy(x)
# x_z1[:, 3] .= 0
# y1 = vcat(y1, y1_z)
# y2 = vcat(y2, y2_z)
# y3 = vcat(y3, y1_z)
# x = vcat(x, x_z1)

# standardize
println("Standardizing")
data = x[:, 2:end]
scaler = StatsBase.fit(ZScoreTransform, data[:, 1:4], dims=1)
# data[:, 1:4] = StatsBase.transform(scaler, data[:, 1:4])
data[isnan.(data)] .= 0
x_gender = Bool.(x[:, 1])
x_cont = data[:, 1:4]
x_rest = round.(Int64, data[:, 5:end])

# create DataFrame
x1 = DataFrame(:male=>x_gender)
x2 = DataFrame(x_cont, ["age", "dose", "bmi", "crp"])
x3 = DataFrame(x_rest, ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
x = Float32.(hcat(x1, x2, x3))
x = coerce(x, :male=>OrderedFactor{2}, :age=>Continuous, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Count, :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)
y2 = DataFrame(group=y2)
y2 = coerce(y2.group, OrderedFactor{2})
# scitype(y)
# levels(y)

println("Splitting (70:30)")
train_idx, test_idx = partition(eachindex(y2), 0.7, shuffle=true)
# train_idx, test_idx = partition(eachindex(y), 0.8, rng=123) # 70:30 split
println()

@info "Creating classifier model"
# rfc = @MLJ.load RandomForestClassifier pkg=DecisionTree verbosity=0

#=
info(XGBoostClassifier)
n_trees_range = range(Int, :n_trees, lower=1, upper=1000)
max_depth_range = range(Int, :max_depth, lower=1, upper=100)
min_samples_leaf_range = range(Int, :min_samples_leaf, lower=1, upper=100)
min_samples_split_range = range(Int, :min_samples_split, lower=1, upper=100)
min_purity_increase_range = range(Float64, :min_purity_increase, lower=0.1, upper=1.0)
sampling_fraction_range = range(Float64, :sampling_fraction, lower=0.0, upper=1.0)
p = [n_trees_range, max_depth_range, min_samples_leaf_range, min_samples_split_range, min_purity_increase_range, sampling_fraction_range]
m = [log_loss, auc, accuracy, f1score, misclassification_rate, cross_entropy]
# Evaluating over 100_000 metamodels
self_tuning_rfc1 = TunedModel(model=rfc(),
                              resampling=CV(nfolds=5),
                              tuning=Grid(resolution=5),
                              range=p,
                              measure=m)
m_self_tuning_rfc1 = machine(self_tuning_rfc1, x[train_idx, :], y2[train_idx])
MLJ.fit!(m_self_tuning_rfc1)
yhat = MLJ.predict(m_self_tuning_rfc1, x[test_idx, :])
cm = confusion_matrix(mode.(yhat), y2[test_idx])
fitted_params(m_self_tuning_rfc1).best_model

MLJ.fit!(m_self_tuning_rfc1)
report(m_self_tuning_rfc1).best_history_entry

evaluate(model,
         x, y,
         resampling=CV(nfolds=10),
         measures=[log_loss, accuracy, f1score, misclassification_rate, cross_entropy])
report(m_self_tuning_rfc).best_history_entry
=#

# model = rfc(max_depth = -1, 
#             min_samples_leaf = 1, 
#             min_samples_split = 2, 
#             min_purity_increase = 0, 
#             n_subfeatures = -1, 
#             n_trees = 750, 
#             sampling_fraction = 1.0, 
#             feature_importance = :impurity)
# xgbc = @MLJ.load XGBoostClassifier pkg=XGBoost verbosity=0
# model = xgbc(test = 1, 
#              num_round = 100, 
#              booster = "gbtree", 
#              disable_default_eval_metric = 0, 
#              eta = 0.2, 
#              num_parallel_tree = 1, 
#              gamma = 0.0, 
#              max_depth = 128, 
#              min_child_weight = 1.0, 
#              max_delta_step = 0.0, 
#              subsample = 1.0, 
#              colsample_bytree = 1.0, 
#              colsample_bylevel = 1.0, 
#              colsample_bynode = 1.0, 
#              lambda = 1.0, 
#              alpha = 0.0, 
#              tree_method = "auto", 
#              sketch_eps = 0.03, 
#              scale_pos_weight = 10.0, 
#              updater = nothing, 
#              refresh_leaf = 1, 
#              process_type = "default", 
#              grow_policy = "depthwise", 
#              max_leaves = 0, 
#              max_bin = 512, 
#              predictor = "cpu_predictor", 
#              sample_type = "uniform", 
#              normalize_type = "tree", 
#              rate_drop = 0.0, 
#              one_drop = 0, 
#              skip_drop = 0.0, 
#              feature_selector = "cyclic", 
#              top_k = 0, 
#              tweedie_variance_power = 1.5, 
#              objective = "automatic", 
#              base_score = 0.1, 
#              watchlist = nothing, 
#              importance_type = "gain", 
#              seed = nothing, 
#              validate_parameters = false, 
#              eval_metric = String[])

nnc = @MLJ.load NeuralNetworkClassifier pkg=MLJFlux verbosity=0
model = nnc(builder = MLJFlux.Short(n_hidden = 100, 
                                    dropout = 0.1, 
                                    σ = NNlib.σ),
            finaliser = NNlib.softmax, 
            optimiser = Flux.Optimise.Adam(0.001, (0.9, 0.999), 1.0e-8, IdDict{Any, Any}()), 
            # acceleration=CUDALibs(), 
            loss = Flux.Losses.crossentropy, 
            epochs = 1000, 
            batch_size = 10, 
            lambda = 0.0, 
            alpha = 1.0)
mach = machine(model, x[train_idx, :], y2[train_idx], scitype_check_level=0)
MLJ.fit!(mach, force=true, verbosity=1)
yhat = MLJ.predict(mach, x[train_idx, :])

#=
dtc = @MLJ.load DecisionTreeClassifier pkg=DecisionTree verbosity=0
model = dtc(max_depth = -1, 
            min_samples_leaf = 1, 
            min_samples_split = 2, 
            min_purity_increase = 0.0, 
            n_subfeatures = 0, 
            post_prune = true,
            merge_purity_threshold = 1.0, 
            feature_importance = :impurity)

max_depth_range = range(Int, :max_depth, lower=1, upper=100)
min_samples_leaf_range = range(Int, :min_samples_leaf, lower=1, upper=100)
min_samples_split_range = range(Int, :min_samples_split, lower=1, upper=100)
min_purity_increase_range = range(Float64, :min_purity_increase, lower=0.1, upper=1.0)
n_subfeatures_range = range(Int, :n_subfeatures, lower=1, upper=11)
p = [min_samples_leaf_range, min_samples_split_range, min_purity_increase_range, n_subfeatures_range]
m = [log_loss, auc, accuracy, f1score, misclassification_rate, cross_entropy]

p = [n_trees_range, max_depth_range]
m = [log_loss]

# Evaluating over 100_000 metamodels
self_tuning_rfc1 = TunedModel(model=dtc(),
                              resampling=CV(nfolds=5),
                              tuning=Grid(resolution=5),
                              range=p,
                              measure=m)
m_self_tuning_rfc1 = machine(self_tuning_rfc1, x[train_idx, :], y2[train_idx])
MLJ.fit!(m_self_tuning_rfc1)
yhat = MLJ.predict(m_self_tuning_rfc1, x[test_idx, :])
cm = confusion_matrix(mode.(yhat), y2[test_idx])
fitted_params(m_self_tuning_rfc1).best_model

mach = machine(model, x[train_idx, :], y2[train_idx])
MLJ.fit!(mach, force=true, verbosity=2)
MLJ.fit!(mach, force=true, verbosity=0)
yhat = MLJ.predict(mach)
# n = names(x)
# for idx in eachindex(n)
#     println("Feature $(lpad(string(idx), 2, ' ')): $(n[idx])")
# end
=#

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

# mach_test = machine(model, x[test_idx, :], y2[test_idx], scitype_check_level=0)
# MLJ.fit!(mach_test, verbosity=0)
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
xbr = @MLJ.load XGBoostRegressor pkg=XGBoost verbosity=0

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
m = [root_mean_squared_error]
self_tuning_rfr1 = TunedModel(model=rfr(),
                              resampling=CV(nfolds=10),
                              tuning=Grid(resolution=10),
                              range=p,
                              measure=m)
m_self_tuning_rfr1 = machine(self_tuning_rfr1, x, y1)
MLJ.fit!(m_self_tuning_rfr1)
fitted_params(m_self_tuning_rfr1).best_model

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

evaluate(model_clo,
         x, y,
         resampling=CV(nfolds=10),
         measure=[rsq, root_mean_squared_error])
=#

# model_clo = rfr(max_depth = -1, 
#                 min_samples_leaf = 1, 
#                 min_samples_split = 2, 
#                 min_purity_increase = 0.0, 
#                 n_subfeatures = -1, 
#                 n_trees = 750, 
#                 sampling_fraction = 1.0, 
#                 feature_importance = :impurity)
#model_clo = xbr(test = 1, 
#                num_round = 10000, 
#                booster = "gbtree", 
#                disable_default_eval_metric = 0, 
#                eta = 0.01, 
#                num_parallel_tree = 1, 
#                gamma = 0.0, 
#                max_depth = 1, 
#                min_child_weight = 1.0, 
#                max_delta_step = 0.0, 
#                subsample = 1.0, 
#                colsample_bytree = 1.0, 
#                colsample_bylevel = 1.0, 
#                colsample_bynode = 1.0, 
#                lambda = 10.0, 
#                alpha = 0.1, 
#                tree_method = "auto", 
#                sketch_eps = 0.03, 
#                scale_pos_weight = 1.0, 
#                updater = nothing, 
#                refresh_leaf = 1, 
#                process_type = "default", 
#                grow_policy = "depthwise", 
#                max_leaves = 0, 
#                max_bin = 512, 
#                predictor = "cpu_predictor", 
#                sample_type = "uniform", 
#                normalize_type = "tree", 
#                rate_drop = 0.0, 
#                one_drop = 0, 
#                skip_drop = 0.0, 
#                feature_selector = "cyclic", 
#                top_k = 0, 
#                tweedie_variance_power = 1.5, 
#                objective = "reg:squarederror", 
#                base_score = 0.5, 
#                watchlist = nothing, 
#                importance_type = "gain", 
#                seed = nothing, 
#                validate_parameters = false, 
#                eval_metric = String[])
nnr = @MLJ.load NeuralNetworkRegressor pkg=MLJFlux verbosity=0
model_clo = nnr(builder =  MLJFlux.Short(n_hidden=200,
                                          dropout=0.1, 
                                          σ = NNlib.relu), 
                optimiser = Adam(0.001, (0.9, 0.999), 1.0e-8, IdDict{Any, Any}()), 
                loss = Flux.Losses.mse, 
                epochs = 1000, 
                batch_size = 10, 
                lambda = 0.0, 
                alpha = 0.2) 
mach_clo = machine(model_clo, x[train_idx, :], y1[train_idx], scitype_check_level=0)
MLJ.fit!(mach_clo, force=true, verbosity=1)
yhat = MLJ.predict(mach_clo, x[train_idx, :])
#yhat_reconstructed = round.(((yhat .* scaler_y.scale[1]) .+ scaler_y.mean[1]), digits=1)
#y_reconstructed = round.(((y .* scaler_y.scale[1]) .+ scaler_y.mean[1]), digits=1)
# regression parameters
# params = fitted_params(mach_clo)
# params.coefs # coefficient of the regression with names
# params.intercept # intercept
println("Regressor training accuracy")
m = RSquared()
println("\tR²: ", round(m(yhat, y1[train_idx]), digits=2))
m = RootMeanSquaredError()
println("\tRMSE: ", round(m(yhat, y1[train_idx]), digits=2))

# mach_clo_test = machine(model_clo, x[test_idx, :], y1[test_idx], scitype_check_level=0)
# MLJ.fit!(mach_clo_test, force=true, verbosity=0)
yhat = MLJ.predict(mach_clo, x[test_idx, :])
#yhat_reconstructed = round.(((yhat .* scaler_y.scale[1]) .+ scaler_y.mean[1]), digits=1)
#y_reconstructed = round.(((y .* scaler_y.scale[1]) .+ scaler_y.mean[1]), digits=1)
# regression parameters
# params = fitted_params(mach_clo)
# params.coefs # coefficient of the regression with names
# params.intercept # intercept
println("Regressor testing accuracy")
m = RSquared()
println("\tR²: ", round(m(yhat, y1[test_idx]), digits=2))
m = RootMeanSquaredError()
println("\tRMSE: ", round(m(yhat, y1[test_idx]), digits=2))

println("Predicting: NORCLOZAPINE")

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
p = [n_trees_range, max_depth_range, min_samples_leaf_range, min_samples_split_range, n_subfeatures_range, sampling_fraction_range]
p = [n_trees_range]
m = [root_mean_squared_error]
self_tuning_rfr1 = TunedModel(model=rfr(feature_importance=:split),
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
m_self_tuning_rfr1 = machine(self_tuning_rfr1, x, y3)
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

# model_nclo = rfr(max_depth = -1, 
#                  min_samples_leaf = 1, 
#                  min_samples_split = 2, 
#                  min_purity_increase = 0.0, 
#                  n_subfeatures = -1, 
#                  n_trees = 250, 
#                  sampling_fraction = 1.0, 
#                  feature_importance = :impurity)
model_nclo = nnr(builder =  MLJFlux.Short(n_hidden=100,
                                          dropout=0.1, 
                                          σ = NNlib.relu), 
                 optimiser = Adam(0.001, (0.9, 0.999), 1.0e-8, IdDict{Any, Any}()), 
                 loss = Flux.Losses.mse, 
                 epochs = 10000, 
                 batch_size = 10, 
                 lambda = 0.0, 
                 alpha = 0.2) 
mach_nclo = machine(model_nclo, x[train_idx, :], y3[train_idx], scitype_check_level=0)
MLJ.fit!(mach_nclo, force=true, verbosity=1)
yhat = MLJ.predict(mach_nclo, x[train_idx, :])
# mach_nclo = machine(model_nclo, x, y, scitype_check_level=0)
# MLJ.fit!(mach_nclo, force=true, verbosity=0)
# yhat = MLJ.predict(mach_nclo, x)
# yhat_reconstructed = round.(((yhat .* scaler.scale[1]) .+ scaler.mean[1]), digits=1)
# y_reconstructed = round.(((y .* scaler.scale[1]) .+ scaler.mean[1]), digits=1)
# regression parametersmach_nclo)
# params.coefs # coefficient of the regression with names
# params.intercept # intercept
println("Regressor training accuracy")
m = RSquared()
println("\tR²: ", round(m(yhat, y3[train_idx]), digits=2))
m = RootMeanSquaredError()
println("\tRMSE: ", round(m(yhat, y3[train_idx]), digits=2))

# mach_nclo_test = machine(model_nclo, x[test_idx, :], y3[test_idx], scitype_check_level=0)
# MLJ.fit!(mach_nclo_test, force=true, verbosity=0)
yhat = MLJ.predict(mach_nclo, x[test_idx, :])
# regression parametersmach_nclo)
# params.coefs # coefficient of the regression with names
# params.intercept # intercept
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
