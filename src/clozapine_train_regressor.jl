println("Importing packages")
using CSV
using DataFrames
using MLJ
using MLJLinearModels
using MLJDecisionTreeInterface
using Random
using StatsBase
using Plots
using JLD2

Random.seed!(123)

# load training data
if isfile("clozapine_train.csv")
    println("Loading raw data: clozapine_train.csv")
    train_raw_data = CSV.read("clozapine_train.csv", header=true, DataFrame)
else
    error("File clozapine_train.csv cannot be opened!")
    exit(-1)
end
if isfile("scaler.jld")
    println("Loading: scaler.jld")
    scaler = JLD2.load_object("scaler.jld")
else
    error("File scaler.jld cannot be opened!")
    exit(-1)
end

function preprocess_clo(train_raw_data)
    y = train_raw_data[:, 1]
    x = Matrix(train_raw_data[:, 5:end])
    return x, y
end

function preprocess_nclo(train_raw_data)
    y = train_raw_data[:, 3]
    x = Matrix(train_raw_data[:, 5:end])
    return x, y
end

println("Processing: CLOZAPINE")

x, y = preprocess_clo(train_raw_data)

println("Number of entries: $(size(y, 1))")

# standardize
println("Processing: standardize")
# data = hcat(y, x[:, 2:end])
data = x[:, 2:7]
scaler = StatsBase.fit(ZScoreTransform, data, dims=1)
data = StatsBase.transform(scaler, data)
# m = scaler.mean
# s = scaler.scale
# or
# for idx in 1:size(data, 1)
#     data[idx, :] = (data[idx, :] .- m) ./ s
# end
# data[isnan.(data)] .= 0
# y = data[:, 1]
x_gender = Bool.(x[:, 1])
# x_cont = data[:, 2:7]
x_cont = data
# x_rest = data[:, 8:end]
x_rest = x[:, 8:end]
x1 = DataFrame(:male=>x_gender)
# x2 = DataFrame(x_cont, ["age", "dose", "bmi", "weight", "duration", "crp"])
x2 = DataFrame(x_cont[:, [1, 2, 3, 6]], ["age", "dose", "bmi", "crp"])
x3 = DataFrame(x_rest, ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
x = hcat(x1, x2, x3)
# x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :weight=>Continuous, :duration=>Continuous, :crp=>Continuous, :inducers_3a4=>Count,  :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)
x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Count, :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)
# y = coerce(y, Continuous)

function train(model, x, y, train_idx, test_idx)
    Random.seed!(123)
    mach = machine(model, x, y)
    MLJ.fit!(mach, rows=train_idx, force=true, verbosity=0)
    yhat = MLJ.predict(mach, x[test_idx, :])
    return yhat
end

train_idx, test_idx = partition(eachindex(y), 0.7, multi=true, rng=123) # 70:30 split

println("Creating regressor model: RandomForestRegressor")
# rr = @MLJ.load RidgeRegressor pkg=MLJLinearModels
# model = rr(lambda=0.001)
rfr = @MLJ.load RandomForestRegressor pkg=DecisionTree

#=
info(RandomForestRegressor)
evaluate(model,
         x, y,
         resampling=CV(nfolds=10),
         measure=[rsq, root_mean_squared_error])
n_trees_range = range(Int, :n_trees, lower=1, upper=10000)
max_depth_range = range(Int, :max_depth, lower=1, upper=100)
min_samples_leaf_range = range(Int, :min_samples_leaf, lower=1, upper=100)
min_samples_split_range = range(Int, :min_samples_split, lower=1, upper=100)
n_subfeatures_range = range(Int, :n_subfeatures, lower=1, upper=11)
min_purity_increase_range = range(Float64, :min_purity_increase, lower=0.1, upper=1.0)
sampling_fraction_range = range(Float64, :sampling_fraction, lower=0.1, upper=1.0)
params = [n_trees_range, max_depth_range, min_samples_leaf_range, min_samples_split_range, n_subfeatures_range, min_purity_increase_range, sampling_fraction_range]
params = [n_trees_range]
measures = [root_mean_squared_error]
self_tuning_rfr1 = TunedModel(model=rfr(),
                            resampling=CV(nfolds=5),
                            tuning=Grid(resolution=5),
                            range=params,
                            measure=measures)
self_tuning_rfr2 = TunedModel(model=rfr(),
                            resampling=CV(nfolds=5),
                            tuning=Grid(resolution=5),
                            range=params,
                            measure=measures)
m_self_tuning_rfr1 = machine(self_tuning_rfr1, x, y)
m_self_tuning_rfr2 = machine(self_tuning_rfr2, x, y)
MLJ.fit!(m_self_tuning_rfr1)
MLJ.fit!(m_self_tuning_rfr2)

MLJ.fit!(m_self_tuning_rfr, rows=train_idx)
MLJ.fit!(m_self_tuning_rfr, rows=test_idx)

fitted_params(m_self_tuning_rfr1).best_model
fitted_params(m_self_tuning_rfr2).best_model
report(m_self_tuning_rfr1).best_history_entry
report(m_self_tuning_rfr2).best_history_entry

model = fitted_params(m_self_tuning_rfr1).best_model
model = fitted_params(m_self_tuning_rfr2).best_model
=#

model = rfr(max_depth = -1, 
            min_samples_leaf = 1, 
            min_samples_split = 2, 
            min_purity_increase = 0.0, 
            n_subfeatures = 1, 
            n_trees = 75, 
            sampling_fraction = 0.9, 
            feature_importance = :split)
mach = machine(model, x, y, scitype_check_level=0)
MLJ.fit!(mach, force=true, verbosity=0)
yhat = MLJ.predict(mach, x)
#yhat_reconstructed = round.(((yhat .* scaler_y.scale[1]) .+ scaler_y.mean[1]), digits=1)
#y_reconstructed = round.(((y .* scaler_y.scale[1]) .+ scaler_y.mean[1]), digits=1)
p1 = Plots.plot(y, label="CLO: data")
p1 = Plots.plot!(yhat, label="CLO: prediction", line=:dot, lw=2)
# regression parameters
# params = fitted_params(mach)
# params.coefs # coefficient of the regression with names
# params.intercept # intercept
println("RandomForestRegressor accuracy report:")
m = RSquared()
println("\tR²: ", round(m(yhat, y), digits=4))
m = RootMeanSquaredError()
println("\tRMSE: ", round(m(yhat, y), digits=4))
println()
println("Saving model: clozapine_regressor_model.jlso")
MLJ.save("clozapine_regressor_model.jlso", model_final)
println()

println("Processing: NORCLOZAPINE")

x, y = preprocess_nclo(train_raw_data)

println("Number of entries: $(size(y, 1))")

# standardize
println("Processing: standardize")
# data = hcat(y, x[:, 2:end])
data = x[:, 2:7]
# m = scaler.mean
# s = scaler.scale
scaler = StatsBase.fit(ZScoreTransform, data, dims=1)
data = StatsBase.transform(scaler, data)
# or
# for idx in 1:size(data, 1)
#     data[idx, :] = (data[idx, :] .- m) ./ s
# end
# data[isnan.(data)] .= 0
# y = data[:, 1]
x_gender = Bool.(x[:, 1])
# x_cont = data[:, 2:7]
x_cont = data
# x_rest = data[:, 8:end]
x_rest = x[:, 8:end]
x1 = DataFrame(:male=>x_gender)
x2 = DataFrame(x_cont, ["age", "dose", "bmi", "weight", "duration", "crp"])
x2 = DataFrame(x_cont[:, [1, 2, 3, 6]], ["age", "dose", "bmi", "crp"])
x3 = DataFrame(x_rest, ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
x = hcat(x1, x2, x3)
# x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :weight=>Continuous, :duration=>Continuous, :crp=>Continuous, :inducers_3a4=>Count,  :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)
x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Count,  :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)
# y = coerce(y, Continuous)

# RidgeRegressor
println("Creating model: RandomForestRegressor")
# @MLJ.load RidgeRegressor pkg=MLJLinearModels
# model = RidgeRegressor(lambda = 0.001,
#                       fit_intercept=false,
#                       penalize_intercept=true,
#                       scale_penalty_with_samples=false, 
#                       solver=CG()
#                       )
rfr = @MLJ.load RandomForestRegressor pkg=DecisionTree
model = rfr(max_depth = -1, 
            min_samples_leaf = 1, 
            min_samples_split = 2, 
            min_purity_increase = 0.0, 
            n_subfeatures = -1, 
            n_trees = 890, 
            sampling_fraction = 0.7, 
            feature_importance = :impurity)
mach = machine(model, x, y, scitype_check_level=0)
mach = machine(model, x, y, scitype_check_level=0)
# fit
MLJ.fit!(mach)
yhat, model_final = train_final(model, x, y)
# yhat_reconstructed = round.(((yhat .* scaler.scale[1]) .+ scaler.mean[1]), digits=1)
# y_reconstructed = round.(((y .* scaler.scale[1]) .+ scaler.mean[1]), digits=1)
p2 = Plots.plot(y, label="NCLO: data")
p2 = Plots.plot!(yhat, label="NCLO: prediction", line=:dot, lw=2)
# regression parameters
# params = fitted_params(mach)
# params.coefs # coefficient of the regression with names
# params.intercept # intercept
println("RidgeRegressor accuracy report:")
m = RSquared()
println("\tR²: ", round(m(yhat, y), digits=4))
m = RootMeanSquaredError()
println("\tRMSE: ", round(m(yhat, y), digits=4))
println()
println("Saving model: norclozapine_regressor_model.jlso")
MLJ.save("norclozapine_regressor_model.jlso", model_final)
println()

p = Plots.plot(p1, p2, layout=(2, 1))
savefig(p, "rr_accuracy.png")

println("Training completed.")