println("Importing packages")
using CSV
using DataFrames
using MLJ
using MLJDecisionTreeInterface
using Random
using StatsBase
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

function preprocess(train_raw_data)
    y1 = train_raw_data[:, 1]
    y2 = repeat(["norm"], length(y1))
    y2[y1 .> 550] .= "high"
    x = Matrix(train_raw_data[:, 3:end])
    return x, y1, y2
end

x, y1, y2 = preprocess(train_raw_data)

# add zero-dose data
#=
n = 10_000
y1_z = zeros(n)
y2_z = repeat(["norm"], n)
age_z = rand(18:100, n)
sex_z = rand(0:1, n)
dose_z = zeros(n)
bmi_z = rand(12:100, n)
weight_z = rand(10:100, n)
duration_z = rand(1:1000, n)
crp_z = rand(0:0.1:20, n)
cyp_z = rand(0:3, n, 6)
x_z = hcat(sex_z, age_z, dose_z, bmi_z, weight_z, duration_z, crp_z, cyp_z)
x = vcat(x, x_z)
y1 = vcat(y1, y1_z)
y2 = vcat(y2, y2_z)
=#

println("Number of entries: $(size(y1, 1))")

# standardize
println("Processing: standardize")
# data = hcat(y1, x[:, 2:end])
data = x[:, 2:7]
scaler = StatsBase.fit(ZScoreTransform, data, dims=1)
data = StatsBase.transform(scaler, data)
data[isnan.(data)] .= 0
x_gender = Bool.(x[:, 1])
x_cont = data[:, :]
x_rest = x[:, 8:end]
x1 = DataFrame(:male=>x_gender)
# x2 = DataFrame(x_cont, ["age", "dose", "bmi", "weight", "duration", "crp"])
x2 = DataFrame(x_cont[:, [1, 2, 3, 6]], ["age", "dose", "bmi", "crp"])
x3 = DataFrame(x_rest, ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
x = hcat(x1, x2, x3)
# x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :weight=>Continuous, :duration=>Continuous, :crp=>Continuous, :inducers_3a4=>Count,  :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)
x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Count, :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)
y = DataFrame(group=y2)
y = coerce(y.group, OrderedFactor{2})
# scitype(y)
# levels(y)

# train_idx, test_idx = partition(eachindex(y), 0.7, rng=123) # 70:30 split

println("Creating classifier model: RandomForrest")
rfc = @MLJ.load RandomForestClassifier pkg=DecisionTree verbosity=0
#=
info(RandomForestClassifier)
evaluate(model,
         x, y,
         resampling=CV(nfolds=10),
         measures=[log_loss, accuracy, f1score, misclassification_rate, cross_entropy])
n_trees_range = range(Int, :n_trees, lower=1, upper=1000)
max_depth_range = range(Int, :max_depth, lower=1, upper=100)
min_samples_leaf_range = range(Int, :min_samples_leaf, lower=1, upper=100)
min_samples_split_range = range(Int, :min_samples_split, lower=1, upper=100)
n_subfeatures_range = range(Int, :n_subfeatures, lower=1, upper=11)
min_purity_increase_range = range(Float64, :min_purity_increase, lower=0.1, upper=1.0)
sampling_fraction_range = range(Float64, :sampling_fraction, lower=0.1, upper=1.0)
params = [n_trees_range, max_depth_range, min_samples_leaf_range, min_samples_split_range, n_subfeatures_range, min_purity_increase_range, sampling_fraction_range]
measures = [log_loss, accuracy, f1score, misclassification_rate, cross_entropy]
measures = [log_loss, auc]
self_tuning_rfc1 = TunedModel(model=rfc(feature_importance=:split),
                              resampling=CV(nfolds=5),
                              tuning=Grid(resolution=5),
                              range=params,
                              measure=measures)
self_tuning_rfc2 = TunedModel(model=rfc(feature_importance=:impurity),
                              resampling=CV(nfolds=5),
                              tuning=Grid(resolution=5),
                              range=params,
                              measure=measures)
m_self_tuning_rfc1 = machine(self_tuning_rfc1, x, y)
m_self_tuning_rfc2 = machine(self_tuning_rfc2, x, y)
MLJ.fit!(m_self_tuning_rfc1)
MLJ.fit!(m_self_tuning_rfc2)

MLJ.fit!(m_self_tuning_rfc, rows=train_idx)
MLJ.fit!(m_self_tuning_rfc, rows=test_idx)

fitted_params(m_self_tuning_rfc1).best_model
fitted_params(m_self_tuning_rfc2).best_model
report(m_self_tuning_rfc1).best_history_entry
report(m_self_tuning_rfc2).best_history_entry

model = fitted_params(m_self_tuning_rfc1).best_model
model = fitted_params(m_self_tuning_rfc2).best_model

model = RandomForestClassifier(max_depth = 50, 
                               min_samples_leaf = 1, 
                               min_samples_split = 2, 
                               min_purity_increase = 0.1, 
                               n_subfeatures = 8, 
                               n_trees = 251, 
                               sampling_fraction = 0.55, 
                               feature_importance = :impurity, 
                               rng = Random._GLOBAL_RNG())

report(m_self_tuning_rfc).best_history_entry
=#
model = rfc(max_depth = -1, 
            min_samples_leaf = 1, 
            min_samples_split = 2, 
            min_purity_increase = 0.0, 
            n_subfeatures = -1, 
            n_trees = 1750, 
            sampling_fraction = 0.95, 
            feature_importance = :split)
mach = machine(model, x, y)
MLJ.fit!(mach, force=true, verbosity=0)
yhat = MLJ.predict(mach, x)

println("RF model accuracy report:")
println("\tlog_loss: ", round(log_loss(yhat, y) |> mean, digits=4))
println("\tmisclassification_rate: ", round(misclassification_rate(mode.(yhat), y), digits=2))
println("\taccuracy: ", 1 - round(misclassification_rate(mode.(yhat), y), digits=2))
println("confusion matrix:")
cm = confusion_matrix(mode.(yhat), y)
println("\tsensitivity (TP): ", round(count(mode.(yhat) .== "high") / count(y .== "high"), digits=2))
println("\tspecificity (TN): ", round(count(mode.(yhat) .== "norm") / count(y .== "norm"), digits=2))
println("""
                    group
                  norm   high   
                ┌──────┬──────┐
           norm │ $(lpad(cm[4], 4, " ")) │ $(lpad(cm[3], 4, " ")) │
prediction      ├──────┼──────┤
           high │ $(lpad(cm[2], 4, " ")) │ $(lpad(cm[1], 4, " ")) │
                └──────┴──────┘
         """)
println("Saving model: clozapine_classifier_model.jlso")
MLJ.save("clozapine_classifier_model.jlso", mach)

println("Saving: scaler.jld")
JLD2.save_object("scaler.jld", scaler)

println()
println("Training completed.")