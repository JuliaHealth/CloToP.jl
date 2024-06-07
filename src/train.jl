@info "Loading packages"

using Pkg
# packages = ["CSV", "DataFrames", "JLD2", "MLJ", "MLJFlux", "NNlib", "Flux" "Plots", "StatsBase"]
# Pkg.add(packages)

using CSV
using DataFrames
using JLD2
using MLJ
using MLJFlux
using NNlib
using Flux
using Random
using Plots
using StatsBase
using ProgressMeter

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

@info "Loading data"

# load training data
if isfile("data/clozapine_train.csv")
    println("Loading: clozapine_train.csv")
    train_raw_data = CSV.read("data/clozapine_train.csv", header=true, DataFrame)
else
    error("File data/clozapine_train.csv cannot be opened!")
    exit(-1)
end

println("Number of entries: $(nrows(train_raw_data))")
println("Number of features: $(ncol(train_raw_data) - 2)")
println()

# preprocess
@info "Preprocessing"
y1 = train_raw_data[:, 1]
y2 = repeat(["norm"], length(y1))
y2[y1 .> 550] .= "high"
y3 = train_raw_data[:, 2]
x = Matrix(train_raw_data[:, 3:end])


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

init_n_hidden = 32
init_dropout = 0.01
init_η = 0.01
init_epochs = 1000
init_batch_size = 2
init_λ = 0.0
init_α = 0.001

optimize_classifier = true

nnc = @MLJ.load NeuralNetworkClassifier pkg=MLJFlux verbosity=0
model = nnc(builder = MLJFlux.Short(n_hidden = init_n_hidden, 
                                    dropout = init_dropout, 
                                    σ = NNlib.tanh_fast),
            finaliser = NNlib.softmax, 
            optimiser = Adam(init_η, (0.9, 0.999), IdDict{Any,Any}()),
            loss = Flux.Losses.crossentropy, 
            epochs = init_epochs, 
            batch_size = init_batch_size, 
            lambda = init_λ, 
            alpha = init_α)

if optimize_classifier

    mach = machine(model, x[train_idx, :], y2[train_idx], scitype_check_level=0)
    MLJ.fit!(mach, force=true, verbosity=0)

    error_first = cross_entropy(MLJ.predict(mach, x[test_idx, :]), y2[test_idx])
    println("Initial cross-entropy: $(round(error_first, digits=4))")

    @info "Optimizing: n_hidden"
    n_hidden = 2:2:256
    training_error = zeros(length(n_hidden))
    progbar = Progress(length(n_hidden), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(n_hidden)
        model.builder.n_hidden = n_hidden[idx]
        MLJ.fit!(mach, verbosity=0)
        training_error[idx] = cross_entropy(MLJ.predict(mach, x[test_idx, :]), y2[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model.builder.n_hidden = n_hidden[idx]
    MLJ.fit!(mach, verbosity=0)
    error_new = cross_entropy(MLJ.predict(mach, x[test_idx, :]), y2[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model.builder.n_hidden = init_n_hidden
    end

    @info "Optimizing: dropout"
    drop = 0.0:0.01:1.0
    training_error = zeros(length(drop))
    progbar = Progress(length(drop), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(drop)
        model.builder.dropout = drop[idx]
        MLJ.fit!(mach, verbosity=0)
        training_error[idx] = cross_entropy(MLJ.predict(mach, x[test_idx, :]), y2[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model.builder.dropout = drop[idx]
    MLJ.fit!(mach, verbosity=0)
    error_new = cross_entropy(MLJ.predict(mach, x[test_idx, :]), y2[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model.builder.dropout = init_dropout
    end

    @info "Optimizing: epochs"
    ep = 100:100:10_000
    training_error = zeros(length(ep))
    progbar = Progress(length(ep), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(ep)
        model.epochs = ep[idx]
        MLJ.fit!(mach, verbosity=0)
        training_error[idx] = cross_entropy(MLJ.predict(mach, x[test_idx, :]), y2[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model.epochs = ep[idx]
    MLJ.fit!(mach, verbosity=0)
    error_new = cross_entropy(MLJ.predict(mach, x[test_idx, :]), y2[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model.epochs = init_epochs
    end

    @info "Optimizing: batch_size"
    batch_size = 1:10
    training_error = zeros(length(batch_size))
    progbar = Progress(length(batch_size), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(batch_size)
        model.batch_size = batch_size[idx]
        MLJ.fit!(mach, verbosity=0)
        training_error[idx] = cross_entropy(MLJ.predict(mach, x[test_idx, :]), y2[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model.batch_size = batch_size[idx]
    MLJ.fit!(mach, verbosity=0)
    error_new = cross_entropy(MLJ.predict(mach, x[test_idx, :]), y2[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model.batch_size = init_batch_size
    end

    @info "Optimizing: η"
    η = 0.001:0.001:0.1
    training_error = zeros(length(η))
    progbar = Progress(length(η), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(η)
        model.optimiser.eta = η[idx]
        MLJ.fit!(mach, verbosity=0)
        training_error[idx] = cross_entropy(MLJ.predict(mach, x[test_idx, :]), y2[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model.optimiser.eta = η[idx]
    MLJ.fit!(mach, verbosity=0)
    error_new = cross_entropy(MLJ.predict(mach, x[test_idx, :]), y2[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model.optimiser.eta = init_η
    end

    @info "Optimizing: λ"
    λ = 0.0:0.1:10
    training_error = zeros(length(λ))
    progbar = Progress(length(λ), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(λ)
        model.lambda = λ[idx]
        MLJ.fit!(mach, verbosity=0)
        training_error[idx] = cross_entropy(MLJ.predict(mach, x[test_idx, :]), y2[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model.lambda = λ[idx]
    MLJ.fit!(mach, verbosity=0)
    error_new = cross_entropy(MLJ.predict(mach, x[test_idx, :]), y2[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model.lambda = init_λ
    end

    @info "Optimizing: α"
    α = 0.001:0.001:1
    training_error = zeros(length(α))
    progbar = Progress(length(α), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(α)
        model.alpha = α[idx]
        MLJ.fit!(mach, verbosity=0)
        training_error[idx] = cross_entropy(MLJ.predict(mach, x[test_idx, :]), y2[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model.alpha = α[idx]
    MLJ.fit!(mach, verbosity=0)
    error_new = cross_entropy(MLJ.predict(mach, x[test_idx, :]), y2[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model.alpha = init_α
    end

    MLJ.fit!(mach, verbosity=0)
    error_last = cross_entropy(MLJ.predict(mach, x[test_idx, :]), y2[test_idx])
    println("Final cross-entropy: $(round(error_last, digits=4))")
    println("Model parameters:")
    println("  n_hidden: $(model.builder.n_hidden)")
    println("  dropout: $(model.builder.dropout)")
    println("  η: $(model.optimiser.eta)")
    println("  epochs: $(model.epochs)")
    println("  batch_size: $(model.batch_size)")
    println("  λ: $(model.lambda)")
    println("  α: $(model.alpha)")
    println()

    if error_last > error_first
        model = nnc(builder = MLJFlux.Short(n_hidden = init_n_hidden, 
                                            dropout = init_dropout, 
                                            σ = NNlib.tanh_fast),
                    finaliser = NNlib.softmax, 
                    optimiser = Adam(init_η, (0.9, 0.999), IdDict{Any,Any}()),
                    loss = Flux.Losses.crossentropy, 
                    epochs = init_epochs, 
                    batch_size = init_batch_size, 
                    lambda = init_λ, 
                    alpha = init_α)
    end
end

@info "Classifier accuracy"

println("Training:")
mach = machine(model, x[train_idx, :], y2[train_idx], scitype_check_level=0)
MLJ.fit!(mach, force=true, verbosity=0)
yhat = MLJ.predict(mach, x[train_idx, :])
println("  cross-entropy: ", round(cross_entropy(yhat, y2[train_idx]), digits=2))
println("  log-loss: ", round(log_loss(yhat, y2[train_idx]), digits=2))
println("  AUC: ", round(auc(yhat, y2[train_idx]), digits=2))
println("  misclassification rate: ", round(misclassification_rate(mode.(yhat), y2[train_idx]), digits=2))
println("  accuracy: ", round(1 - misclassification_rate(mode.(yhat), y2[train_idx]), digits=2))
println("Confusion matrix:")
cm = confusion_matrix(mode.(yhat), y2[train_idx])
println("  sensitivity (TPR): ", round(cm.mat[1, 1] / sum(cm.mat[:, 1]), digits=2))
println("  specificity (TNR): ", round(cm.mat[2, 2] / sum(cm.mat[:, 2]), digits=2))
println("""
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │ $(lpad(cm.mat[4], 4, " ")) │ $(lpad(cm.mat[2], 4, " ")) │
prediction      ├──────┼──────┤
           high │ $(lpad(cm.mat[3], 4, " ")) │ $(lpad(cm.mat[1], 4, " ")) │
                └──────┴──────┘
         """)

println("Validating:")
yhat = MLJ.predict(mach, x[test_idx, :])
println("  cross-entropy: ", round(cross_entropy(yhat, y2[test_idx]), digits=2))
println("  log-loss: ", round(log_loss(yhat, y2[test_idx]) |> mean, digits=2))
println("  AUC: ", round(auc(yhat, y2[test_idx]), digits=2))
println("  misclassification rate: ", round(misclassification_rate(mode.(yhat), y2[test_idx]), digits=2))
println("  accuracy: ", round(1 - misclassification_rate(mode.(yhat), y2[test_idx]), digits=2))
println("Confusion matrix:")
cm = confusion_matrix(mode.(yhat), y2[test_idx])
println("  sensitivity (TPR): ", round(cm.mat[1, 1] / sum(cm.mat[:, 1]), digits=2))
println("  specificity (TNR): ", round(cm.mat[2, 2] / sum(cm.mat[:, 2]), digits=2))
println("""
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │ $(lpad(cm.mat[4], 4, " ")) │ $(lpad(cm.mat[2], 4, " ")) │
prediction      ├──────┼──────┤
           high │ $(lpad(cm.mat[3], 4, " ")) │ $(lpad(cm.mat[1], 4, " ")) │
                └──────┴──────┘
         """)

@info "Creating regressor model: clozapine"

init_n_hidden = 64
init_dropout = 0.1
init_η = 0.01
init_epochs = 1000
init_batch_size = 2
init_λ = 0.1
init_α = 0.0

optimize_clo_regressor = true

nnr = @MLJ.load NeuralNetworkRegressor pkg=MLJFlux verbosity=0
model_clo = nnr(builder = MLJFlux.Short(n_hidden=init_n_hidden,
                                        dropout=init_dropout,
                                        σ=hardtanh),
                optimiser = Adam(init_η, (0.9, 0.999), 1.0e-8, IdDict{Any, Any}()), 
                loss = Flux.Losses.mse, 
                epochs = init_epochs, 
                batch_size = init_batch_size, 
                lambda = init_λ, 
                alpha = init_α) 

if optimize_clo_regressor

    mach_clo = machine(model_clo, x[train_idx, :], y1[train_idx], scitype_check_level=0)
    MLJ.fit!(mach_clo, force=true, verbosity=0)

    error_first = RootMeanSquaredError()(MLJ.predict(mach_clo, x[test_idx, :]), y1[test_idx])
    println("Initial RMSE: $(round(error_first, digits=4))")

    @info "Optimizing: n_hidden"
    n_hidden = 2:2:256
    training_error = zeros(length(n_hidden))
    progbar = Progress(length(n_hidden), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(n_hidden)
        model_clo.builder.n_hidden = n_hidden[idx]
        MLJ.fit!(mach_clo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_clo, x[test_idx, :]), y1[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_clo.builder.n_hidden = n_hidden[idx]
    MLJ.fit!(mach_clo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_clo, x[test_idx, :]), y1[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_clo.builder.n_hidden = init_n_hidden
    end

    @info "Optimizing: dropout"
    drop = 0.0:0.01:1.0
    training_error = zeros(length(drop))
    progbar = Progress(length(drop), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(drop)
        model_clo.builder.dropout = drop[idx]
        MLJ.fit!(mach_clo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_clo, x[test_idx, :]), y1[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_clo.builder.dropout = drop[idx]
    MLJ.fit!(mach_clo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_clo, x[test_idx, :]), y1[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_clo.builder.dropout = init_dropout
    end

    @info "Optimizing: epochs"
    ep = 100:100:10_000
    training_error = zeros(length(ep))
    progbar = Progress(length(ep), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(ep)
        model_clo.epochs = ep[idx]
        MLJ.fit!(mach_clo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_clo, x[test_idx, :]), y1[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_clo.epochs = ep[idx]
    MLJ.fit!(mach_clo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_clo, x[test_idx, :]), y1[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_clo.epochs = init_epochs
    end

    @info "Optimizing: batch_size"
    batch_size = 1:10
    training_error = zeros(length(batch_size))
    progbar = Progress(length(batch_size), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(batch_size)
        model_clo.batch_size = batch_size[idx]
        MLJ.fit!(mach_clo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_clo, x[test_idx, :]), y1[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_clo.batch_size = batch_size[idx]
    MLJ.fit!(mach_clo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_clo, x[test_idx, :]), y1[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_clo.batch_size = init_batch_size
    end

    @info "Optimizing: η"
    η = 0.001:0.001:0.1
    training_error = zeros(length(η))
    progbar = Progress(length(η), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(η)
        model_clo.optimiser.eta = η[idx]
        MLJ.fit!(mach_clo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_clo, x[test_idx, :]), y1[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_clo.optimiser.eta = η[idx]
    MLJ.fit!(mach_clo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_clo, x[test_idx, :]), y1[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_clo.optimiser.eta = init_η
    end

    @info "Optimizing: λ"
    λ = 0.0:0.1:10
    training_error = zeros(length(λ))
    progbar = Progress(length(λ), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(λ)
        model_clo.lambda = λ[idx]
        MLJ.fit!(mach_clo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_clo, x[test_idx, :]), y1[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_clo.lambda = λ[idx]
    MLJ.fit!(mach_clo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_clo, x[test_idx, :]), y1[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_clo.lambda = init_λ
    end

    @info "Optimizing: α"
    α = 0.001:0.001:1
    training_error = zeros(length(α))
    progbar = Progress(length(α), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(α)
        model_clo.alpha = α[idx]
        MLJ.fit!(mach_clo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_clo, x[test_idx, :]), y1[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_clo.alpha = α[idx]
    MLJ.fit!(mach_clo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_clo, x[test_idx, :]), y1[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_clo.alpha = init_α
    end

    MLJ.fit!(mach_clo, verbosity=0)
    error_last = RootMeanSquaredError()(MLJ.predict(mach_clo, x[test_idx, :]), y1[test_idx])
    println("Final RMSE: $(round(error_last, digits=4))")
    println("Model parameters:")
    println("  n_hidden: $(model_clo.builder.n_hidden)")
    println("  dropout: $(model_clo.builder.dropout)")
    println("  η: $(model_clo.optimiser.eta)")
    println("  epochs: $(model_clo.epochs)")
    println("  batch_size: $(model_clo.batch_size)")
    println("  λ: $(model_clo.lambda)")
    println("  α: $(model_clo.alpha)")
    println()

    if error_last > error_first
        model_clo = nnr(builder = MLJFlux.Short(n_hidden=init_n_hidden,
                                                dropout=init_dropout,
                                                σ=hardtanh),
                        optimiser = Adam(init_η, (0.9, 0.999), 1.0e-8, IdDict{Any, Any}()), 
                        loss = Flux.Losses.mse, 
                        epochs = init_epochs, 
                        batch_size = init_batch_size, 
                        lambda = init_λ, 
                        alpha = init_α) 
    end
end

@info "Regressor accuracy: clozapine"

mach_clo = machine(model_clo, x[train_idx, :], y1[train_idx], scitype_check_level=0)
MLJ.fit!(mach_clo, force=true, verbosity=0)

println("Training:")
yhat = MLJ.predict(mach_clo, x[train_idx, :])
m = RSquared()
println("  R²: ", round(m(yhat, y1[train_idx]), digits=2))
m = RootMeanSquaredError()
println("  RMSE: ", round(m(yhat, y1[train_idx]), digits=2))

println("Validating:")
yhat = MLJ.predict(mach_clo, x[test_idx, :])
m = RSquared()
println("  R²: ", round(m(yhat, y1[test_idx]), digits=2))
m = RootMeanSquaredError()
println("  RMSE: ", round(m(yhat, y1[test_idx]), digits=2))
println()

@info "Creating regressor model: norclozapine"

init_n_hidden = 64
init_dropout = 0.1
init_η = 0.01
init_epochs = 1000
init_batch_size = 2
init_λ = 0.1
init_α = 0.0

optimize_nclo_regressor = true

model_nclo = nnr(builder = MLJFlux.Short(n_hidden=init_n_hidden,
                                         dropout=init_dropout,
                                         σ=hardtanh),
                 optimiser = Adam(init_η, (0.9, 0.999), 1.0e-8, IdDict{Any, Any}()), 
                 loss = Flux.Losses.mse, 
                 epochs = init_epochs, 
                 batch_size = init_batch_size, 
                 lambda = init_λ, 
                 alpha = init_α) 

if optimize_nclo_regressor

    mach_nclo = machine(model_nclo, x[train_idx, :], y3[train_idx], scitype_check_level=0)
    MLJ.fit!(mach_nclo, force=true, verbosity=0)

    error_first = RootMeanSquaredError()(MLJ.predict(mach_nclo, x[test_idx, :]), y3[test_idx])
    println("Initial RMSE: $(round(error_first, digits=4))")

    @info "Optimizing: n_hidden"
    n_hidden = 2:2:256
    training_error = zeros(length(n_hidden))
    progbar = Progress(length(n_hidden), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(n_hidden)
        model_nclo.builder.n_hidden = n_hidden[idx]
        MLJ.fit!(mach_nclo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_nclo, x[test_idx, :]), y3[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_nclo.builder.n_hidden = n_hidden[idx]
    MLJ.fit!(mach_nclo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_nclo, x[test_idx, :]), y3[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_nclo.builder.n_hidden = init_n_hidden
    end

    @info "Optimizing: dropout"
    drop = 0.0:0.01:1.0
    training_error = zeros(length(drop))
    progbar = Progress(length(drop), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(drop)
        model_nclo.builder.dropout = drop[idx]
        MLJ.fit!(mach_nclo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_nclo, x[test_idx, :]), y3[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_nclo.builder.dropout = drop[idx]
    MLJ.fit!(mach_nclo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_nclo, x[test_idx, :]), y3[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_nclo.builder.dropout = init_dropout
    end

    @info "Optimizing: epochs"
    ep = 100:100:10_000
    training_error = zeros(length(ep))
    progbar = Progress(length(ep), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(ep)
        model_nclo.epochs = ep[idx]
        MLJ.fit!(mach_nclo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_nclo, x[test_idx, :]), y3[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_nclo.epochs = ep[idx]
    MLJ.fit!(mach_nclo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_nclo, x[test_idx, :]), y3[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_nclo.epochs = init_epochs
    end

    @info "Optimizing: batch_size"
    batch_size = 1:10
    training_error = zeros(length(batch_size))
    progbar = Progress(length(batch_size), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(batch_size)
        model_nclo.batch_size = batch_size[idx]
        MLJ.fit!(mach_nclo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_nclo, x[test_idx, :]), y3[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_nclo.batch_size = batch_size[idx]
    MLJ.fit!(mach_nclo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_nclo, x[test_idx, :]), y3[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_nclo.batch_size = init_batch_size
    end

    @info "Optimizing: η"
    η = 0.001:0.001:0.1
    training_error = zeros(length(η))
    progbar = Progress(length(η), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(η)
        model_nclo.optimiser.eta = η[idx]
        MLJ.fit!(mach_nclo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_nclo, x[test_idx, :]), y3[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_nclo.optimiser.eta = η[idx]
    MLJ.fit!(mach_nclo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_nclo, x[test_idx, :]), y3[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_nclo.optimiser.eta = init_η
    end

    @info "Optimizing: lambda"
    lambda = 0.0:0.1:10
    training_error = zeros(length(lambda))
    progbar = Progress(length(lambda), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(lambda)
        model_nclo.lambda = lambda[idx]
        MLJ.fit!(mach_nclo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_nclo, x[test_idx, :]), y1[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_nclo.lambda = lambda[idx]
    MLJ.fit!(mach_nclo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_nclo, x[test_idx, :]), y1[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_nclo.lambda = init_λ
    end

    @info "Optimizing: α"
    α = 0.001:0.001:1
    training_error = zeros(length(α))
    progbar = Progress(length(α), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(α)
        model_nclo.alpha = α[idx]
        MLJ.fit!(mach_nclo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_nclo, x[test_idx, :]), y3[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_nclo.alpha = α[idx]
    MLJ.fit!(mach_nclo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_nclo, x[test_idx, :]), y3[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_nclo.alpha = init_α
    end

    MLJ.fit!(mach_nclo, verbosity=0)
    error_last = RootMeanSquaredError()(MLJ.predict(mach_nclo, x[test_idx, :]), y3[test_idx])
    println("Final RMSE: $(round(error_last, digits=4))")
    println("Model parameters:")
    println("  n_hidden: $(model_nclo.builder.n_hidden)")
    println("  dropout: $(model_nclo.builder.dropout)")
    println("  η: $(model_nclo.optimiser.eta)")
    println("  epochs: $(model_nclo.epochs)")
    println("  batch_size: $(model_nclo.batch_size)")
    println("  λ: $(model_nclo.lambda)")
    println("  α: $(model_nclo.alpha)")
    println()
    
    if error_last > error_first
        model_nclo = nnr(builder = MLJFlux.Short(n_hidden=init_n_hidden,
                                                 dropout=init_dropout,
                                                 σ=hardtanh),
                         optimiser = Adam(init_η, (0.9, 0.999), 1.0e-8, IdDict{Any, Any}()), 
                         loss = Flux.Losses.mse, 
                         epochs = init_epochs, 
                         batch_size = init_batch_size, 
                         lambda = init_λ, 
                         alpha = init_α) 
    end
end

@info "Regressor accuracy: norclozapine"

mach_nclo = machine(model_nclo, x[train_idx, :], y3[train_idx], scitype_check_level=0)
MLJ.fit!(mach_nclo, force=true, verbosity=0)

println("Training:")
yhat = MLJ.predict(mach_nclo, x[train_idx, :])
m = RSquared()
println("  R²: ", round(m(yhat, y3[train_idx]), digits=2))
m = RootMeanSquaredError()
println("  RMSE: ", round(m(yhat, y3[train_idx]), digits=2))

println("Validating:")
yhat = MLJ.predict(mach_nclo, x[test_idx, :])
m = RSquared()
println("  R²: ", round(m(yhat, y3[test_idx]), digits=2))
m = RootMeanSquaredError()
println("  RMSE: ", round(m(yhat, y3[test_idx]), digits=2))
println()

@info "Training final model"

mach_clas = machine(model, x, y2, scitype_check_level=0)
MLJ.fit!(mach_clas, force=true, verbosity=0)
yhat = MLJ.predict(mach_clas, x)
println("Classifier accuracy:")
println("  log-loss: ", round(log_loss(yhat, y2) |> mean, digits=2))
println("  AUC: ", round(auc(yhat, y2), digits=2))
println("  misclassification rate: ", round(misclassification_rate(mode.(yhat), y2), digits=2))
println("  accuracy: ", round(1 - misclassification_rate(mode.(yhat), y2), digits=2))
println("Confusion matrix:")
cm = confusion_matrix(mode.(yhat), y2)
println("  sensitivity (TPR): ", round(cm.mat[1, 1] / sum(cm.mat[:, 1]), digits=2))
println("  specificity (TNR): ", round(cm.mat[2, 2] / sum(cm.mat[:, 2]), digits=2))
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
MLJ.fit!(mach_clo, force=true, verbosity=0)
yhat = MLJ.predict(mach_clo, x)
sorting_idx = sortperm(y1)
p1 = Plots.plot(y1[sorting_idx], label="data", ylims=(0, 2000), xlabel="patients", ylabel="clozapine [ng/mL]")
p1 = Plots.plot!(yhat[sorting_idx], label="prediction", line=:dot, lw=2)
println("Regressor accuracy")
m = RSquared()
println("  R²: ", round(m(yhat, y1), digits=2))
m = RootMeanSquaredError()
println("  RMSE: ", round(m(yhat, y1), digits=2))

println("Predicting: NORCLOZAPINE")
mach_nclo = machine(model_nclo, x, y3, scitype_check_level=0)
MLJ.fit!(mach_nclo, force=true, verbosity=0)
yhat = MLJ.predict(mach_nclo, x)
sorting_idx = sortperm(y3)
p2 = Plots.plot(y3[sorting_idx], label="data", ylims=(0, 2000), xlabel="patients", ylabel="norclozapine [ng/mL]")
p2 = Plots.plot!(yhat[sorting_idx], label="prediction", line=:dot, lw=2)
println("Regressor accuracy")
m = RSquared()
println("  R²: ", round(m(yhat, y3), digits=2))
m = RootMeanSquaredError()
println("  RMSE: ", round(m(yhat, y3), digits=2))
println()

@info "Saving models"

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
