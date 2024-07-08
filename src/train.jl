@info "Loading packages"

optimize_clo_regressor = false
optimize_nclo_regressor = false
standardize_data = true

using Pkg
# packages = ["CSV", "DataFrames", "JLD2", "Flux", "MLJ", "MLJFlux", "NNlib", "Optimisers", "Plots", "ProgressMeter", "StatsBase"]
# Pkg.add(packages)

using CSV
using DataFrames
using JLD2
using MLJ
using MLJFlux
using NNlib
using Optimisers
using Flux
using Random
using Plots
using ProgressMeter
using StatsBase

m = Pkg.Operations.Context().env.manifest
println("       CSV $(m[findfirst(v -> v.name == "CSV", m)].version)")
println("DataFrames $(m[findfirst(v -> v.name == "DataFrames", m)].version)")
println("      JLD2 $(m[findfirst(v -> v.name == "JLD2", m)].version)")
println("      Flux $(m[findfirst(v -> v.name == "Flux", m)].version)")
println("       MLJ $(m[findfirst(v -> v.name == "MLJ", m)].version)")
println("   MLJFlux $(m[findfirst(v -> v.name == "MLJFlux", m)].version)")
println("     NNlib $(m[findfirst(v -> v.name == "MLJFlux", m)].version)")
println("Optimisers $(m[findfirst(v -> v.name == "Optimisers", m)].version)")
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

println()
println("Number of entries: $(nrows(train_raw_data))")
println("Number of features: $(ncol(train_raw_data) - 2)")
println()

# preprocess
@info "Preprocessing"
clo_level = train_raw_data[:, 1]
nclo_level = train_raw_data[:, 2]
x = Matrix(train_raw_data[:, 3:end])

# CLOZAPINE

# add zero-dose data for each patient
# we need artificial zero-dose data to force the lowest concentration to equal 0
clo_level_z = zeros(size(x, 1))
nclo_level_z = zeros(size(x, 1))
x_z1 = deepcopy(x)
x_z1[:, 1] .= 0
x_z1[:, 3] .= 0
append!(clo_level, clo_level_z)
append!(nclo_level, nclo_level_z)
data_clo = vcat(x, x_z1)
data_clo = hcat(data_clo[:, 1], nclo_level, data_clo[:, 2:end])

# standardize
println("Standardizing")
scaler_clo = StatsBase.fit(ZScoreTransform, data_clo[:, 2:6], dims=1)
standardize_data && (data_clo[:, 2:6] = StatsBase.transform(scaler_clo, data_clo[:, 2:6]))
data_clo[isnan.(data_clo)] .= 0

# create DataFrame
x1 = DataFrame(:male=>data_clo[:, 1])
x2 = DataFrame(data_clo[:, 2:6], ["nclo", "age", "dose", "bmi", "crp"])
x3 = DataFrame(data_clo[:, 7:end], ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
data_clo = Float32.(hcat(x1, x2, x3))
data_clo = coerce(data_clo, :male=>OrderedFactor{2}, :nclo=>Continuous, :age=>Continuous, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Continuous, :inhibitors_3a4=>Continuous, :substrates_3a4=>Continuous, :inducers_1a2=>Continuous, :inhibitors_1a2=>Continuous, :substrates_1a2=>Continuous)

# NORCLOZAPINE

# add zero-dose data for each patient
# we need artificial zero-dose data to force the lowest concentration to equal 0
x = Matrix(train_raw_data[:, 3:end])
x_z1 = deepcopy(x)
x_z1[:, 1] .= 0
x_z1[:, 3] .= 0
data_nclo = vcat(x, x_z1)

# standardize
scaler_nclo = StatsBase.fit(ZScoreTransform, data_nclo[:, 2:5], dims=1)
standardize_data && (data_nclo[:, 2:5] = StatsBase.transform(scaler_nclo, data_nclo[:, 2:5]))
data_nclo[isnan.(data_nclo)] .= 0

# create DataFrame
x1 = DataFrame(:male=>data_nclo[:, 1])
x2 = DataFrame(data_nclo[:, 2:5], ["age", "dose", "bmi", "crp"])
x3 = DataFrame(data_nclo[:, 6:end], ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
data_nclo = Float32.(hcat(x1, x2, x3))
data_nclo = coerce(data_nclo, :male=>OrderedFactor{2}, :age=>Continuous, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Continuous, :inhibitors_3a4=>Continuous, :substrates_3a4=>Continuous, :inducers_1a2=>Continuous, :inhibitors_1a2=>Continuous, :substrates_1a2=>Continuous)

println("Splitting: 70:30")
train_idx, test_idx = partition(eachindex(clo_level), 0.7, shuffle=true)
println()

@info "Creating regressor model: clozapine"

if standardize_data
    # model parameters for standardized data
    init_n_hidden= 84
    init_dropout = 0.07
    init_η = 0.013
    init_η = 0.01
    init_epochs = 5600
    init_batch_size = 2
    init_λ = 0.1
    init_α = 0.0
else
    # model parameters for non-standardized data
    init_n_hidden = 84
    init_dropout = 0.07
    init_η = 0.01
    init_epochs = 7300
    init_batch_size = 7
    init_λ = 6.4
    init_α = 0.97
end

nnr = @MLJ.load NeuralNetworkRegressor pkg=MLJFlux verbosity=0
model_clo = nnr(builder = MLJFlux.Short(n_hidden=init_n_hidden,
                                        dropout=init_dropout,
                                        σ=hardtanh),
                optimiser = Optimisers.Adam(init_η, (0.9, 0.999), 1.0e-8), 
                loss = Flux.Losses.mse, 
                epochs = init_epochs, 
                batch_size = init_batch_size, 
                lambda = init_λ, 
                alpha = init_α)

if optimize_clo_regressor

    mach_clo = machine(model_clo, data_clo[train_idx, :], clo_level[train_idx], scitype_check_level=0)
    MLJ.fit!(mach_clo, force=true, verbosity=0)

    error_first = RootMeanSquaredError()(MLJ.predict(mach_clo, data_clo[test_idx, :]), clo_level[test_idx])
    println("Initial RMSE: $(round(error_first, digits=4))")

    @info "Optimizing: n_hidden"
    n_hidden = 2:2:256
    training_error = zeros(length(n_hidden))
    progbar = Progress(length(n_hidden), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(n_hidden)
        model_clo.builder.n_hidden = n_hidden[idx]
        MLJ.fit!(mach_clo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_clo, data_clo[test_idx, :]), clo_level[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_clo.builder.n_hidden = n_hidden[idx]
    MLJ.fit!(mach_clo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_clo, data_clo[test_idx, :]), clo_level[test_idx])
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
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_clo, data_clo[test_idx, :]), clo_level[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_clo.builder.dropout = drop[idx]
    MLJ.fit!(mach_clo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_clo, data_clo[test_idx, :]), clo_level[test_idx])
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
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_clo, data_clo[test_idx, :]), clo_level[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_clo.epochs = ep[idx]
    MLJ.fit!(mach_clo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_clo, data_clo[test_idx, :]), clo_level[test_idx])
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
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_clo, data_clo[test_idx, :]), clo_level[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_clo.batch_size = batch_size[idx]
    MLJ.fit!(mach_clo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_clo, data_clo[test_idx, :]), clo_level[test_idx])
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
        model_clo.optimiser = Optimisers.Adam(η[idx], (0.9, 0.999), 1.0e-8)
        MLJ.fit!(mach_clo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_clo, data_clo[test_idx, :]), clo_level[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_clo.optimiser = Optimisers.Adam(η[idx], (0.9, 0.999), 1.0e-8)
    MLJ.fit!(mach_clo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_clo, data_clo[test_idx, :]), clo_level[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_clo.optimiser = Optimisers.Adam(init_η, (0.9, 0.999), 1.0e-8)
    end

    @info "Optimizing: λ"
    λ = 0.0:0.1:10
    training_error = zeros(length(λ))
    progbar = Progress(length(λ), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(λ)
        model_clo.lambda = λ[idx]
        MLJ.fit!(mach_clo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_clo, data_clo[test_idx, :]), clo_level[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_clo.lambda = λ[idx]
    MLJ.fit!(mach_clo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_clo, data_clo[test_idx, :]), clo_level[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_clo.lambda = init_λ
    end

    @info "Optimizing: α"
    # α = 0.001:0.001:1
    α = 0.01:0.01:1
    training_error = zeros(length(α))
    progbar = Progress(length(α), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(α)
        model_clo.alpha = α[idx]
        MLJ.fit!(mach_clo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_clo, data_clo[test_idx, :]), clo_level[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_clo.alpha = α[idx]
    MLJ.fit!(mach_clo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_clo, data_clo[test_idx, :]), clo_level[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_clo.alpha = init_α
    end

    MLJ.fit!(mach_clo, verbosity=0)
    error_last = RootMeanSquaredError()(MLJ.predict(mach_clo, data_clo[test_idx, :]), clo_level[test_idx])
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
                        optimiser = Optimisers.Adam(init_η, (0.9, 0.999), 1.0e-8), 
                        loss = Flux.Losses.mse, 
                        epochs = init_epochs, 
                        batch_size = init_batch_size, 
                        lambda = init_λ, 
                        alpha = init_α) 
    end
end

mach_clo = machine(model_clo, data_clo[train_idx, :], clo_level[train_idx], scitype_check_level=0)
MLJ.fit!(mach_clo, force=true, verbosity=0)

println("Regressor accuracy: training")
clo_level_pred = MLJ.predict(mach_clo, data_clo[train_idx, :])
println("  R²: ", round(RSquared()(clo_level_pred, clo_level[train_idx]), digits=2))
println("  RMSE: ", round(RootMeanSquaredError()(clo_level_pred, clo_level[train_idx]), digits=2))

println("Regressor accuracy: validating")
clo_level_pred = MLJ.predict(mach_clo, data_clo[test_idx, :])
println("  R²: ", round(RSquared()(clo_level_pred, clo_level[test_idx]), digits=2))
println("  RMSE: ", round(RootMeanSquaredError()(clo_level_pred, clo_level[test_idx]), digits=2))

@info "Training final model"

mach_clo = machine(model_clo, data_clo, clo_level, scitype_check_level=0)
MLJ.fit!(mach_clo, force=true, verbosity=0)
clo_level_pred = MLJ.predict(mach_clo, data_clo)
sorting_idx = sortperm(clo_level)
p1 = Plots.plot(clo_level[sorting_idx] .- clo_level_pred[sorting_idx], ylims=(-200, 200), xlabel="patients", ylabel="error", title="clozapine [ng/ml]", legend=false)
println("Regressor accuracy:")
println("  R²: ", round(RSquared()(clo_level_pred, clo_level), digits=2))
println("  RMSE: ", round(RootMeanSquaredError()(clo_level_pred, clo_level), digits=2))
println()

@info "Creating regressor model: norclozapine"

if standardize_data
    # model parameters for standardized data
    init_n_hidden = 64
    init_dropout = 0.1
    init_η = 0.01
    init_epochs = 1000
    init_batch_size = 2
    init_λ = 0.1
    init_α = 0.0
else
    # model parameters for non-standardized data
    init_n_hidden = 64
    init_dropout = 0.1
    init_η = 0.01
    init_epochs = 1000
    init_batch_size = 2
    init_λ = 0.1
    init_α = 0.0
end

model_nclo = nnr(builder = MLJFlux.Short(n_hidden=init_n_hidden,
                                         dropout=init_dropout,
                                         σ=hardtanh),
                 optimiser = Optimisers.Adam(init_η, (0.9, 0.999), 1.0e-8), 
                 loss = Flux.Losses.mse, 
                 epochs = init_epochs, 
                 batch_size = init_batch_size, 
                 lambda = init_λ, 
                 alpha = init_α) 

if optimize_nclo_regressor

    mach_nclo = machine(model_nclo, data_nclo[train_idx, :], nclo_level[train_idx], scitype_check_level=0)
    MLJ.fit!(mach_nclo, force=true, verbosity=0)

    error_first = RootMeanSquaredError()(MLJ.predict(mach_nclo, data_nclo[test_idx, :]), nclo_level[test_idx])
    println("Initial RMSE: $(round(error_first, digits=4))")

    @info "Optimizing: n_hidden"
    n_hidden = 2:2:256
    training_error = zeros(length(n_hidden))
    progbar = Progress(length(n_hidden), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(n_hidden)
        model_nclo.builder.n_hidden = n_hidden[idx]
        MLJ.fit!(mach_nclo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_nclo, data_nclo[test_idx, :]), nclo_level[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_nclo.builder.n_hidden = n_hidden[idx]
    MLJ.fit!(mach_nclo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_nclo, data_nclo[test_idx, :]), nclo_level[test_idx])
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
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_nclo, data_nclo[test_idx, :]), nclo_level[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_nclo.builder.dropout = drop[idx]
    MLJ.fit!(mach_nclo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_nclo, data_nclo[test_idx, :]), nclo_level[test_idx])
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
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_nclo, data_nclo[test_idx, :]), nclo_level[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_nclo.epochs = ep[idx]
    MLJ.fit!(mach_nclo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_nclo, data_nclo[test_idx, :]), nclo_level[test_idx])
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
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_nclo, data_nclo[test_idx, :]), nclo_level[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_nclo.batch_size = batch_size[idx]
    MLJ.fit!(mach_nclo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_nclo, data_nclo[test_idx, :]), nclo_level[test_idx])
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
        model_nclo.optimiser = Optimisers.Adam(η[idx], (0.9, 0.999), 1.0e-8)
        MLJ.fit!(mach_nclo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_nclo, data_nclo[test_idx, :]), nclo_level[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_nclo.optimiser = Optimisers.Adam(η[idx], (0.9, 0.999), 1.0e-8)
    MLJ.fit!(mach_nclo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_nclo, data_nclo[test_idx, :]), nclo_level[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_nclo.optimiser = Optimisers.Adam(init_η, (0.9, 0.999), 1.0e-8)
    end

    @info "Optimizing: lambda"
    lambda = 0.0:0.1:10
    training_error = zeros(length(lambda))
    progbar = Progress(length(lambda), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(lambda)
        model_nclo.lambda = lambda[idx]
        MLJ.fit!(mach_nclo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_nclo, data_nclo[test_idx, :]), nclo_level[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_nclo.lambda = lambda[idx]
    MLJ.fit!(mach_nclo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_nclo, data_nclo[test_idx, :]), nclo_level[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_nclo.lambda = init_λ
    end

    @info "Optimizing: α"
    # α = 0.001:0.001:1
    α = 0.01:0.01:1
    training_error = zeros(length(α))
    progbar = Progress(length(α), dt=1, barlen=20, color=:white)
    @Threads.threads for idx in eachindex(α)
        model_nclo.alpha = α[idx]
        MLJ.fit!(mach_nclo, verbosity=0)
        training_error[idx] = RootMeanSquaredError()(MLJ.predict(mach_nclo, data_nclo[test_idx, :]), nclo_level[test_idx])
        next!(progbar)
    end
    _, idx = findmin(training_error)
    model_nclo.alpha = α[idx]
    MLJ.fit!(mach_nclo, verbosity=0)
    error_new = RootMeanSquaredError()(MLJ.predict(mach_nclo, data_nclo[test_idx, :]), nclo_level[test_idx])
    if error_new < error_first
        error_first = error_new
    else
        model_nclo.alpha = init_α
    end

    MLJ.fit!(mach_nclo, verbosity=0)
    error_last = RootMeanSquaredError()(MLJ.predict(mach_nclo, data_nclo[test_idx, :]), nclo_level[test_idx])
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
                         optimiser = Optimisers.Adam(init_η, (0.9, 0.999), 1.0e-8), 
                         loss = Flux.Losses.mse, 
                         epochs = init_epochs, 
                         batch_size = init_batch_size, 
                         lambda = init_λ, 
                         alpha = init_α) 
    end
end

mach_nclo = machine(model_nclo, data_nclo[train_idx, :], nclo_level[train_idx], scitype_check_level=0)
MLJ.fit!(mach_nclo, force=true, verbosity=0)

println("Regressor accuracy: training")
nclo_level_pred = MLJ.predict(mach_nclo, data_nclo[train_idx, :])
println("  R²: ", round(RSquared()(nclo_level_pred, nclo_level[train_idx]), digits=2))
println("  RMSE: ", round(RootMeanSquaredError()(nclo_level_pred, nclo_level[train_idx]), digits=2))

println("Regressor accuracy: validating")
nclo_level_pred = MLJ.predict(mach_nclo, data_nclo[test_idx, :])
println("  R²: ", round(RSquared()(nclo_level_pred, nclo_level[test_idx]), digits=2))
println("  RMSE: ", round(RootMeanSquaredError()(nclo_level_pred, nclo_level[test_idx]), digits=2))

@info "Training final model"

mach_nclo = machine(model_nclo, data_nclo, nclo_level, scitype_check_level=0)
MLJ.fit!(mach_nclo, force=true, verbosity=0)
nclo_level_pred = MLJ.predict(mach_nclo, data_nclo)
sorting_idx = sortperm(nclo_level)
p2 = Plots.plot(nclo_level[sorting_idx] .- nclo_level_pred[sorting_idx], ylims=(-200, 200), xlabel="patients", ylabel="error", title="norclozapine [ng/ml]", legend=false)
println("Regressor accuracy:")
println("  R²: ", round(RSquared()(nclo_level_pred, nclo_level), digits=2))
println("  RMSE: ", round(RootMeanSquaredError()(nclo_level_pred, nclo_level), digits=2))
println()

@info "Classifying into groups"

println("Classification based on predicted clozapine level:")
nclo_level_pred[(nclo_level_pred) .< 0] .= 0
x = Matrix(train_raw_data[:, 3:end])

# add zero-dose data for each patient
clo_level_z = zeros(size(x, 1))
x_z1 = deepcopy(x)
x_z1[:, 1] .= 0
x_z1[:, 3] .= 0
x = vcat(x, x_z1)
data_group = hcat(x[:, 1], nclo_level_pred, x[:, 2:end])

# standardize
standardize_data && (data_group[:, 2:6] = StatsBase.transform(scaler_clo, data_group[:, 2:6]))
data_group[isnan.(data_group)] .= 0

# create DataFrame
x1 = DataFrame(:male=>data_group[:, 1])
x2 = DataFrame(data_group[:, 2:6], ["age", "nclo", "dose", "bmi", "crp"])
x3 = DataFrame(data_group[:, 7:end], ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
data_group = Float32.(hcat(x1, x2, x3))
data_group = coerce(data_group, :male=>OrderedFactor{2}, :age=>Continuous, :nclo=>Continuous, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Continuous, :inhibitors_3a4=>Continuous, :substrates_3a4=>Continuous, :inducers_1a2=>Continuous, :inhibitors_1a2=>Continuous, :substrates_1a2=>Continuous)

# predict clozapine level and assign patients to groups
clo_level_pred = MLJ.predict(mach_clo, data_group)
clo_level_pred[clo_level_pred .< 0] .= 0
clo_group_pred = repeat(["norm"], length(clo_level_pred))
clo_group_pred[clo_level_pred .> 550] .= "high"

cm = zeros(Int64, 2, 2)
cm[1, 1] = count(clo_group_pred[clo_group .== "norm"] .== "norm")
cm[1, 2] = count(clo_group_pred[clo_group .== "high"] .== "norm")
cm[2, 2] = count(clo_group_pred[clo_group .== "high"] .== "high")
cm[2, 1] = count(clo_group_pred[clo_group .== "norm"] .== "high")

println("Confusion matrix:")
println("  misclassification rate: ", round(sum([cm[1, 2], cm[2, 1]]) / sum(cm), digits=2))
println("  accuracy: ", round(1 - sum([cm[1, 2], cm[2, 1]]) / sum(cm), digits=2))
println("""
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │ $(lpad(cm[1, 1], 4, " ")) │ $(lpad(cm[1, 2], 4, " ")) │
prediction      ├──────┼──────┤
           high │ $(lpad(cm[2, 1], 4, " ")) │ $(lpad(cm[2, 2], 4, " ")) │
                └──────┴──────┘
         """)

println("Classification adjusted for predicted norclozapine level:")
clo_group_pred_adj = repeat(["norm"], length(clo_level_pred))
clo_group_pred_adj[clo_level_pred .> 550 .|| nclo_level_pred .> 270] .= "high"

cm = zeros(Int64, 2, 2)
cm[1, 1] = count(clo_group_pred_adj[clo_group .== "norm"] .== "norm")
cm[1, 2] = count(clo_group_pred_adj[clo_group .== "high"] .== "norm")
cm[2, 2] = count(clo_group_pred_adj[clo_group .== "high"] .== "high")
cm[2, 1] = count(clo_group_pred_adj[clo_group .== "norm"] .== "high")

println("Confusion matrix:")
println("  misclassification rate: ", round(sum([cm[1, 2], cm[2, 1]]) / sum(cm), digits=2))
println("  accuracy: ", round(1 - sum([cm[1, 2], cm[2, 1]]) / sum(cm), digits=2))
println("""
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │ $(lpad(cm[1, 1], 4, " ")) │ $(lpad(cm[1, 2], 4, " ")) │
prediction      ├──────┼──────┤
           high │ $(lpad(cm[2, 1], 4, " ")) │ $(lpad(cm[2, 2], 4, " ")) │
                └──────┴──────┘
         """)

@info "Saving models"

println("Saving: clozapine_regressor_model.jlso")
MLJ.save("models/clozapine_regressor_model.jlso", mach_clo)
println("Saving: norclozapine_regressor_model.jlso")
MLJ.save("models/norclozapine_regressor_model.jlso", mach_nclo)
println("Saving: scaler_clo.jld")
JLD2.save_object("models/scaler_clo.jld", scaler_clo)
println("Saving: scaler_nclo.jld")
JLD2.save_object("models/scaler_nclo.jld", scaler_nclo)
println()

p = Plots.plot(p1, p2, layout=(2, 1))
savefig(p, "images/rr_training_accuracy.png")
