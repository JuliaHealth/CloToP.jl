using Plots
# using CUDA

train_idx, test_idx = partition(eachindex(y2), 0.8, shuffle=true)

nnc = @MLJ.load NeuralNetworkClassifier pkg=MLJFlux
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
            alpha = 0.0)
mach = machine(model, x[train_idx, :], y2[train_idx], scitype_check_level=0)
MLJ.fit!(mach, force=true, verbosity=0)
yhat = MLJ.predict(mach, x[test_idx, :])
cm = confusion_matrix(mode.(yhat), y2[test_idx])

nnr = @MLJ.load NeuralNetworkRegressor pkg=MLJFlux
model_clo = nnr(builder =  MLJFlux.Short(n_hidden=200,
                                          dropout=0.1, 
                                          σ = NNlib.relu), 
                optimiser = Adam(0.001, (0.9, 0.999), 1.0e-8, IdDict{Any, Any}()), 
                loss = Flux.Losses.mse, 
                epochs = 1000, 
                batch_size = 10, 
                lambda = 0.0, 
                alpha = 0.2) 
mach = machine(model_clo, x[train_idx, :], y1[train_idx], scitype_check_level=0)
MLJ.fit!(mach, force=true, verbosity=1)
yhat = MLJ.predict(mach, x[test_idx, :])
plot(y1[test_idx])
plot!(yhat)

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
yhat = MLJ.predict(mach_nclo, x[test_idx, :])
plot(y3[test_idx])
plot!(yhat)
