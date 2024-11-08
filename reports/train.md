       CSV 0.10.14
DataFrames 1.7.0
      JLD2 0.5.8
      Flux 0.14.25
       MLJ 0.20.7
   MLJFlux 0.6.0
     NNlib 0.9.24
Optimisers 0.3.4
     Plots 1.40.8
 StatsBase 0.34.3

Loading: clozapine_train.csv

Number of entries: 87
Number of features: 11

Standardizing
Splitting: 70:30

Initial RMSE: 79.7404
Final RMSE: 74.1407
Model parameters:
  n_hidden: 170
  dropout: 0.07
  η: 0.01
  epochs: 5600
  batch_size: 2
  λ: 0.1
  α: 0.0

Regressor accuracy: training
  R²: 1.0
  RMSE: 16.03
Regressor accuracy: validating
  R²: 0.9
  RMSE: 89.49
Regressor accuracy:
  R²: 1.0
  RMSE: 17.39

Initial RMSE: 61.3673
Final RMSE: 61.4346
Model parameters:
  n_hidden: 64
  dropout: 0.16
  η: 0.01
  epochs: 1000
  batch_size: 2
  λ: 0.1
  α: 0.0

Regressor accuracy: training
  R²: 0.98
  RMSE: 22.76
Regressor accuracy: validating
  R²: 0.85
  RMSE: 54.89
Regressor accuracy:
  R²: 0.98
  RMSE: 20.37

Classification based on predicted clozapine level:
Confusion matrix:
  misclassification rate: 0.02
  accuracy: 0.98
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │  143 │    3 │
prediction      ├──────┼──────┤
           high │    0 │   28 │
                └──────┴──────┘
         
Classification adjusted for predicted norclozapine level:
Confusion matrix:
  misclassification rate: 0.06
  accuracy: 0.94
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │  134 │    2 │
prediction      ├──────┼──────┤
           high │    9 │   29 │
                └──────┴──────┘
         
Saving: clozapine_regressor_model.jlso
Saving: norclozapine_regressor_model.jlso
Saving: scaler_clo.jld
Saving: scaler_nclo.jld

