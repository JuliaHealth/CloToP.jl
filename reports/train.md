       CSV 0.10.15
DataFrames 1.7.0
      JLD2 0.5.8
      Flux 0.14.25
       MLJ 0.20.7
   MLJFlux 0.6.0
     NNlib 0.6.0
Optimisers 0.3.4
     Plots 1.40.9
 StatsBase 0.34.3

Loading: clozapine_train.csv

Number of entries: 103
Number of features: 11

Standardizing
Splitting: 70:30

Initial RMSE: 106.2541
Final RMSE: 113.4594
Model parameters:
  n_hidden: 54
  dropout: 0.14
  η: 0.01
  epochs: 9900
  batch_size: 2
  λ: 0.1
  α: 0.69

Regressor accuracy: training
  R²: 1.0
  RMSE: 12.29
Regressor accuracy: validating
  R²: 0.89
  RMSE: 108.63
Regressor accuracy:
  R²: 1.0
  RMSE: 13.39

Initial RMSE: 81.9708
Final RMSE: 80.3017
Model parameters:
  n_hidden: 202
  dropout: 0.05
  η: 0.01
  epochs: 1000
  batch_size: 2
  λ: 0.1
  α: 0.0

Regressor accuracy: training
  R²: 0.98
  RMSE: 18.17
Regressor accuracy: validating
  R²: 0.79
  RMSE: 77.61
Regressor accuracy:
  R²: 0.99
  RMSE: 17.22

Classification based on predicted clozapine level:
Confusion matrix:
  misclassification rate: 0.01
  accuracy: 0.99
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │  169 │    3 │
prediction      ├──────┼──────┤
           high │    0 │   34 │
                └──────┴──────┘
         
Classification adjusted for predicted norclozapine level:
Confusion matrix:
  misclassification rate: 0.05
  accuracy: 0.95
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │  160 │    2 │
prediction      ├──────┼──────┤
           high │    9 │   35 │
                └──────┴──────┘
         
Saving: clozapine_regressor_model.jlso
Saving: norclozapine_regressor_model.jlso
Saving: scaler_clo.jld
Saving: scaler_nclo.jld

