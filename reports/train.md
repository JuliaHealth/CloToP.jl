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

Number of entries: 119
Number of features: 11

Standardizing
Splitting: 70:30

Initial RMSE: 91.3994
Final RMSE: 95.3455
Model parameters:
  n_hidden: 84
  dropout: 0.26
  η: 0.01
  epochs: 2900
  batch_size: 2
  λ: 0.1
  α: 0.69

Regressor accuracy: training
  R²: 1.0
  RMSE: 15.59
Regressor accuracy: validating
  R²: 0.88
  RMSE: 93.29
Regressor accuracy:
  R²: 1.0
  RMSE: 14.76

Initial RMSE: 103.397
Final RMSE: 82.8903
Model parameters:
  n_hidden: 118
  dropout: 0.64
  η: 0.01
  epochs: 700
  batch_size: 2
  λ: 0.1
  α: 0.0

Regressor accuracy: training
  R²: 0.99
  RMSE: 16.35
Regressor accuracy: validating
  R²: 0.57
  RMSE: 95.78
Regressor accuracy:
  R²: 0.98
  RMSE: 20.38

Classification based on predicted clozapine level:
Confusion matrix:
  misclassification rate: 0.01
  accuracy: 0.99
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │  194 │    3 │
prediction      ├──────┼──────┤
           high │    0 │   41 │
                └──────┴──────┘
         
Classification adjusted for predicted norclozapine level:
Confusion matrix:
  misclassification rate: 0.05
  accuracy: 0.95
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │  184 │    1 │
prediction      ├──────┼──────┤
           high │   10 │   43 │
                └──────┴──────┘
         
Saving: clozapine_regressor_model.jlso
Saving: norclozapine_regressor_model.jlso
Saving: scaler_clo.jld
Saving: scaler_nclo.jld

