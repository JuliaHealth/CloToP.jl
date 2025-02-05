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

Number of entries: 110
Number of features: 11

Standardizing
Splitting: 70:30

Regressor accuracy: training
  R²: 1.0
  RMSE: 15.35
Regressor accuracy: validating
  R²: 0.87
  RMSE: 116.83
Regressor accuracy:
  R²: 1.0
  RMSE: 17.14

Regressor accuracy: training
  R²: 0.98
  RMSE: 20.03
Regressor accuracy: validating
  R²: 0.87
  RMSE: 53.48
Regressor accuracy:
  R²: 0.99
  RMSE: 17.32

Classification based on predicted clozapine level:
Confusion matrix:
  misclassification rate: 0.01
  accuracy: 0.99
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │  179 │    2 │
prediction      ├──────┼──────┤
           high │    1 │   38 │
                └──────┴──────┘

Classification adjusted for predicted norclozapine level:
Confusion matrix:
  misclassification rate: 0.05
  accuracy: 0.95
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │  172 │    2 │
prediction      ├──────┼──────┤
           high │    8 │   38 │
                └──────┴──────┘

Saving: clozapine_regressor_model.jlso
Saving: norclozapine_regressor_model.jlso
Saving: scaler_clo.jld
Saving: scaler_nclo.jld
