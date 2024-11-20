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

Number of entries: 88
Number of features: 11

Standardizing
Splitting: 70:30

Regressor accuracy: training
  R²: 1.0
  RMSE: 14.94
Regressor accuracy: validating
  R²: 0.86
  RMSE: 105.1
Regressor accuracy:
  R²: 1.0
  RMSE: 18.82

Regressor accuracy: training
  R²: 0.98
  RMSE: 19.87
Regressor accuracy: validating
  R²: 0.74
  RMSE: 72.4
Regressor accuracy:
  R²: 0.98
  RMSE: 23.14

Classification based on predicted clozapine level:
Confusion matrix:
  misclassification rate: 0.03
  accuracy: 0.97
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │  141 │    2 │
prediction      ├──────┼──────┤
           high │    3 │   30 │
                └──────┴──────┘
         
Classification adjusted for predicted norclozapine level:
Confusion matrix:
  misclassification rate: 0.08
  accuracy: 0.92
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │  131 │    1 │
prediction      ├──────┼──────┤
           high │   13 │   31 │
                └──────┴──────┘
         
Saving: clozapine_regressor_model.jlso
Saving: norclozapine_regressor_model.jlso
Saving: scaler_clo.jld
Saving: scaler_nclo.jld

