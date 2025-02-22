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

Number of entries: 139
Number of features: 11

Standardizing
Splitting: 70:30

Regressor accuracy: training
  R²: 0.98
  RMSE: 40.35
Regressor accuracy: validating
  R²: 0.87
  RMSE: 108.44
Regressor accuracy:
  R²: 0.97
  RMSE: 50.37

Regressor accuracy: training
  R²: 0.78
  RMSE: 94.66
Regressor accuracy: validating
  R²: 0.73
  RMSE: 70.17
Regressor accuracy:
  R²: 0.79
  RMSE: 85.14

Classification based on predicted clozapine level:
Confusion matrix:
  misclassification rate: 0.03
  accuracy: 0.97
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │  223 │    3 │
prediction      ├──────┼──────┤
           high │    6 │   46 │
                └──────┴──────┘
         
Classification adjusted for predicted norclozapine level:
Confusion matrix:
  misclassification rate: 0.06
  accuracy: 0.94
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │  215 │    3 │
prediction      ├──────┼──────┤
           high │   14 │   46 │
                └──────┴──────┘
         
Saving: clozapine_regressor_model.jlso
Saving: norclozapine_regressor_model.jlso
Saving: scaler_clo.jld
Saving: scaler_nclo.jld

