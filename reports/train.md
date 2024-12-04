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

Number of entries: 94
Number of features: 11

Standardizing
Splitting: 70:30

Regressor accuracy: training
  R²: 1.0
  RMSE: 13.21
Regressor accuracy: validating
  R²: 0.9
  RMSE: 96.61
Regressor accuracy:
  R²: 1.0
  RMSE: 13.37

Regressor accuracy: training
  R²: 0.99
  RMSE: 16.72
Regressor accuracy: validating
  R²: 0.69
  RMSE: 83.42
Regressor accuracy:
  R²: 0.99
  RMSE: 16.75

Classification based on predicted clozapine level:
Confusion matrix:
  misclassification rate: 0.02
  accuracy: 0.98
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │  153 │    3 │
prediction      ├──────┼──────┤
           high │    0 │   32 │
                └──────┴──────┘
         
Classification adjusted for predicted norclozapine level:
Confusion matrix:
  misclassification rate: 0.06
  accuracy: 0.94
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │  144 │    2 │
prediction      ├──────┼──────┤
           high │    9 │   33 │
                └──────┴──────┘
         
Saving: clozapine_regressor_model.jlso
Saving: norclozapine_regressor_model.jlso
Saving: scaler_clo.jld
Saving: scaler_nclo.jld

