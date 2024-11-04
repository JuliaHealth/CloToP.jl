       CSV 0.10.15
DataFrames 1.7.0
      JLD2 0.4.53
       MLJ 0.20.7
   MLJFlux 0.6.0
      Flux 0.14.23
     NNlib 0.9.24
Optimisers 0.3.3
     Plots 1.40.7
 StatsBase 0.34.3

Loading: clozapine_test.csv
Loading: clozapine_regressor_model.jlso
Loading: norclozapine_regressor_model.jlso
Loading: scaler_clo.jld
Loading: scaler_nclo.jld

Number of entries: 3
Number of features: 11



Predicted levels:
Subject ID: 1 	  CLO level: 806.4 	 prediction: 410.5 	 error: -395.9
Subject ID: 1 	 NCLO level: 317.7 	 prediction: 233.0 	 error: -84.7

Subject ID: 2 	  CLO level: 300.5 	 prediction: 417.7 	 error: 117.2
Subject ID: 2 	 NCLO level: 138.3 	 prediction: 236.8 	 error: 98.5

Subject ID: 3 	  CLO level: 264.5 	 prediction: 280.5 	 error: 16.0
Subject ID: 3 	 NCLO level: 161.1 	 prediction: 147.5 	 error: -13.6

Predicting: CLOZAPINE
  R²:	0.07
  RMSE:	238.56
Predicting: NORCLOZAPINE
  R²:	0.11
  RMSE:	75.41

Classification based on predicted clozapine level:
Confusion matrix:
  misclassification rate: 0.33
  accuracy: 0.67
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │    2 │    1 │
prediction      ├──────┼──────┤
           high │    0 │    0 │
                └──────┴──────┘
         
Classification adjusted for predicted norclozapine level:
Confusion matrix:
  misclassification rate: 0.33
  accuracy: 0.67
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │    2 │    1 │
prediction      ├──────┼──────┤
           high │    0 │    0 │
                └──────┴──────┘
         
