       CSV 0.10.15
DataFrames 1.7.0
      JLD2 0.5.8
       MLJ 0.20.7
   MLJFlux 0.6.0
      Flux 0.14.25
     NNlib 0.9.24
Optimisers 0.3.4
     Plots 1.40.9
 StatsBase 0.34.3

Loading: clozapine_test.csv
Loading: clozapine_regressor_model.jlso
Loading: norclozapine_regressor_model.jlso
Loading: scaler_clo.jld
Loading: scaler_nclo.jld

Number of entries: 3
Number of features: 11



Predicted levels:
Subject ID: 1 	  CLO level: 806.4 	 prediction: 467.6 	 error: -338.8
Subject ID: 1 	 NCLO level: 317.7 	 prediction: 242.2 	 error: -75.5

Subject ID: 2 	  CLO level: 300.5 	 prediction: 390.5 	 error: 90.0
Subject ID: 2 	 NCLO level: 138.3 	 prediction: 232.6 	 error: 94.3

Subject ID: 3 	  CLO level: 264.5 	 prediction: 265.1 	 error: 0.6
Subject ID: 3 	 NCLO level: 161.1 	 prediction: 152.7 	 error: -8.4

Predicting: CLOZAPINE
  R²:	0.33
  RMSE:	202.39
Predicting: NORCLOZAPINE
  R²:	0.23
  RMSE:	69.91

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
         
