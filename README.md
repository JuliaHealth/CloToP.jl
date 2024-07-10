# Clozapine Toxicity Predictor

[![DOI](images/zenodo.11048224.png)](https://doi.org/10.5281/zenodo.11048224)

This [Julia](https://julialang.org/) toolbox allows predicting [clozapine](https://en.wikipedia.org/wiki/Clozapine) (CLO) and [norclozapine](https://en.wikipedia.org/wiki/Desmethylclozapine) (NCLO) blood concentrations.

Individual recommended safe dose range can also be predicted:

![](images/dose-level.png)

## Performance

The models are actively developed and we expect their accuracy to improve.

### Training

```
[ Info: Loading packages
       CSV 0.10.14
DataFrames 1.6.1
      JLD2 0.4.48
       MLJ 0.20.5
   MLJFlux 0.4.0
      Flux 0.14.15
     NNlib 0.4.0
     Plots 1.40.4
 StatsBase 0.34.3

[ Info: Loading data
Loading: clozapine_train.csv

Number of entries: 69
Number of features: 11

[ Info: Preprocessing
Standardizing
Splitting: 70:30

[ Info: Creating regressor model: clozapine
Regressor accuracy: training
  R²: 1.0
  RMSE: 13.98
Regressor accuracy: validating
  R²: 0.92
  RMSE: 85.23
[ Info: Training final model
Regressor accuracy:
  R²: 1.0
  RMSE: 18.42

[ Info: Creating regressor model: norclozapine
Regressor accuracy: training
  R²: 0.99
  RMSE: 19.32
Regressor accuracy: validating
  R²: 0.8
  RMSE: 62.76
[ Info: Training final model
Regressor accuracy:
  R²: 0.98
  RMSE: 21.04

[ Info: Classifying into groups
Classification based on predicted clozapine level:
Confusion matrix:
  misclassification rate: 0.01
  accuracy: 0.99
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │  108 │    1 │
prediction      ├──────┼──────┤
           high │    0 │   29 │
                └──────┴──────┘
         
Classification adjusted for predicted norclozapine level:
Confusion matrix:
  misclassification rate: 0.07
  accuracy: 0.93
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │  100 │    1 │
prediction      ├──────┼──────┤
           high │    8 │   29 │
                └──────┴──────┘
         
[ Info: Saving models
Saving: clozapine_regressor_model.jlso
Saving: norclozapine_regressor_model.jlso
Saving: scaler_clo.jld
Saving: scaler_nclo.jld
```

![](images/rr_training_accuracy.png)

### Testing

```
[ Info: Loading packages
       CSV 0.10.14
DataFrames 1.6.1
      JLD2 0.4.48
       MLJ 0.20.0
   MLJFlux 0.5.1
      Flux 0.14.15
     NNlib 0.5.1
     Plots 1.40.4
 StatsBase 0.33.21

[ Info: Loading data
Loading: clozapine_test.csv
Loading: clozapine_regressor_model.jlso
Loading: norclozapine_regressor_model.jlso
Loading: scaler_clo.jld
Loading: scaler_nclo.jld

Number of entries: 3
Number of features: 11

[ Info: Predicting norclozapine level

[ Info: Predicting clozapine level

[ Info: Regressor accuracy
Predicted levels:
Subject ID: 1    CLO level: 806.4   prediction: 641.4   error: -165.0
Subject ID: 1   NCLO level: 317.7   prediction: 306.0   error: -11.7

Subject ID: 2    CLO level: 300.5   prediction: 415.2   error: 114.7
Subject ID: 2   NCLO level: 138.3   prediction: 227.3   error: 89.0

Subject ID: 3    CLO level: 264.5   prediction: 233.3   error: -31.2
Subject ID: 3   NCLO level: 161.1   prediction: 148.5   error: -12.6

Predicting: CLOZAPINE
  R²: 0.77
  RMSE: 117.41
Predicting: NORCLOZAPINE
  R²: 0.57
  RMSE: 52.33

[ Info: Classifying into groups
Classification based on predicted clozapine level:
Confusion matrix:
  misclassification rate: 0.0
  accuracy: 1.0
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │    2 │    0 │
prediction      ├──────┼──────┤
           high │    0 │    1 │
                └──────┴──────┘
         
Classification adjusted for predicted norclozapine level:
Confusion matrix:
  misclassification rate: 0.0
  accuracy: 1.0
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │    2 │    0 │
prediction      ├──────┼──────┤
           high │    0 │    1 │
                └──────┴──────┘
```

![](images/rr_testing_accuracy.png)

## Quickstart

Clone this repository, go to its folder and run:

```sh
julia src/server.jl
```

Next, go to the local website at [http://localhost:8080](http://localhost:8080), enter patient's data and click the "PREDICT" button.

![](images/webpage.png)

(!) Adjusted clozapine level is the classifier prediction modified by predicted concentration.

Toxic clozapine level has been defined as > 550 ng/mL, recommended therapeutic concentration has been defined as > 250 ng/mL [source: [10.1192/bjp.2023.27](https://doi.org/10.1192/bjp.2023.27)].

## How to Cite

If you use this tool, please acknowledge us by citing our [paper](https://zenodo.org/records/11048224).

## Contributors

Below is the list of contributors and their affiliations.

[Adam Wysokiński](mailto:adam.wysokinski@umed.lodz.pl) [![ORCID](images/orcid.png)](https://orcid.org/0000-0002-6159-6579)

[Joanna Dreczka](mailto:jdreczka@csk.umed.pl)

[![Medical University of Lodz](images/umed.png)](https://en.umed.pl)

## License

This software is licensed under [The 2-Clause BSD License](LICENSE).

## Disclaimers

**DISCLAIMER: THIS TOOL HAS THE RESEARCH USE ONLY (RUO) STATUS**

This tool and all associated information, including but not limited to, text, graphics, images and other material contained on this website, has the Research Use Only (RUO) status. It is intended for scientific research only. It must not be used for diagnostic or medical purposes.

**DISCLAIMER: THIS WEBSITE DOES NOT PROVIDE MEDICAL ADVICE**

This tool and all associated information, including but not limited to, text, graphics, images and other material contained on this website are for informational purposes only. No material on this site is intended to be a substitute for professional medical advice, diagnosis or treatment. Always seek the advice of your physician or other qualified health care provider with any questions you may have regarding a medical condition or treatment and before undertaking a new health care regimen, and never disregard professional medical advice or delay in seeking it because of something you have read on this website.