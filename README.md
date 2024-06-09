# Clozapine Toxicity Predictor

[![DOI](images/zenodo.11048224.png)](https://doi.org/10.5281/zenodo.11048224)

This [Julia](https://julialang.org/) toolbox allows predicting [clozapine](https://en.wikipedia.org/wiki/Clozapine) (CLO) and [norclozapine](https://en.wikipedia.org/wiki/Desmethylclozapine) (NCLO) blood concentrations.

Individual recommended safe dose range can also be predicted:

![](images/dose-level.png)

## Performance

The models are actively developed and we expect their accuracy to improve.

### Training

```
[ Info: Creating regressor model: clozapine
Regressor accuracy: training
  R²: 0.99
  RMSE: 35.32
Regressor accuracy: validating
  R²: 0.9
  RMSE: 96.74
[ Info: Training final model
Regressor accuracy:
  R²: 0.99
  RMSE: 39.19

[ Info: Creating regressor model: norclozapine
Regressor accuracy: training
  R²: 0.98
  RMSE: 26.39
Regressor accuracy: validating
  R²: 0.73
  RMSE: 72.88
[ Info: Training final model
Regressor accuracy:
  R²: 0.98
  RMSE: 21.78

[ Info: Classifying into groups
Confusion matrix:
  misclassification rate: 0.03
  accuracy: 0.97
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │  102 │    1 │
prediction      ├──────┼──────┤
           high │    3 │   30 │
                └──────┴──────┘
```

![](images/rr_training_accuracy.png)

### Testing

```
Regressor:
Subject ID: 1   CLO level: 240.5    prediction: 74.7    RMSE: 165.8
Subject ID: 1   NCLO level: 90.7    prediction: 110.3   RMSE: 19.6

Subject ID: 2   CLO level: 292.4    prediction: 793.8   RMSE: 501.4
Subject ID: 2   NCLO level: 283.8   prediction: 343.0   RMSE: 59.2

Subject ID: 3   CLO level: 390.5    prediction: 593.8   RMSE: 203.3
Subject ID: 3   NCLO level: 162.5   prediction: 250.1   RMSE: 87.6

Subject ID: 4   CLO level: 586.1    prediction: 771.9   RMSE: 185.8
Subject ID: 4   NCLO level: 189.4   prediction: 321.7   RMSE: 132.3

Regressor accuracy:
Predicting: CLOZAPINE
  R²: -4.09
  RMSE:   297.8
Predicting: NORCLOZAPINE
  R²: -0.52
  RMSE:   85.24

Classifier:
Subject ID: 1   group: NORM     prediction: NORM, prob = 1.0    adj. prediction: NORM, prob = 1.0
Subject ID: 2   group: NORM     prediction: HIGH, prob = 0.69   adj. prediction: NORM, prob = 0.61
Subject ID: 3   group: NORM     prediction: HIGH, prob = 0.63   adj. prediction: NORM, prob = 0.67
Subject ID: 4   group: HIGH     prediction: HIGH, prob = 0.92   adj. prediction: HIGH, prob = 0.62

Classifier accuracy:
  log_loss: 0.56
  AUC: 1.0
  misclassification rate: 0.5
  accuracy: 0.5
Confusion matrix:
  sensitivity (TP): 1.0
  specificity (TP): 0.33
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │    1 │    0 │
prediction      ├──────┼──────┤
           high │    2 │    1 │
                └──────┴──────┘
         
Adjusted classifier accuracy:
  log_loss: 0.35
  AUC: 1.0
  misclassification rate: 0.0
  accuracy: 1.0
Confusion matrix:
  sensitivity (TP): 1.0
  specificity (TP): 1.0
                     group
                  norm   high   
                ┌──────┬──────┐
           norm │    3 │    0 │
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

Toxic clozapine level has been defined as > 550 ng/mL, recommended therapeutic concentration has been defined as > 220 ng/mL [source: [10.1192/bjp.2023.27](https://doi.org/10.1192/bjp.2023.27)].

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