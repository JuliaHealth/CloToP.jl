# Clozapine Toxicity Predictor

[![DOI](images/zenodo.11048224.png)](https://doi.org/10.5281/zenodo.11048224)

This [Julia](https://julialang.org/) toolbox allows predicting [clozapine](https://en.wikipedia.org/wiki/Clozapine) (CLO) and [norclozapine](https://en.wikipedia.org/wiki/Desmethylclozapine) (NCLO) blood concentrations.

Individual recommended safe dose range can also be predicted:

![](images/dose-level.png)

## Performance

The models are actively developed and we expect their accuracy to improve.

### Classifier

Current classifier accuracy (train-test split 70:30):

    Classifier training accuracy:
        log_loss: 0.04
        AUC: 1.0
        misclassification rate: 0.01
        accuracy: 0.99
    confusion matrix:
        sensitivity (TPR): 0.95
        specificity (TNR): 1.0
                         group
                      norm   high   
                    ┌──────┬──────┐
               norm │   71 │    1 │
    prediction      ├──────┼──────┤
               high │    0 │   20 │
                    └──────┴──────┘
             
    Classifier testing accuracy:
        log_loss: 0.48
        AUC: 0.91
        misclassification rate: 0.12
        accuracy: 0.88
    confusion matrix:
        sensitivity (TPR): 0.7
        specificity (TNR): 0.93
                         group
                      norm   high   
                    ┌──────┬──────┐
               norm │   28 │    3 │
    prediction      ├──────┼──────┤
               high │    2 │    7 │
                    └──────┴──────┘

Final model accuracy:

    Classifier accuracy:
        log_loss: 0.04
        AUC: 1.0
        misclassification rate: 0.02
        accuracy: 0.98
    confusion matrix:
        sensitivity (TPR): 0.97
        specificity (TNR): 0.98
                         group
                      norm   high   
                    ┌──────┬──────┐
               norm │   99 │    1 │
    prediction      ├──────┼──────┤
               high │    2 │   30 │
                    └──────┴──────┘

### Regressor

Current regressor model accuracy (train-test split 70:30):

    Predicting: CLOZAPINE
    Regressor training accuracy
        R²: 0.95
        RMSE: 72.99
    Regressor testing accuracy
        R²: 0.71
        RMSE: 198.4

    Predicting: NORCLOZAPINE
    Regressor training accuracy
        R²: 0.95
        RMSE: 39.51
    Regressor testing accuracy
        R²: 0.81
        RMSE: 71.68

Final model accuracy:

    Predicting: CLOZAPINE
    Regressor accuracy
        R²: 0.96
        RMSE: 72.83
    
    Predicting: NORCLOZAPINE
    Regressor accuracy
        R²: 0.96
        RMSE: 32.38

![](images/rr_training_accuracy.png)

### Testing

Current model accuracy:

    Regressor:
    Subject ID: 1   CLO level: 806.4    prediction: 598.7   RMSE: 207.7
    Subject ID: 1   NCLO level: 317.7   prediction: 296.7   RMSE: 21.0
    
    Subject ID: 2   CLO level: 487.1    prediction: 379.4   RMSE: 107.7
    Subject ID: 2   NCLO level: 322.3   prediction: 238.4   RMSE: 83.9
    
    Subject ID: 3   CLO level: 115.6    prediction: 272.2   RMSE: 156.6
    Subject ID: 3   NCLO level: 148.2   prediction: 183.6   RMSE: 35.4
    
    Regressor accuracy:
    Predicting: CLOZAPINE
        R²: 0.6684
        RMSE:   162.5445
    Predicting: NORCLOZAPINE
        R²: 0.5564
        RMSE:   53.9548

    Classifier:
    Subject ID: 1   group: NORM     prediction: HIGH, prob = 0.98   adj. prediction: HIGH, prob = 1.0
    Subject ID: 2   group: NORM     prediction: NORM, prob = 0.99   adj. prediction: NORM, prob = 0.69
    Subject ID: 3   group: HIGH     prediction: NORM, prob = 0.85   adj. prediction: NORM, prob = 0.55
    
    Classifier accuracy:
        log_loss: 1.2104
        AUC: 0.5
        misclassification rate: 0.33
        accuracy: 0.67
    confusion matrix:
        sensitivity (TP): 0.0
        specificity (TP): 1.0
                         group
                      norm   high   
                    ┌──────┬──────┐
               norm │    2 │    1 │
    prediction      ├──────┼──────┤
               high │    0 │    0 │
                    └──────┴──────┘
             
    Adjusted classifier accuracy:
        log_loss: 0.479
        AUC: 1.0
        misclassification rate: 0.33
        accuracy: 0.67
    confusion matrix:
        sensitivity (TP): 0.0
        specificity (TP): 1.0
                         group
                      norm   high   
                    ┌──────┬──────┐
               norm │    2 │    1 │
    prediction      ├──────┼──────┤
               high │    0 │    0 │
                    └──────┴──────┘

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