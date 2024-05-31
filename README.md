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
        log_loss: 0.08
        AUC: 1.0
        misclassification rate: 0.02
        accuracy: 0.98
    confusion matrix:
        sensitivity (TPR): 0.95
        specificity (TNR): 1.0
                         group
                      norm   high   
                    ┌──────┬──────┐
               norm │   26 │    1 │
    prediction      ├──────┼──────┤
               high │    0 │   19 │
                    └──────┴──────┘
             
    Classifier testing accuracy:
        log_loss: 1.03
        AUC: 0.65
        misclassification rate: 0.35
        accuracy: 0.65
    confusion matrix:
        sensitivity (TPR): 0.82
        specificity (TNR): 0.44
                         group
                      norm   high   
                    ┌──────┬──────┐
               norm │    4 │    2 │
    prediction      ├──────┼──────┤
               high │    5 │    9 │
                    └──────┴──────┘

Final model accuracy:

    Classifier accuracy:
        log_loss: 0.06
        AUC: 1.0
        misclassification rate: 0.0
        accuracy: 1.0
    confusion matrix:
        sensitivity (TPR): 1.0
        specificity (TNR): 1.0
                         group
                      norm   high   
                    ┌──────┬──────┐
               norm │  167 │    0 │
    prediction      ├──────┼──────┤
               high │    0 │   31 │
                    └──────┴──────┘

### Regressor

Current regressor model accuracy (train-test split 70:30):

    Regressor training accuracy
        R²: 0.28
        RMSE: 261.56
    Regressor testing accuracy
        R²: 0.32
        RMSE: 237.95
    Predicting: NORCLOZAPINE
    Regressor training accuracy
        R²: 0.96
        RMSE: 26.99
    Regressor testing accuracy
        R²: -0.74
        RMSE: 244.64

Final model accuracy:

    Predicting: CLOZAPINE
    Regressor accuracy
        R²: 0.44
        RMSE: 226.44
    Predicting: NORCLOZAPINE
    Regressor accuracy
        R²: 0.94
        RMSE: 37.01

![](images/rr_training_accuracy.png)

### Testing

Current model accuracy:

    Regressor:
    Subject ID: 1   CLO level: 270.9    prediction: 554.7   RMSE: 283.8
    Subject ID: 1   NCLO level: 388.5   prediction: 226.6   RMSE: 161.9
    
    Subject ID: 2   CLO level: 278.1    prediction: 723.7   RMSE: 445.6
    Subject ID: 2   NCLO level: 145.9   prediction: 440.6   RMSE: 294.7
    
    Subject ID: 3   CLO level: 603.6    prediction: 571.3   RMSE: 32.3
    Subject ID: 3   NCLO level: 325.0   prediction: 494.5   RMSE: 169.5
    
    Regressor accuracy:
    Predicting: CLOZAPINE
        R²: -2.8785
        RMSE:   305.5842
    Predicting: NORCLOZAPINE
        R²: -3.4793
        RMSE:   217.4013

    Classifier:
    Subject ID: 1   group: NORM     prediction: HIGH, prob = 0.98   adj. prediction: HIGH, prob = 1.0
    Subject ID: 2   group: NORM     prediction: NORM, prob = 0.99   adj. prediction: NORM, prob = 0.69
    Subject ID: 3   group: HIGH     prediction: NORM, prob = 0.85   adj. prediction: NORM, prob = 0.55
    
    Classifier accuracy:
        log_loss: 1.9722
        AUC: 0.5
        misclassification rate: 0.67
        accuracy: 0.33
    confusion matrix:
        sensitivity (TP): 0.0
        specificity (TP): 0.5
                         group
                      norm   high   
                    ┌──────┬──────┐
               norm │    1 │    1 │
    prediction      ├──────┼──────┤
               high │    1 │    0 │
                    └──────┴──────┘
             
    Adjusted classifier accuracy:
        misclassification rate: 0.67
        accuracy: 0.33
    confusion matrix:
        sensitivity (TP): 0.0
        specificity (TP): 0.5
                         group
                      norm   high   
                    ┌──────┬──────┐
               norm │    1 │    1 │
    prediction      ├──────┼──────┤
               high │    1 │    0 │
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