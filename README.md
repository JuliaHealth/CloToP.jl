# Clozapine Toxicity Predictor

[![DOI](images/zenodo.11048224.png)](https://doi.org/10.5281/zenodo.11048224)

This [Julia](https://julialang.org/) toolbox allows predicting [clozapine](https://en.wikipedia.org/wiki/Clozapine) (CLO) and [norclozapine](https://en.wikipedia.org/wiki/Desmethylclozapine) (NCLO) blood concentrations using RandomForestClassifier and RandomForestRegressor models.

Individual recommended safe dose range can also be predicted:

![](images/dose-level.png)

## Performance

The models are actively developed and we expect their accuracy to improve.

### Classifier

Current classifier model training accuracy:

    Classifier prediction accuracy:
        log_loss: 0.0576
        AUC: 1.0
        misclassification rate: 0.0
        accuracy: 1.0
    confusion matrix:
        sensitivity (TPR): 1.0
        specificity (TNR): 1.0
                         group
                      norm   high   
                    ┌──────┬──────┐
               norm │  153 │    0 │
    prediction      ├──────┼──────┤
               high │    0 │   30 │
                    └──────┴──────┘

Current classifier model testing accuracy:

    Subject ID: 1   group: HIGH     prediction: HIGH, prob = 0.59   adj. prediction: HIGH, prob = 0.69
    Subject ID: 2   group: NORM     prediction: NORM, prob = 0.6    adj. prediction: NORM, prob = 0.5
    Subject ID: 3   group: HIGH     prediction: HIGH, prob = 0.66   adj. prediction: HIGH, prob = 0.96
    Subject ID: 4   group: NORM     prediction: NORM, prob = 0.67   adj. prediction: NORM, prob = 0.97

    Classifier prediction accuracy:
        log_loss: 0.5029
        AUC: 1.0
        misclassification rate: 0.33
        accuracy: 0.67
    confusion matrix:
        sensitivity (TP): 1.0
        specificity (TP): 0.5
                         group
                      norm   high   
                    ┌──────┬──────┐
               norm │    1 │    0 │
    prediction      ├──────┼──────┤
               high │    1 │    1 │
                    └──────┴──────┘
             
    Adjusted classifier prediction accuracy:
        misclassification rate: 0.33
        accuracy: 0.67
    confusion matrix:
        sensitivity (TP): 1.0
        specificity (TP): 0.5
                         group
                      norm   high   
                    ┌──────┬──────┐
               norm │    1 │    0 │
    prediction      ├──────┼──────┤
               high │    1 │    1 │
                    └──────┴──────┘

### Regressor

Current regressor model training accuracy:

    Regressor prediction accuracy:
    Predicting: CLOZAPINE
        R²: 0.9643
        RMSE: 60.0249
    Predicting: NORCLOZAPINE
        R²: 0.9686
        RMSE: 27.5317

![](images/rr_train_accuracy.png)

Current regressor model testing accuracy:

    Subject ID: 1   CLO level: 270.9    prediction: 653.1   RMSE: 382.2
    Subject ID: 1   NCLO level: 388.5   prediction: 220.1   RMSE: 168.4
    
    Subject ID: 2   CLO level: 603.6    prediction: 688.4   RMSE: 84.8
    Subject ID: 2   NCLO level: 325.0   prediction: 393.7   RMSE: 68.7
    
    Subject ID: 3   CLO level: 375.2    prediction: 499.4   RMSE: 124.2
    Subject ID: 3   NCLO level: 160.7   prediction: 234.8   RMSE: 74.1

    Regressor prediction accuracy:
    Predicting: CLOZAPINE
        R²: -1.913
        RMSE:   237.1311
    Predicting: NORCLOZAPINE
        R²: -0.3954
        RMSE:   113.3859

![](images/rr_test_accuracy.png)

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