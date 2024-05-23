# Clozapine Toxicity Predictor

[![DOI](images/zenodo.11048224.png)](https://doi.org/10.5281/zenodo.11048224)

This [Julia](https://julialang.org/) toolbox allows predicting [clozapine](https://en.wikipedia.org/wiki/Clozapine) (CLO) and [norclozapine](https://en.wikipedia.org/wiki/Desmethylclozapine) (NCLO) blood concentrations using RandomForestClassifier and RandomForestRegressor models.

Individual recommended safe dose range can also be predicted:

![](images/dose-level.png)

## Performance

The models are actively developed and we expect their accuracy to improve.

### Classifier

Current classifier accuracy (train-test split 80:20):

    Classifier training accuracy:
        log_loss: 0.0608
        AUC: 1.0
        misclassification rate: 0.0
        accuracy: 1.0
    confusion matrix:
        sensitivity (TPR): 1.0
        specificity (TNR): 1.0
                         group
                      norm   high   
                    ┌──────┬──────┐
               norm │  126 │    0 │
    prediction      ├──────┼──────┤
               high │    0 │   20 │
                    └──────┴──────┘
             
    Classifier testing accuracy:
        log_loss: 0.0738
        AUC: 1.0
        misclassification rate: 0.0
        accuracy: 1.0
    confusion matrix:
        sensitivity (TPR): 1.0
        specificity (TNR): 1.0
                         group
                      norm   high   
                    ┌──────┬──────┐
               norm │   27 │    0 │
    prediction      ├──────┼──────┤
               high │    0 │   10 │
                    └──────┴──────┘

Final model accuracy:

    Classifier accuracy:
        log_loss: 0.0562
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

### Regressor

Current regressor model accuracy (train-test split 80:20):

    Predicting: CLOZAPINE
    Regressor training accuracy
        R²: 0.9612
        RMSE: 64.1466
    Regressor testing accuracy
        R²: 0.9368
        RMSE: 68.8626
    Predicting: NORCLOZAPINE
    Regressor training accuracy
        R²: 0.96
        RMSE: 31.065
    Regressor testing accuracy
        R²: 0.937
        RMSE: 40.4948

Final model accuracy:

    Predicting: CLOZAPINE
    Regressor accuracy
        R²: 0.9671
        RMSE: 57.6821
    Predicting: NORCLOZAPINE
    Regressor accuracy
        R²: 0.9705
        RMSE: 26.9172

![](images/rr_training_accuracy.png)

### Testing

Current model accuracy:

    Regressor:
    Subject ID: 1   CLO level: 278.1    prediction: 768.2   RMSE: 490.1
    Subject ID: 1   NCLO level: 145.9   prediction: 347.5   RMSE: 201.6
    
    Subject ID: 2   CLO level: 603.6    prediction: 737.4   RMSE: 133.8
    Subject ID: 2   NCLO level: 325.0   prediction: 422.8   RMSE: 97.8
    
    Subject ID: 3   CLO level: 375.2    prediction: 453.8   RMSE: 78.6
    Subject ID: 3   NCLO level: 160.7   prediction: 241.3   RMSE: 80.6

    Regressor accuracy:
    Predicting: CLOZAPINE
        R²: -3.7321
        RMSE:   296.8043
    Predicting: NORCLOZAPINE
        R²: -1.8691
        RMSE:   137.4818

    Classifier:
    Subject ID: 1   group: NORM     prediction: HIGH, prob = 0.61   adj. prediction: HIGH, prob = 0.71
    Subject ID: 2   group: HIGH     prediction: HIGH, prob = 0.75   adj. prediction: HIGH, prob = 1.0
    Subject ID: 3   group: NORM     prediction: NORM, prob = 0.69   adj. prediction: NORM, prob = 0.99
        
    Classifier accuracy:
        log_loss: 0.5345
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
             
    Adjusted classifier accuracy:
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