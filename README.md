# Clozapine Toxicity Predictor

Current status of the regressor model training accuracy:

![](rr_train_accuracy.png)

Current status of the regressor model testing accuracy:

![](rr_test_accuracy.png)

Individual recommended safe dose range is also predicted:

![](dose-level.png)

## Quickstart

```sh
julia src/server.jl
```

Next, go to the local website at [http://localhost:8080](http://localhost:8080), enter patients data and click the "PREDICT" button.

![](webpage.png)

## How to Cite

If you use this package please acknowledge us by citing our [paper](https://neuroanalyzer.org#how-to-cite).

## Contributors

Below is the list of contributors and their affiliations.

[Adam Wysokiński](mailto:adam.wysokinski@neuroanalyzer.org) [![ORCID](images/orcid.png)](https://orcid.org/0000-0002-6159-6579)

[Joanna Dreczka](mailto:jdreczka@csk.umed.pl)

[![Medical University of Lodz](images/umed.png)](https://en.umed.pl)

# License

This software is licensed under [The 2-Clause BSD License](LICENSE).

## Disclaimer

**DISCLAIMER: THIS WEBSITE DOES NOT PROVIDE MEDICAL ADVICE**

The information, including but not limited to, text, graphics, images and other material contained on this website are for informational purposes only. No material on this site is intended to be a substitute for professional medical advice, diagnosis or treatment. Always seek the advice of your physician or other qualified health care provider with any questions you may have regarding a medical condition or treatment and before undertaking a new health care regimen, and never disregard professional medical advice or delay in seeking it because of something you have read on this website.