# Clozapine Toxicity Predictor

[![DOI](images/zenodo.11048224.png)](https://doi.org/10.5281/zenodo.11048224)

This [Julia](https://julialang.org/) toolbox allows predicting [clozapine](https://en.wikipedia.org/wiki/Clozapine) (CLO) and [norclozapine](https://en.wikipedia.org/wiki/Desmethylclozapine) (NCLO) blood concentrations.

Individual recommended safe dose range can also be predicted:

![](images/dose-level.png)

Adjusted clozapine level is the classifier prediction modified by predicted norclozapine concentration.

Toxic clozapine level has been defined as > 550 ng/mL, recommended therapeutic concentration has been defined as > 250 ng/mL [source: [10.1192/bjp.2023.27](https://doi.org/10.1192/bjp.2023.27)].

## Performance

The model are actively developed and we expect its accuracy to improve.

Current model performance measures (training and testing) are in the `reports` directory.

## Quickstart

Install the latest Julia version from [https://julialang.org/downloads/](https://julialang.org/downloads/) (version ≥1.11 is required).

Clone this repository, go to its folder and run to install required packages (this has to be done only once):

```sh
julia src/install.jl
```

Start the server:

```sh
julia src/install.jl
```

The server is listening on port 8080. With your web browser, go to [http://localhost:8080](http://localhost:8080), enter patient's data and click the "PREDICT" button.

![](images/webapp.png)

Alternatively, the server is also available online at [https://csk.umed.pl/clotop](https://csk.umed.pl/clotop).

Conversion factors from ng/mL to nmol/L for clozapine and norclozapine are 3.06 and 3.20, respectively (source: [https://testguide.adhb.govt.nz/EGuide/?elv=1&name=Clozapine&pn=1672&mn=833&sd=3&ts=148a090cbe5](https://testguide.adhb.govt.nz/EGuide/?elv=1&name=Clozapine&pn=1672&mn=833&sd=3&ts=148a090cbe5)).

## How to Cite

If you use this tool, please acknowledge us by citing our [paper](https://doi.org/10.1016/j.psychres.2024.116256).

```bibtex
@article{wysokinski_2024,
    title = {Clozapine Toxicity Predictor: deep neural network model predicting clozapine toxicity and its therapeutic dose range},
    journal = {Psychiatry Research},
    pages = {116256},
    year = {2024},
    issn = {0165-1781},
    doi = {https://doi.org/10.1016/j.psychres.2024.116256},
    url = {https://www.sciencedirect.com/science/article/pii/S0165178124005419},
    author = {Adam Wysokiński and Joanna Dreczka},
}
```

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