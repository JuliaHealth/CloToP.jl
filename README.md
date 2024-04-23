# Clozapine Toxicity Predictor

Current status of the regressor model training accuracy:

![](rr_train_accuracy.png)

Current status of the regressor model testing accuracy:

![](rr_test_accuracy.png)

Individual recommended save dose range is also predicted:

![](dose-level.png)

## Quickstart

```sh
julia src/server.jl
```

Next, go to the local website at [http://localhost:8080](http://localhost:8080), enter patients data and click the "PREDICT" button.