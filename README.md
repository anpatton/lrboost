<img src=https://github.com/anpatton/lrboost/blob/main/doc/images/lrboost.png width=300>

[![Documentation Status](https://readthedocs.org/projects/lrboost/badge/?version=latest)](https://lrboost.readthedocs.org)

LRBoost is a [sckit-learn](https://scikit-learn.org/) compatible package for linear residual boosting. LRBoost combines a linear estimator and a non-linear estimator to leverage the strengths of both models. We're just getting started but plan to build this out as much as possible for our own use and that of the community. This very basic functionality comes standardish for [LightGBM](https://github.com/microsoft/LightGBM), but we used all sorts of different flavors of this approach so much we wanted to make something useful. 

* LRBoostRegressor can be used like any other sklearn estimator and is built off a sklearn template.
* ``.predict()`` returns an array-like of final predictions
* ``.predict_detail()`` returns a dictionary with the linear, non-linear, and final predictions.

```python
from sklearn.datasets import load_iris
from lrboost import LRBoostRegressor
X, y = load_iris(return_X_y=True)
lrb = LRBoostRegressor.fit(X, y)
predictions = lrb.predict(X)
detailed_predictions = lrb.predict_detail(X)
```

[More detailed documentation can be found here!](https://lrboost.readthedocs.io/en/latest/) 

*Andrew Patton & Kostya Medvedovsky*
