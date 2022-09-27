<img src=https://raw.githubusercontent.com/anpatton/lrboost/main/doc/images/lrboost.png width=300>

[![Documentation Status](https://readthedocs.org/projects/lrboost/badge/?version=latest)](https://lrboost.readthedocs.org)

lrboost is a [sckit-learn](https://scikit-learn.org/) compatible package for linear residual boosting. lrboost uses a linear estimator to first remove any linear trends from the data, and then uses a separate non-linear estimator to model the remaining non-linear trends. We find that extrapolation tasks or data with linear and non-linear components are the best use cases. Not every modeling task will benefit from lrboost, but we use this in our own work and wanted to share something that made it easy to use.  

For a stable version, install using ``pip``:

```python
pip install lrboost
```

lrboost was inspired by ['Regression-Enhanced Random Forests' by Haozhe Zhang, Dan Nettleton, and Zhengyuan Zhu.](https://arxiv.org/abs/1904.10416v1) An excellent PyData talk by Gabby Shklovsky explaining the intuition underlying the approach may also be found here: ['Random Forest Best Practices for the Business World'](https://youtu.be/E7VLE-U07x0?t=341).

* LRBoostRegressor can be used like any other sklearn estimator and is built off a sklearn template.
* ``predict(X)`` returns an array-like of final predictions
* Adding ``predict(X, detail=True)`` returns a dictionary with primary, secondary, and final predictions.


```python
from sklearn.datasets import load_diabetes
from lrboost import LRBoostRegressor
X, y = load_diabetes(return_X_y=True)
lrb = LRBoostRegressor().fit(X, y)
predictions = lrb.predict(X)
detailed_predictions = lrb.predict(X, detail=True)

print(lrb.primary_model.score(X, y)) #R2
print(lrb.score(X, y)) #R2

>>> 0.512
>>> 0.933
```

[More detailed documentation can be found here!](https://lrboost.readthedocs.io/en/latest/) 

*[Andrew Patton](https://twitter.com/anpatt7), [Kostya Medvedovsky](https://twitter.com/kmedved), and [Nathan Walker](https://twitter.com/bbstats)*
