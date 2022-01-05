.. -*- mode: rst -*-

|ReadTheDocs|_

.. |ReadTheDocs| image:: https://readthedocs.org/projects/lrboost/badge/?version=latest
.. _ReadTheDocs: https://lrboost.readthedocs.io/en/latest/?badge=latest

LRBoost - Linear Residual Boosting with scikit-learn
============================================================

.. _scikit-learn: https://scikit-learn.org

**LRBoost** is a combination of a linear estimator and gradient boosting that is scikit-learn_ compatible. 

.. _ReadTheDocs: https://lrboost.readthedocs.io/en/latest/

* LRBoostRegressor can be used like any other sklearn estimator and is built off a sklearn template.
* ``predict`` returns an array-like of final predictions
* ``predict_detail`` returns a dictionary with the base linear estimator predictions (base), tree-based predictions (resid), and then the difference of the two (pred). 
* ``predict(X)`` and ``predict_detail(X)['pred']`` are equivalent values

    >>> from sklearn.datasets import load_iris
    >>> from lrboost import LRBoostRegressor
    >>> X, y = load_iris(return_X_y=True)
    >>> lrb = LRBoostRegressor.fit(X, y)
    >>> predictions = lrb.predict(X)
    >>> detailed_predictions = lrb.predict_detail(X)

More detailed documentation can be found here -> https://readthedocs.org/projects/lrboost. 

*Andrew Patton & Kostya Medvedovsky*
