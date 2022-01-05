.. lrboost documentation master file, created by
   sphinx-quickstart on Mon Jan 4 14:44:12 2022.

Welcome to LRBoost's documentation!
============================================

lrboost is an scikit-learn compatible simple stacking protocol for prediction.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   quick_start

Getting Started
-------------------------------------

LRBoostRegressor works in three steps.

* Fit a linear model to a target ``y``
* Fit a tree-based model to the residual (``y_pred - y``) of the linear model 
* Combine the two predictions into a final prediction in the scale of the original target

LRBoostRegressor defaults to ``sklearn.linear_model.RidgeCV()`` and ``sklearn.ensemble.HistGradientBoostingRegressor()`` as the linear and non-linear model respectively.

    >>> from sklearn.datasets import load_iris
    >>> from lrboost import LRBoostRegressor
    >>> X, y = load_iris(return_X_y=True)
    >>> lrb = LRBoostRegressor.fit(X, y)
    >>> predictions = lrb.predict(X)
    >>> detailed_predictions = lrb.predict_detail(X)

The linear and non-linear models are both fit in the ``fit()`` method and used to then predict on any new data. Because lrboost is a very slightly modified scklearn class, you can hyperparameter tune the tree model as you would normally.

* ``.predict(X)`` returns an array-like of final predictions
* ``.predict_detail(X)`` returns a dictionary with the base linear estimator predictions (base), tree-based predictions (resid), and then the difference of the two (pred). 
* ``.predict(X)`` and ``.predict_detail(X)['pred']`` are equivalent values
* ``.predict_proba(X)`` will provide probabilistic predictions associated with ``NGBoost`` or ``XGBoost-Distribution`` as the non-linear estimators.

Any sklearn compatible estimator can be used with LRBoost, and you can unpack kwargs as needed.

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestRegressor 
    >>> from lrboost import LRBoostRegressor
    >>> X, y = load_iris(return_X_y=True)
    >>> ridge_args = {"alphas": np.logspace(-4, 3, 10, endpoint=True),
                     "cv": 5}
    >>> rf_args = {"n_estimators": 50, 
                  "n_jobs": -1}
    >>> lrb = LRBoostRegressor(linear_model=RidgeCV(**ridge_args),
                        non_linear_model=RandomForestRegressor(**rf_args))
    >>> lrb = LRBoostRegressor.fit(X, y)
    >>> predictions = lrb.predict(X)

