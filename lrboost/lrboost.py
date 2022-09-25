import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.utils.validation import check_is_fitted, _check_sample_weight, has_fit_parameter
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_PRIMARY_MODEL = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3)))
DEFAULT_SECONDARY_MODEL = HistGradientBoostingRegressor(
    early_stopping=True, max_iter=1_000, random_state=42
)

class LRBoostRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, primary_model=None, secondary_model=None):
        self.default_primary = primary_model is None
        self.default_secondary = secondary_model is None

        if self.default_primary:
            primary_model = DEFAULT_PRIMARY_MODEL
        if self.default_secondary:
            secondary_model = DEFAULT_SECONDARY_MODEL

        self.primary_model = primary_model
        self.secondary_model = secondary_model

    def fit(self, X, y, primary_fit_params=None, secondary_fit_params=None):
        if primary_fit_params is None:
            primary_fit_params = {}
        if secondary_fit_params is None:
            secondary_fit_params = {}

        self._fit_primary_model(
            X, y, **primary_fit_params
        )
        primary_residual = y - self.primary_prediction
        self._fit_secondary_model(X, primary_residual, **secondary_fit_params)
        self.fitted_ = True
        return self

    def _fit_primary_model(self, X, y, **fit_params):
       
        self.primary_model.fit(X, y, **fit_params)
        self.primary_prediction = self.primary_model.predict(X)

    def _fit_secondary_model(self, X, y, **fit_params):
        self.secondary_model.fit(X, y, **fit_params)

    def predict(self, X):
        check_is_fitted(self)

        primary_prediction = self.primary_model.predict(X)
        secondary_prediction = self.secondary_model.predict(X)
        return primary_prediction + secondary_prediction