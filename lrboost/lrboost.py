from typing import Dict
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.utils.validation import (
    check_is_fitted,
    _check_sample_weight,
    has_fit_parameter,
)
from sklearn.pipeline import make_pipeline
from ngboost import NGBRegressor

DEFAULT_PRIMARY_MODEL = make_pipeline(
    StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3))
)
DEFAULT_SECONDARY_MODEL = HistGradientBoostingRegressor(
    early_stopping=True, max_iter=1_000, random_state=42
)
DEFAULT_SECONDARY_MODEL_DIST = NGBRegressor(
    Base=DecisionTreeRegressor(
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=75,
        splitter="best",
        random_state=42,
        max_depth=10,
    ),
    random_state=42,
    learning_rate=0.01,
    n_estimators=1000,
    verbose_eval=100,
)


def _lrb_validate_X(X, feature_set):
    if feature_set is not None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("If manually setting features per model, X must be a pandas DataFrame.")
        return X.loc[:, feature_set]
    else:
        return X

def _pull_out_prefixed_keys(d, prefix):
    return {
            k.replace(prefix, ""): v
            for k, v in d.items()
            if k.startswith(prefix)
        }

class LRBoostRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, primary_model=None, secondary_model=None):
        if primary_model is None:
            primary_model = DEFAULT_PRIMARY_MODEL
        if secondary_model is None:
            secondary_model = DEFAULT_SECONDARY_MODEL

        self.primary_model = primary_model
        self.secondary_model = secondary_model

    def fit(self, X, y, **fit_params):
        if fit_params is None:
            primary_fit_params = {}
            secondary_fit_params = {}
        # any kwarg with prefix "primary_model__" will be passed to primary model
        # any kwarg with prefix "secondary_model__" will be passed to secondary model
        primary_fit_params = _pull_out_prefixed_keys(fit_params, "primary_model__")
        secondary_fit_params = _pull_out_prefixed_keys(fit_params, "secondary_model__")

        self._fit_primary_model(X, y, **primary_fit_params)
        primary_residual = y - self.primary_prediction

        self._fit_secondary_model(X, primary_residual, **secondary_fit_params)
        self.fitted_ = True
        return self

    def _validate_primary_X(self, X):
        return _lrb_validate_X(X, self.primary_features_)
    
    def _validate_secondary_X(self, X):
        return _lrb_validate_X(X, self.secondary_features_)

    def _fit_primary_model(self, X, y, **fit_params):
        self.primary_features_ = fit_params.pop('features', None)

        _X = self._validate_primary_X(X)
        self.primary_model.fit(_X, y, **fit_params)

        self.primary_prediction = self.primary_model.predict(_X).reshape(-1,1)
        

    def _fit_secondary_model(self, X, y, **fit_params):
        self.secondary_features_ = fit_params.pop('features', None)
        
        _X = self._validate_secondary_X(X)
        self.secondary_model.fit(_X, y, **fit_params)

    def predict(self, X, detail=False):
        check_is_fitted(self)

        primary_X = self._validate_primary_X(X)
        secondary_X = self._validate_secondary_X(X)

        primary_prediction = self.primary_model.predict(primary_X).reshape(-1,1)
        secondary_prediction = self.secondary_model.predict(secondary_X).reshape(-1,1)
        final_prediction = primary_prediction + secondary_prediction
        if detail:
            prediction_dict = {
                "final_prediction": final_prediction,
                "primary_prediction": primary_prediction,
                "secondary_prediction": secondary_prediction,
            }
            return prediction_dict
        else:
            return final_prediction


class LRBoostRegressorDist(LRBoostRegressor):
    def __init__(self, primary_model=None, secondary_model=None):
        if primary_model is None:
            primary_model = DEFAULT_PRIMARY_MODEL
        if secondary_model is None:
            secondary_model = DEFAULT_SECONDARY_MODEL_DIST
        if not type(secondary_model).__name__ in ["NGBRegressor", "XGBDistribution"]:
            raise Exception(
                "pred_dist() method requires an NGBoostRegressor or XGBDistribution object"
            )

        self.secondary_type = type(secondary_model).__name__
        super().__init__(primary_model=primary_model, secondary_model=secondary_model)

    def pred_dist(self, X, detail=False) -> Dict:
        """Creates final predictions from primary and secondary models.
            Models must be NGBoost or XGBoost-Distribution. Be extremely careful
            with interpretation of the secondary model variance.
        Args:
            X (array-like): Input features
        Raises:
            Exception: Throws error if non-probabilistic model used.
        Returns:
            Dict: final prediction, sd of secondary prediction
        """
        check_is_fitted(self)
        primary_prediction = self.primary_model.predict(X)
        if self.secondary_type == "NGBRegressor":
            secondary_prediction = self.secondary_model.pred_dist(X)

        if self.secondary_type == "XGBDistribution":
            secondary_prediction = self.secondary_model.predict(X)

        final_prediction = np.add(secondary_prediction.loc, primary_prediction)

        if detail:
            prediction_dict = {
                "final_pred": final_prediction,
                "primary_pred": primary_prediction,
                "secondary_pred": secondary_prediction.loc,
                "secondary_variance": secondary_prediction.scale,
            }

        else:
            prediction_dict = {
                "final_pred": final_prediction,
                "secondary_variance": secondary_prediction.scale,
            }
        return prediction_dict
