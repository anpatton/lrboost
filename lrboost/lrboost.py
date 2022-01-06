import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Dict


class LRBoostRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, primary_model=None, secondary_model=None):
        if primary_model is None:
            primary_model = RidgeCV()

        if secondary_model is None:
            secondary_model = HistGradientBoostingRegressor()

        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.secondary_type = type(self.secondary_model).__name__

    def __sklearn_is_fitted__(self):
        """Internal sklearn helper that indicates the object has been fitted

        Returns:
            bool: True
        """
        return True

    def fit(self, X, y, sample_weight=None):
        """Fits both the primary and secondary estimator and returns fitted LRBoostRegressor

        Args:
            X (array-like): Input features
            y (array-like): Raw target
            sample_weight (array-like, optional): Sample weights for estimators.
                Only accepts one weight for both. Defaults to None.

        Returns:
            self: Fitted LRBoostRegressor
        """
        self._fit_primary_model(X, y, sample_weight=sample_weight)
        self.primary_residual = np.subtract(self.primary_prediction, y)
        self._fit_secondary_model(X, self.primary_residual, sample_weight=sample_weight)

        return self

    def _fit_primary_model(self, X, y, sample_weight=None):
        self.primary_model.fit(X, y, sample_weight=sample_weight)
        self.primary_prediction = self.primary_model.predict(X)

    def _fit_secondary_model(self, X, y, sample_weight=None):
        self.secondary_model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X, detail=False):
        """[summary]

        Args:
            X (array-type): Input features
            detail (bool, optional):  Flag to include primary and secondary predictions.
                Defaults to False.

        Returns:
            Dict: If detail=True with primary, secondary, and final predictions.
            np.array: If detail=False just final predictions.
        """
        check_is_fitted(self)
        primary_prediction = self.primary_model.predict(X)

        if self.secondary_type == "NGBRegressor":
            secondary_prediction = self.secondary_model.pred_dist(X).loc
        elif self.secondary_type == "XGBDistribution":
            secondary_prediction = self.secondary_model.predict(X).loc
        else:
            secondary_prediction = self.secondary_model.predict(X)

        if detail:

            preds = {
                "primary_prediction": primary_prediction,
                "secondary_prediction": secondary_prediction,
                "final_prediction": np.subtract(primary_prediction, secondary_prediction),
            }

        else:
            preds = np.subtract(primary_prediction, secondary_prediction)

        return preds

    def predict_dist(self, X) -> tuple:
        """[summary]

        Args:
            X (array-like): Input features

        Raises:
            Exception: Throws error if non probabilistic model used. 

        Returns:
            tuple: final prediction, sd of secondary prediction
        """
        check_is_fitted(self)

        if not self.secondary_type in ["NGBRegressor", "XGBDistribution"]:
            raise Exception(
                "predict_dist() method requires an NGboostRegressor or XGBDistribution object"
            )

        if self.secondary_type == "NGBRegressor":
            preds = self.secondary_model.pred_dist(X)
            final_prediction = np.add(preds.loc, self.primary_model.predict(X))
            return final_prediction, preds.scale

        if self.secondary_type == "XGBDistribution":
            preds = self.secondary_model.predict(X)
            final_prediction = np.add(preds.loc, self.primary_model.predict(X))
            return final_prediction, preds.scale

    def fit_and_tune(
        self,
        X,
        y,
        tuner,
        param_distributions,
        fit_params=None,
        *tuner_args,
        **tuner_kwargs
    ):
        """[summary]

        Args:
            X (array-like): Input features
            y (array-like): Raw target
            tuner ([type]): [description]
            param_distributions ([type]): [description]
            fit_params ([type], optional): [description]. Defaults to None.
        """
        if fit_params is None:
            fit_params = {}
        self._fit_primary_model(X, y, **fit_params)
        self._tune_secondary_model(
            tuner,
            param_distributions,
            X,
            y,
            *tuner_args,
            fit_params=fit_params,
            **tuner_kwargs
        )

    def _tune_secondary_model(
        self, tuner, param_distributions, X, y, *args, fit_params=None, **kwargs
    ):
        check_is_fitted(self.primary_model)
        if fit_params is None:
            fit_params = {}

        """In lieu of implementing sklearn compatibility which would require an API change to allow frozen estimators"""
        self.secondary_model = (
            tuner(self.secondary_model, param_distributions, *args, **kwargs)
            .fit(X, y, **fit_params)
            .best_estimator_
        )
