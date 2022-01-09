import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import FunctionTransformer
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

    def fit(self, X, y, sample_weight=None, primary_scaler=FunctionTransformer()):
        """Fits both the primary and secondary estimator and returns fitted LRBoostRegressor

        Args:
            X (array-like): Input features
            y (array-like): Raw target
            sample_weight (array-like, optional): Sample weights for estimators.
                Only accepts one weight for both. Defaults to None.
            primary_scaler (FunctionTransformer): Scaling function from sklearn. Defaults to FunctionTransformer().

        Returns:
            self: Fitted LRBoostRegressor
        """
        self._fit_primary_model(
            X, y, primary_scaler=primary_scaler, sample_weight=sample_weight
        )
        self.primary_residual = np.subtract(self.primary_prediction, y)
        self._fit_secondary_model(X, self.primary_residual, sample_weight=sample_weight)

        return self

    def _fit_primary_model(self, X, y, primary_scaler, sample_weight=None):
        X_scaled = primary_scaler.fit_transform(X)
        self.primary_model.fit(X_scaled, y, sample_weight=sample_weight)
        self.primary_prediction = self.primary_model.predict(X_scaled)
        self.primary_scaler = primary_scaler

    def _fit_secondary_model(self, X, y, sample_weight=None):
        self.secondary_model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X, detail=False):
        """Creates final predictions from primary and secondary models.

        Args:
            X (array-type): Input features
            detail (bool, optional):  Flag to include primary and secondary predictions.
                Defaults to False.

        Returns:
            Dict: If detail=True with primary, secondary, and final predictions.
            np.array: If detail=False just final predictions.
        """
        check_is_fitted(self)
        X_scaled = self.primary_scaler.transform(X)
        primary_prediction = self.primary_model.predict(X_scaled)

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
                "final_prediction": np.subtract(
                    primary_prediction, secondary_prediction
                ),
            }

        else:
            preds = np.subtract(primary_prediction, secondary_prediction)

        return preds

    def predict_dist(self, X) -> tuple:
        """Creates final predictions from primary and secondary models.
            Models must be NGBoost or XGBoost-Distribution. Be careful
            with interpretation of the secondary model variance.

        Args:
            X (array-like): Input features

        Raises:
            Exception: Throws error if non-probabilistic model used.

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
        primary_scaler=FunctionTransformer(),
        sample_weight=None,
        primary_fit_params=None,
        secondary_fit_params=None,
        *tuner_args,
        **tuner_kwargs
    ):
        """[summary]

        Args:
            X ([type]): [description]
            y ([type]): [description]
            tuner ([type]): [description]
            param_distributions ([type]): [description]
            primary_scaler (FunctionTransformer): Scaling function from sklearn. Defaults to FunctionTransformer().
            sample_weight ([type], optional): [description]. Defaults to None.
            primary_fit_params ([type], optional): [description]. Defaults to None.
            secondary_fit_params ([type], optional): [description]. Defaults to None.

        Raises:
            Exception: [description]

        Returns:
            [type]: [description]
        """
        if primary_fit_params is None:
            primary_fit_params = {}

        if (
            "sample_weight" in primary_fit_params
            or "sample_weight" in secondary_fit_params
        ) and sample_weight is not None:
            raise Exception("Conflicting sample weights.")

        self._fit_primary_model(
            X,
            y,
            primary_scaler=primary_scaler,
            sample_weight=sample_weight,
            **primary_fit_params
        )
        self._tune_secondary_model(
            tuner,
            param_distributions,
            X,
            y,
            *tuner_args,
            sample_weight=sample_weight,
            fit_params=secondary_fit_params,
            **tuner_kwargs
        )
        return self

    def _tune_secondary_model(
        self,
        tuner,
        param_distributions,
        X,
        y,
        *tuner_args,
        sample_weight=None,
        secondary_fit_params=None,
        **tuner_kwargs
    ):
        """[summary]

        Args:
            tuner ([type]): [description]
            param_distributions ([type]): [description]
            X ([type]): [description]
            y ([type]): [description]
            sample_weight ([type], optional): [description]. Defaults to None.
            secondary_fit_params ([type], optional): [description]. Defaults to None.
        """
        check_is_fitted(self.primary_model)
        if secondary_fit_params is None:
            secondary_fit_params = {}

        self.secondary_model = (
            tuner(
                self.secondary_model, param_distributions, *tuner_args, **tuner_kwargs
            )
            .fit(X, y, sample_weight=sample_weight, **secondary_fit_params)
            .best_estimator_
        )
