from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
import numpy as np
from sklearn.utils.validation import check_is_fitted
from typing import Dict


class LRBoostRegressor:
    def __init__(self, linear_model=None, non_linear_model=None):
        if linear_model is None:
            linear_model = RidgeCV()

        if non_linear_model is None:
            non_linear_model = HistGradientBoostingRegressor()

        self.linear_model = linear_model
        self.non_linear_model = non_linear_model
        self.non_linear_type = type(self.non_linear_model).__name__

    def __sklearn_is_fitted__(self):
        """Internal sklearn helper that indicates the object has been fitted

        Returns:
            bool: True
        """
        return True

    def fit(self, X, y, sample_weight=None):
        """Fits both the linear and non-linear estimator and returns fitted LRBoostRegressor

        Args:
            X (array-like): Input features
            y (array-like): Raw target
            sample_weight (array-like, optional): Sample weights for estimators.
                Only accepts one weight for both. Defaults to None.

        Returns:
            self: Fitted LRBoostRegressor
        """
        self.linear_model.fit(X, y, sample_weight=sample_weight)
        self.linear_prediction = self.linear_model.predict(X)
        self.linear_residual = np.subtract(self.linear_prediction, y)
        self.non_linear_model.fit(X, self.linear_residual, sample_weight=sample_weight)

        return self

    def predict(self, X, detail=False):
        """[summary]

        Args:
            X ([type]): [description]
            detail (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        check_is_fitted(self)
        linear_prediction = self.linear_model.predict(X)

        if self.non_linear_type == "NGBRegressor":
            non_linear_prediction = self.non_linear_model.pred_dist(X).loc
        elif self.non_linear_type == "XGBDistribution":
            non_linear_prediction = self.non_linear_model.predict(X).loc
        else:
            non_linear_prediction = self.non_linear_model.predict(X)

        if detail:

            preds = {
                "linear_prediction": linear_prediction,
                "non_linear_prediction": non_linear_prediction,
                "prediction": np.subtract(linear_prediction, non_linear_prediction),
            }

        else:
            preds = np.subtract(linear_prediction, non_linear_prediction)

        return preds

    def predict_dist(self, X):
        """[summary]

        Args:
            X ([type]): [description]

        Returns:
            np.array: [description]
        """
        check_is_fitted(self)

        if not self.non_linear_type in ["NGBRegressor", "XGBDistribution"]:
            raise Exception(
                "predict_dist() method requires an NGboostRegressor or XGBDistribution object"
            )

        if self.non_linear_type == "NGBRegressor":
            preds = self.non_linear_model.pred_dist(X)
            final_preds = np.add(preds.loc, self.linear_model.predict(X))
            return final_preds, preds.scale

        if self.non_linear_type == "XGBDistribution":
            preds = self.non_linear_model.predict(X)
            final_preds = np.add(preds.loc, self.linear_model.predict(X))
            return final_preds, preds.scale
