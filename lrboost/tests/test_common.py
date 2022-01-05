import pytest

from sklearn.utils.estimator_checks import check_estimator

from lrboost import LRBoostRegressor

@pytest.mark.parametrize(
    "estimator",
    [LRBoostRegressor()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
