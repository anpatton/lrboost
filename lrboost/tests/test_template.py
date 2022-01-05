import pytest
import numpy as np

from sklearn.datasets import load_iris
from np.testing import assert_array_equal
from np.testing import assert_allclose

from lrboost import LRBoostRegressor

@pytest.fixture
def data():
    return load_diabetes(return_X_y=True)

def test_template_estimator(data):
    lrb = LRBoostRegressor()
    assert lrb.demo_param == 'demo_param'

    lrb.fit(*data)
    assert hasattr(lrb, 'is_fitted_')

    X = data[0]
    y_pred = lrb.predict(X)
    assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))


