"""
===========================
Plotting LRBoostRegressor 
===========================

"""
import numpy as np
from matplotlib import pyplot as plt
from lrboost import LRBoostRegressor

X = np.arange(100).reshape(100, 1)
y = np.zeros((100, ))
lrb = LRBoostRegressor()
lrb.fit(X, y)
plt.plot(lrb.predict(X))
plt.show()
