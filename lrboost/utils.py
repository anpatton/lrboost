
import numpy as np
from sklearn.metrics import mean_squared_error

base_gbdt_tune_params = {'l2_regularization': 0.0, 
                        'learning_rate': 0.1, 
                        'max_bins': 255, 
                        'max_depth': None, 
                        'max_iter': 100, 
                        'max_leaf_nodes': 31, 
                        'min_samples_leaf': 20, 
                        'validation_fraction': 0.1}

def rmse(pred, actual, digits=3):
     return np.round(mean_squared_error(pred, actual, squared=False), digits)


## ADD NATE-DECORRELATOR?
