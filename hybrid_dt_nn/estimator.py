import logging
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tensorflow import keras

# Setup standard logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Disable optuna output by default
optuna.logging.set_verbosity(optuna.logging.WARNING)

class HybridTreeRegressor(RegressorMixin, BaseEstimator):
    """
    A Hybrid Decision Tree + Neural Network Regressor.
    
    This model first fits a Decision Tree to partition the data into leaves.
    For each leaf that contains at least `nn_min_samples`, it trains a specialized
    Keras Neural Network. If a leaf has fewer samples, it falls back to the DT's prediction.
    """
    def __init__(self, 
                 dt_max_depth: int = 5, 
                 dt_min_samples_leaf: int = 50, 
                 nn_min_samples: int = 50,
                 use_hpo: bool = False,
                 hpo_trials: int = 20,
                 nn_hidden_layers: int = 2,
                 nn_units: int = 64,
                 nn_epochs: int = 30,
                 nn_batch_size: int = 32,
                 random_state: Optional[int] = None,
                 verbose: int = 0):
        # Keep constructor absolutely pure to comply with sklearn get_params()
        self.dt_max_depth = dt_max_depth
        self.dt_min_samples_leaf = dt_min_samples_leaf
        self.nn_min_samples = nn_min_samples
        self.use_hpo = use_hpo
        self.hpo_trials = hpo_trials
        self.nn_hidden_layers = nn_hidden_layers
        self.nn_units = nn_units
        self.nn_epochs = nn_epochs
        self.nn_batch_size = nn_batch_size
        self.random_state = random_state
        self.verbose = verbose
        
    def _validate_params(self) -> None:
        """Validate constructor parameters before fitting."""
        if self.dt_max_depth <= 0:
            raise ValueError("dt_max_depth must be > 0.")
        if self.dt_min_samples_leaf <= 0:
            raise ValueError("dt_min_samples_leaf must be > 0.")
        if self.nn_min_samples <= 0:
            raise ValueError("nn_min_samples must be > 0.")
        if self.hpo_trials <= 0:
            raise ValueError("hpo_trials must be > 0.")
        if self.nn_hidden_layers <= 0:
            raise ValueError("nn_hidden_layers must be > 0.")
        if self.nn_units <= 0:
            raise ValueError("nn_units must be > 0.")
        if self.nn_epochs <= 0:
            raise ValueError("nn_epochs must be > 0.")
        if self.nn_batch_size <= 0:
            raise ValueError("nn_batch_size must be > 0.")

    def fit(self, X: Any, y: Any) -> "HybridTreeRegressor":
        """Fit the model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        self._validate_params()
        
        if self.verbose > 0:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
            
        # Enforce reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)
            tf.random.set_seed(self.random_state)
            random.seed(self.random_state)

        # Sklearn robust validation
        X, y = check_X_y(X, y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of shape (n_samples,).")
        self.n_features_in_ = X.shape[1]
        
        # Internal state
        self.dt_ = None
        self.leaf_models_ = {}
        self.leaf_sample_counts_ = {}
        self.leaf_metrics_ = {}
        self.leaf_ids_ = []
        
        logger.info("1. Fitting Decision Tree...")
            
        self.dt_ = DecisionTreeRegressor(
            max_depth=self.dt_max_depth,
            min_samples_leaf=self.dt_min_samples_leaf,
            random_state=self.random_state
        )
        self.dt_.fit(X, y)
        
        # Get leaf ids for each training sample
        leaf_ids = self.dt_.apply(X)
        self.leaf_ids_ = np.unique(leaf_ids)
        
        logger.info(f"Decision Tree built with {len(self.leaf_ids_)} leaves.")
        logger.info(f"2. Training Neural Networks (use_hpo={self.use_hpo})...")
            
        for i, leaf_id in enumerate(self.leaf_ids_, 1):
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, 'dt_')
        return self.dt_.predict(X)
