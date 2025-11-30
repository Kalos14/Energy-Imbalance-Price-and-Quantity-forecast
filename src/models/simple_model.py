"""Simple model for imbalance price forecasting.

This module implements a simple linear regression-based model for 
predicting imbalance prices using only price and quantity data.
"""

import numpy as np
from typing import Optional, Dict, Any
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator


class SimpleModel:
    """Simple linear model for imbalance price forecasting.
    
    Uses Ridge Regression with lagged features from imbalance price
    and quantity data to predict future prices.
    
    Attributes:
        sequence_length: Number of historical timesteps used as features
        model: The underlying sklearn model
    """
    
    def __init__(
        self, 
        sequence_length: int = 24,
        alpha: float = 1.0
    ):
        """Initialize the simple model.
        
        Args:
            sequence_length: Number of lagged values to use as features
            alpha: Regularization strength for Ridge regression
        """
        self.sequence_length = sequence_length
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.is_fitted = False
        
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> 'SimpleModel':
        """Fit the model on training data.
        
        Args:
            X: Feature matrix of shape (n_samples, sequence_length * 2)
               containing flattened price and quantity sequences
            y: Target values of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict imbalance prices.
        
        Args:
            X: Feature matrix of shape (n_samples, sequence_length * 2)
            
        Returns:
            Predicted prices of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return {
            'sequence_length': self.sequence_length,
            'alpha': self.alpha,
            'coefficients': self.model.coef_ if self.is_fitted else None,
            'intercept': self.model.intercept_ if self.is_fitted else None
        }
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance based on coefficient magnitudes.
        
        Returns:
            Array of absolute coefficient values, or None if not fitted
        """
        if not self.is_fitted:
            return None
        return np.abs(self.model.coef_)
