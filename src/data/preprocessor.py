"""Data preprocessing utilities for time series forecasting."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Preprocess imbalance data for model training and prediction.
    
    Handles feature engineering and normalization using only imbalance
    price and quantity data.
    """
    
    def __init__(self, sequence_length: int = 24, forecast_horizon: int = 1):
        """Initialize the preprocessor.
        
        Args:
            sequence_length: Number of historical timesteps to use as input
            forecast_horizon: Number of timesteps ahead to predict
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.price_scaler = StandardScaler()
        self.quantity_scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame) -> 'DataPreprocessor':
        """Fit the preprocessor on training data.
        
        Args:
            data: DataFrame with imbalance_price and imbalance_quantity columns
            
        Returns:
            Self for method chaining
        """
        self.price_scaler.fit(data[['imbalance_price']])
        self.quantity_scaler.fit(data[['imbalance_quantity']])
        self.is_fitted = True
        return self
        
    def transform(
        self, 
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform data into sequences for model input.
        
        Creates sliding window sequences using only price and quantity features.
        
        Args:
            data: DataFrame with imbalance_price and imbalance_quantity columns
            
        Returns:
            Tuple of (X, y) where:
                X: Input sequences of shape (n_samples, sequence_length, 2)
                y: Target values of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        scaled_price = self.price_scaler.transform(data[['imbalance_price']])
        scaled_quantity = self.quantity_scaler.transform(
            data[['imbalance_quantity']]
        )
        
        features = np.column_stack([scaled_price, scaled_quantity])
        
        X, y = [], []
        for i in range(len(features) - self.sequence_length - self.forecast_horizon + 1):
            X.append(features[i:i + self.sequence_length])
            y.append(scaled_price[i + self.sequence_length + self.forecast_horizon - 1, 0])
            
        return np.array(X), np.array(y)
    
    def fit_transform(
        self, 
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform data in one step.
        
        Args:
            data: DataFrame with imbalance_price and imbalance_quantity columns
            
        Returns:
            Tuple of (X, y) sequences
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform_price(self, scaled_price: np.ndarray) -> np.ndarray:
        """Convert scaled price predictions back to original scale.
        
        Args:
            scaled_price: Normalized price values
            
        Returns:
            Price values in original scale
        """
        return self.price_scaler.inverse_transform(
            scaled_price.reshape(-1, 1)
        ).flatten()
    
    def create_features_flat(
        self, 
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create flattened features for simple models.
        
        Creates a feature matrix suitable for sklearn models by flattening
        the sequence data. Includes lagged price and quantity values.
        
        Args:
            data: DataFrame with imbalance_price and imbalance_quantity columns
            
        Returns:
            Tuple of (X, y) where X has shape (n_samples, sequence_length * 2)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        scaled_price = self.price_scaler.transform(data[['imbalance_price']])
        scaled_quantity = self.quantity_scaler.transform(
            data[['imbalance_quantity']]
        )
        
        features = np.column_stack([scaled_price, scaled_quantity])
        
        X, y = [], []
        for i in range(len(features) - self.sequence_length - self.forecast_horizon + 1):
            X.append(features[i:i + self.sequence_length].flatten())
            y.append(scaled_price[i + self.sequence_length + self.forecast_horizon - 1, 0])
            
        return np.array(X), np.array(y)
