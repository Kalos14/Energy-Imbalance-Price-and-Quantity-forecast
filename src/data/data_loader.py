"""Data loading utilities for imbalance price and quantity data."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


class DataLoader:
    """Load and prepare imbalance price and quantity data for forecasting.
    
    This class handles loading data from CSV files or creating synthetic data
    for testing purposes. It uses only imbalance price and quantity as features.
    """
    
    def __init__(self, filepath: Optional[str] = None):
        """Initialize the DataLoader.
        
        Args:
            filepath: Path to CSV file containing the data. If None, synthetic
                      data will be generated.
        """
        self.filepath = filepath
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load data from file or generate synthetic data.
        
        Returns:
            DataFrame with columns: timestamp, imbalance_price, imbalance_quantity
        """
        if self.filepath is not None:
            self.data = pd.read_csv(self.filepath, parse_dates=['timestamp'])
        else:
            self.data = self._generate_synthetic_data()
        return self.data
    
    def _generate_synthetic_data(self, n_samples: int = 8760) -> pd.DataFrame:
        """Generate synthetic imbalance price and quantity data.
        
        Creates realistic synthetic data with seasonal patterns, trends,
        and noise to simulate real energy imbalance data.
        
        Args:
            n_samples: Number of hourly samples to generate (default: 1 year)
            
        Returns:
            DataFrame with synthetic data
        """
        np.random.seed(42)
        
        timestamps = pd.date_range(
            start='2023-01-01', 
            periods=n_samples, 
            freq='h'
        )
        
        hours = np.arange(n_samples)
        daily_pattern = np.sin(2 * np.pi * hours / 24)
        weekly_pattern = np.sin(2 * np.pi * hours / (24 * 7))
        yearly_pattern = np.sin(2 * np.pi * hours / (24 * 365))
        
        base_price = 50
        price_trend = hours * 0.001
        price_noise = np.random.normal(0, 10, n_samples)
        imbalance_price = (
            base_price + 
            15 * daily_pattern + 
            10 * weekly_pattern + 
            20 * yearly_pattern +
            price_trend + 
            price_noise
        )
        imbalance_price = np.clip(imbalance_price, -50, 200)
        
        base_quantity = 100
        quantity_correlation = -0.3 * (imbalance_price - base_price)
        quantity_noise = np.random.normal(0, 30, n_samples)
        imbalance_quantity = (
            base_quantity + 
            50 * daily_pattern + 
            30 * weekly_pattern +
            quantity_correlation +
            quantity_noise
        )
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'imbalance_price': imbalance_price,
            'imbalance_quantity': imbalance_quantity
        })
        
        return data
    
    def get_train_test_split(
        self, 
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets.
        
        Uses time-based split to maintain temporal order.
        
        Args:
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if self.data is None:
            self.load_data()
            
        split_idx = int(len(self.data) * (1 - test_size))
        train_data = self.data.iloc[:split_idx].copy()
        test_data = self.data.iloc[split_idx:].copy()
        
        return train_data, test_data
