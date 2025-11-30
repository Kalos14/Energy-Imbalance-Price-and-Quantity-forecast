#!/usr/bin/env python
"""Example script demonstrating imbalance price forecasting.

This script shows how to use both the simple (Ridge Regression) and 
complex (LSTM) models for forecasting imbalance prices using only
price and quantity data.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')

from src.data import DataLoader, DataPreprocessor
from src.models import SimpleModel, ComplexModel
from src.utils import calculate_metrics
from src.utils.metrics import print_metrics


def train_simple_model(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    preprocessor: DataPreprocessor
) -> tuple:
    """Train and evaluate the simple model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        preprocessor: Data preprocessor for inverse transform
        
    Returns:
        Tuple of (model, predictions, metrics)
    """
    print("\n" + "="*60)
    print("Training Simple Model (Ridge Regression)")
    print("="*60)
    
    model = SimpleModel(sequence_length=24, alpha=1.0)
    model.fit(X_train, y_train)
    
    predictions_scaled = model.predict(X_test)
    predictions = preprocessor.inverse_transform_price(predictions_scaled)
    y_test_orig = preprocessor.inverse_transform_price(y_test)
    
    metrics = calculate_metrics(y_test_orig, predictions)
    print_metrics(metrics, "Simple Model")
    
    return model, predictions, metrics, y_test_orig


def train_complex_model(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    preprocessor: DataPreprocessor,
    epochs: int = 50
) -> tuple:
    """Train and evaluate the complex model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        preprocessor: Data preprocessor for inverse transform
        epochs: Number of training epochs
        
    Returns:
        Tuple of (model, predictions, metrics)
    """
    print("\n" + "="*60)
    print("Training Complex Model (LSTM)")
    print("="*60)
    
    model = ComplexModel(
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        epochs=epochs,
        batch_size=32
    )
    
    model.fit(X_train, y_train, verbose=True)
    
    predictions_scaled = model.predict(X_test)
    predictions = preprocessor.inverse_transform_price(predictions_scaled)
    y_test_orig = preprocessor.inverse_transform_price(y_test)
    
    metrics = calculate_metrics(y_test_orig, predictions)
    print_metrics(metrics, "Complex Model (LSTM)")
    
    return model, predictions, metrics, y_test_orig


def main():
    """Main function to run the forecasting pipeline."""
    parser = argparse.ArgumentParser(
        description='Imbalance Price Forecasting with Simple and Complex Models'
    )
    parser.add_argument(
        '--data-path', 
        type=str, 
        default=None,
        help='Path to CSV data file (uses synthetic data if not provided)'
    )
    parser.add_argument(
        '--sequence-length', 
        type=int, 
        default=24,
        help='Number of historical timesteps to use (default: 24)'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50,
        help='Number of training epochs for LSTM (default: 50)'
    )
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Imbalance Price Forecasting")
    print("Using only Price and Quantity data")
    print("="*60)
    
    print("\nLoading data...")
    loader = DataLoader(filepath=args.data_path)
    data = loader.load_data()
    print(f"Loaded {len(data)} samples")
    print(f"Features: {list(data.columns)}")
    
    train_data, test_data = loader.get_train_test_split(test_size=args.test_size)
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    
    preprocessor = DataPreprocessor(
        sequence_length=args.sequence_length,
        forecast_horizon=1
    )
    
    # Fit preprocessor on training data
    preprocessor.fit(train_data)
    
    # Get flat features for simple model
    X_train_flat, y_train = preprocessor.create_features_flat(train_data)
    X_test_flat, y_test = preprocessor.create_features_flat(test_data)
    
    simple_model, simple_pred, simple_metrics, y_test_orig = train_simple_model(
        X_train_flat, y_train, X_test_flat, y_test, preprocessor
    )
    
    # Get sequence features for complex model
    X_train_seq, y_train_seq = preprocessor.transform(train_data)
    X_test_seq, y_test_seq = preprocessor.transform(test_data)
    
    complex_model, complex_pred, complex_metrics, _ = train_complex_model(
        X_train_seq, y_train_seq, X_test_seq, y_test_seq, 
        preprocessor, epochs=args.epochs
    )
    
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    print(f"\n{'Metric':<10} {'Simple Model':<15} {'Complex Model':<15}")
    print("-" * 40)
    
    for metric in ['rmse', 'mae', 'mape', 'r2']:
        simple_val = simple_metrics[metric]
        complex_val = complex_metrics[metric]
        
        if metric == 'mape':
            print(f"{metric.upper():<10} {simple_val:<14.2f}% {complex_val:<14.2f}%")
        else:
            print(f"{metric.upper():<10} {simple_val:<15.4f} {complex_val:<15.4f}")
    
    print("\nForecasting complete!")
    
    return {
        'simple': {'model': simple_model, 'metrics': simple_metrics},
        'complex': {'model': complex_model, 'metrics': complex_metrics}
    }


if __name__ == '__main__':
    main()
