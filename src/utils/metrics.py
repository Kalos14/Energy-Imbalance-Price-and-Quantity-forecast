"""Evaluation metrics for forecasting models."""

import numpy as np
from typing import Dict


def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate standard regression metrics for forecast evaluation.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing:
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error
            - mape: Mean Absolute Percentage Error
            - r2: R-squared score
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('inf')
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2)
    }


def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metric names and values
        model_name: Name of the model for display
    """
    print(f"\n{'='*50}")
    print(f"Evaluation Metrics for {model_name}")
    print(f"{'='*50}")
    print(f"MSE:  {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"{'='*50}\n")
