"""Visualization utilities for forecasting results."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Imbalance Price Forecast",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot actual vs predicted values.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size as (width, height)
        save_path: If provided, save the plot to this path
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(y_true))
    ax.plot(x, y_true, label='Actual', alpha=0.7, linewidth=1.5)
    ax.plot(x, y_pred, label='Predicted', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Imbalance Price')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_training_history(
    history: List[float],
    title: str = "Training Loss",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot training loss history.
    
    Args:
        history: List of loss values per epoch
        title: Plot title
        figsize: Figure size as (width, height)
        save_path: If provided, save the plot to this path
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(history) + 1)
    ax.plot(epochs, history, 'b-', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_comparison(
    y_true: np.ndarray,
    predictions_dict: dict,
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot multiple model predictions for comparison.
    
    Args:
        y_true: Ground truth values
        predictions_dict: Dictionary mapping model names to predictions
        title: Plot title
        figsize: Figure size as (width, height)
        save_path: If provided, save the plot to this path
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(y_true))
    ax.plot(x, y_true, label='Actual', color='black', 
            alpha=0.8, linewidth=2)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))
    for (name, pred), color in zip(predictions_dict.items(), colors):
        ax.plot(x, pred, label=name, color=color, 
                alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Imbalance Price')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig
