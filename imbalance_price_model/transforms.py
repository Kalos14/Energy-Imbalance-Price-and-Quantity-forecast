"""
Transformation functions for stationarity.

This module implements the arcsinh(x/std)/std transformation
to make timeseries data stationary for ARMA modeling.
"""

import numpy as np
from typing import Tuple


def arcsinh_transform(
    data: np.ndarray, 
    std: float = None
) -> Tuple[np.ndarray, float]:
    """
    Apply arcsinh(x/std)/std transformation to make data stationary.
    
    The transformation is: arcsinh(x/std) / std
    
    This transformation is particularly useful for financial data with
    heavy tails and potential negative values, as it:
    - Handles both positive and negative values
    - Reduces the impact of extreme values
    - Helps achieve stationarity for ARMA modeling
    
    Parameters
    ----------
    data : np.ndarray
        Input timeseries data (can contain negative values)
    std : float, optional
        Standard deviation to use for scaling. If None, computed from data.
        
    Returns
    -------
    transformed_data : np.ndarray
        Transformed data using arcsinh(x/std)/std
    std : float
        The standard deviation used (for inverse transformation)
        
    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([100, -50, 200, -100, 150])
    >>> transformed, std = arcsinh_transform(data)
    >>> print(f"Original std: {np.std(data):.2f}, Used std: {std:.2f}")
    """
    data = np.asarray(data, dtype=np.float64)
    
    if std is None:
        std = np.std(data)
        # Avoid division by zero
        if std == 0:
            std = 1.0
    
    # Apply transformation: arcsinh(x/std) / std
    transformed = np.arcsinh(data / std) / std
    
    return transformed, std


def inverse_arcsinh_transform(
    transformed_data: np.ndarray, 
    std: float
) -> np.ndarray:
    """
    Apply inverse arcsinh transformation to recover original scale.
    
    The inverse transformation is: std * sinh(std * x)
    
    Parameters
    ----------
    transformed_data : np.ndarray
        Data that was transformed using arcsinh_transform
    std : float
        The standard deviation used in the original transformation
        
    Returns
    -------
    original_data : np.ndarray
        Data in original scale
        
    Examples
    --------
    >>> import numpy as np
    >>> original = np.array([100, -50, 200, -100, 150])
    >>> transformed, std = arcsinh_transform(original)
    >>> recovered = inverse_arcsinh_transform(transformed, std)
    >>> np.allclose(original, recovered)
    True
    """
    transformed_data = np.asarray(transformed_data, dtype=np.float64)
    
    # Inverse transformation: std * sinh(std * x)
    original = std * np.sinh(std * transformed_data)
    
    return original
