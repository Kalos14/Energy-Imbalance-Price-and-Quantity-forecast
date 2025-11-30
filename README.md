# Energy-Imbalance-Price-and-Quantity-forecast

A Python package for forecasting energy imbalance prices using ARMA (AutoRegressive Moving Average) models with arcsinh transformation for stationarity.

## Features

- **ARMA Model**: Implements ARMA(p,q) models for time series forecasting
- **Arcsinh Transformation**: Uses `arcsinh(x/std)/std` transformation to achieve stationarity
- **Stationarity Testing**: Built-in ADF and KPSS tests to verify stationarity
- **Confidence Intervals**: Support for prediction confidence intervals
- **Flexible API**: Works with both NumPy arrays and Pandas Series

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from imbalance_price_model import ImbalancePriceARMA

# Sample imbalance price data
prices = np.random.randn(200).cumsum() + 50

# Create and fit the model
model = ImbalancePriceARMA(p=2, q=1)
model.fit(prices)

# Make predictions
forecast = model.predict(steps=10)
print(f"Next 10 prices: {forecast}")

# Get predictions with confidence intervals
forecast, lower, upper = model.predict(steps=5, return_conf_int=True)
```

## Transformation for Stationarity

The package uses the arcsinh transformation to make the time series stationary:

```
transformed = arcsinh(x / std) / std
```

This transformation is particularly suited for energy imbalance prices because it:
- Handles both positive and negative values
- Reduces the impact of extreme values (heavy tails)
- Is invertible for recovering original scale predictions

### Using the transformation directly

```python
from imbalance_price_model import arcsinh_transform, inverse_arcsinh_transform
import numpy as np

data = np.array([100, -50, 200, -100, 150])
transformed, std = arcsinh_transform(data)

# Recover original data
recovered = inverse_arcsinh_transform(transformed, std)
```

## Model API

### ImbalancePriceARMA

```python
model = ImbalancePriceARMA(p=1, q=1)
```

**Parameters:**
- `p`: Order of the AR (autoregressive) component (default: 1)
- `q`: Order of the MA (moving average) component (default: 1)

**Methods:**
- `fit(data, std=None)`: Fit the model to data
- `predict(steps=1, return_conf_int=False, alpha=0.05)`: Forecast future values
- `get_fitted_values()`: Get in-sample fitted values
- `get_residuals(in_original_scale=False)`: Get model residuals
- `check_stationarity(data=None, significance=0.05)`: Test for stationarity
- `summary()`: Get model summary

**Properties:**
- `aic`: Akaike Information Criterion
- `bic`: Bayesian Information Criterion

## Testing

Run the test suite with:

```bash
pytest tests/ -v
```

## Requirements

- numpy >= 1.21.0
- pandas >= 1.3.0
- statsmodels >= 0.13.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- pytest >= 7.0.0 (for testing)

## License

MIT License