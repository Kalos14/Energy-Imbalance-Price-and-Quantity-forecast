# Energy Imbalance Price and Quantity Forecast

This repository contains simple and complex models for forecasting energy imbalance prices using only imbalance price and quantity data.

## Models

### Simple Model (Ridge Regression)
A linear regression-based approach using lagged features:
- Uses historical price and quantity values as input features
- Ridge regularization to prevent overfitting
- Fast training and inference
- Suitable for baseline comparisons

### Complex Model (LSTM)
A deep learning approach using Long Short-Term Memory networks:
- Multi-layer LSTM architecture for capturing temporal patterns
- Dropout regularization for generalization
- Learns non-linear relationships in the data
- Better suited for complex patterns in time series

## Features

Both models use only two input features:
- **Imbalance Price**: Historical price data
- **Imbalance Quantity**: Historical quantity data

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from src.data import DataLoader, DataPreprocessor
from src.models import SimpleModel, ComplexModel

# Load data (uses synthetic data if no path provided)
loader = DataLoader()
data = loader.load_data()
train_data, test_data = loader.get_train_test_split(test_size=0.2)

# Preprocess data
preprocessor = DataPreprocessor(sequence_length=24)
X_train, y_train = preprocessor.fit_transform(train_data)
X_test, y_test = preprocessor.transform(test_data)

# Train complex model
complex_model = ComplexModel(hidden_size=64, epochs=50)
complex_model.fit(X_train, y_train)
predictions = complex_model.predict(X_test)
```

### Using the Training Script

```bash
# Train with synthetic data
python train.py

# Train with custom data
python train.py --data-path path/to/your/data.csv --epochs 100
```

### Command Line Options

```
--data-path        Path to CSV file (uses synthetic data if not provided)
--sequence-length  Number of historical timesteps to use (default: 24)
--epochs           Number of training epochs for LSTM (default: 50)
--test-size        Fraction of data for testing (default: 0.2)
```

## Data Format

If providing your own data, use a CSV file with the following columns:
- `timestamp`: DateTime column
- `imbalance_price`: Numerical price values
- `imbalance_quantity`: Numerical quantity values

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── data_loader.py      # Data loading utilities
│   │   └── preprocessor.py     # Data preprocessing
│   ├── models/
│   │   ├── simple_model.py     # Ridge Regression model
│   │   └── complex_model.py    # LSTM model
│   └── utils/
│       ├── metrics.py          # Evaluation metrics
│       └── visualization.py    # Plotting utilities
├── train.py                    # Main training script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Evaluation Metrics

Models are evaluated using:
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination